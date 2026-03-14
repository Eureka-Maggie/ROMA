# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from moviepy.editor import VideoFileClip
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import (
    is_librosa_available,
    is_pillow_available,
    is_pyav_available,
    is_transformers_version_greater_than,
)


if is_librosa_available():
    import librosa


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.45.0"):
    from transformers.models.mllama.processing_mllama import (
        convert_sparse_cross_attention_mask_to_dense,
        get_cross_attention_token_mask,
    )


if is_transformers_version_greater_than("4.52.0"):
    from transformers.image_utils import make_flat_list_of_images
    from transformers.video_utils import make_batched_videos
elif is_transformers_version_greater_than("4.49.0"):
    from transformers.image_utils import make_batched_videos, make_flat_list_of_images


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO]
    AudioInput = Union[str, BinaryIO, NDArray]

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            pass


def _get_paligemma_token_type_ids(imglens: list[int], seqlens: list[int], processor: "MMProcessor") -> list[list[int]]:
    r"""Get paligemma token type ids for computing loss.

    It is slightly different with the original token type ids where the prompt part is 0.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * processor.image_seq_length
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


def _get_gemma3_token_type_ids(batch_ids: list[list[int]], processor: "MMProcessor"):
    r"""Get gemma3 token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    image_token_id: int = getattr(processor, "image_token_id")
    batch_token_type_ids = []
    for token_ids in batch_ids:
        token_ids = np.array(token_ids)
        token_type_ids = np.zeros_like(token_ids)
        token_type_ids[token_ids == image_token_id] = 1
        batch_token_type_ids.append(token_type_ids.tolist())

    return batch_token_type_ids


def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: Optional["MMProcessor"],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None) # Qwen2VLImageProcessor 
        video_processor: BaseImageProcessor = getattr( # Qwen2VLImageProcessor 
            processor, "video_processor", getattr(processor, "image_processor", None)
        )
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None) # WhisperFeatureExtractor
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if self.image_token is not None and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if self.video_token is not None and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        # if len(messages) == 2: # [query_list,ans_list]
        for messages_0 in messages:
            for message in messages_0:
                for mes in message:
                    #print(mes)
                    num_image_tokens += mes["text"].count(IMAGE_PLACEHOLDER) # message["content"].count(IMAGE_PLACEHOLDER)
                    num_video_tokens += mes["text"].count(VIDEO_PLACEHOLDER) # message["content"].count(VIDEO_PLACEHOLDER)
                    num_audio_tokens += mes["text"].count(AUDIO_PLACEHOLDER) # message["content"].count(AUDIO_PLACEHOLDER)
        # else:
        #     for message in messages:
        #         #print(mes)
        #         num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
        #         num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
        #         num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            print("数出来的video有：",len(videos)," num_video_tokens:",num_video_tokens)
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(
        self, video_stream: "Stream", video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _get_video_sample_indices_2fps(
        self, container, total_frames, video_path, video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices with enforced 2fps sampling and special handling for low-fps videos."""
        # duration_in_sec = float(video_stream.duration * video_stream.time_base)
        clip = VideoFileClip(video_path)
        duration_in_sec = clip.duration
        clip.close()
        enforced_fps = video_fps

        # if total_frames == 0:  # infinite video
        #     print("视频可能有问题")
        #     return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        enforced_fps = video_fps
        sample_frames = max(1, math.floor(duration_in_sec * enforced_fps))
        
        # 检查是否超长
        if sample_frames > video_maxlen:
            # 如果超长，就从后往前取sample_frames帧
            #sample_frames = sample_frames[-video_maxlen:]
            print(f"[警告] 采样帧数 {sample_frames} 超过 video_maxlen={video_maxlen}, 取最近的{video_maxlen}帧")

        # 特别处理低fps
        if video_fps < 2.0:
            if math.isclose(video_fps, 1.0):
                indices = np.linspace(0, total_frames - 1, total_frames).astype(np.int32)
                indices = np.repeat(indices, 2)
                return indices[:video_maxlen]
            else:
                print(f"[错误] 不支持的低帧率：{video_fps}fps（只能处理 1fps）")
        # 正常采样 linspace
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    
    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            if isinstance(image, (str, BinaryIO)):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> dict[str, list[list["ImageObject"]]]:
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
            frames: list[ImageObject] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results}

    def _regularize_audios(
        self, messages, audios: list["AudioInput"], sampling_rate: float, max_length, **kwargs
    ) -> dict[str, Union[list["NDArray"], list[float]]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        target_sr = 16000
        results, sampling_rates = [], []
        if len(audios) != 0:
            for audio in audios:
                if isinstance(audio, (str, BinaryIO)):
                    audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

                if not isinstance(audio, np.ndarray):
                    raise ValueError(f"Expect input is a list of audios, but got {type(audio)}.")

                results.append(audio)
                sampling_rates.append(sampling_rate)

        elif messages and len(messages[0]) > 0:  #messages[0]是[prompt_list,ans_list]
            for mes, max_time in zip(messages,max_length):
                msg_list_for_audio = mes[0] #query
                processed_segments = []
                for msg_dict in msg_list_for_audio:
                    audio_path: Union[str, None] = msg_dict.get('audio')
                    time_val: Union[int, float, None] = msg_dict.get('time')
                    start_time_sec: float = 0.0
                    valid_time = False
                    if isinstance(time_val, (int, float)):
                        start_time_sec = float(time_val)
                        valid_time = True
                    if isinstance(audio_path, str) and valid_time:
                        segment_audio, original_sr = librosa.load(audio_path, sr=None, mono=True) # 加载音频文件，单声道
                        if original_sr != target_sr:
                            segment_audio = librosa.resample(segment_audio, orig_sr=original_sr, target_sr=target_sr)
                        
                        duration_sec = len(segment_audio) / target_sr
                        end_time_sec = start_time_sec + duration_sec

                        processed_segments.append({
                                'audio': segment_audio,      # np.ndarray
                                'start_time': start_time_sec,  # float, 秒
                                'end_time': end_time_sec,      # float, 秒
                                'sr': target_sr              # int
                            })
                    elif isinstance(audio_path, list) and valid_time: #fake_examples里的[np.zeros(1600)]
                        segment_audio = audio_path[0]
                        original_sr = target_sr

                        duration_sec = len(segment_audio) / target_sr
                        end_time_sec = start_time_sec + duration_sec

                        processed_segments.append({
                                'audio': segment_audio,      # np.ndarray
                                'start_time': start_time_sec,  # float, 秒
                                'end_time': end_time_sec,      # float, 秒
                                'sr': target_sr              # int
                            })
                if processed_segments: # 按照时间片进行排序
                    processed_segments.sort(key=lambda x: x['start_time'])
                    
                    #print(f"mm_plugin_333: 所有片段处理完毕，最晚结束时间: {max_overall_end_time_sec}s")
                    total_samples = int(max_time * target_sr) #int(max_overall_end_time_sec * target_sr)
                    if total_samples > 0:
                        # 1. 初始化最终的音频数组，填充“近乎无声的白噪音”
                        # librosa 处理的音频值域通常在 -1.0 到 1.0
                        final_audio = (np.random.uniform(-1, 1, total_samples) * 0.0001).astype(np.float32)

                        # 2. 将每个处理过的音频片段替换到 final_audio 的正确位置
                        for segment in processed_segments:
                            s_audio: np.ndarray = segment['audio']
                            s_start_sec: float = segment['start_time']

                            # 计算在 final_audio 中的绝对开始和结束采样点
                            abs_start_sample = int(s_start_sec * target_sr)
                            abs_end_sample = abs_start_sample + len(s_audio)

                            # --- 处理片段和最终音频数组之间的切片逻辑 ---
                            # 源（片段音频）的开始切片索引
                            src_slice_start = 0
                            if abs_start_sample < 0: # 如果片段的理论开始时间早于0秒
                                src_slice_start = -abs_start_sample # 从片段的这个偏移量开始取
                            
                            target_slice_start = max(0, abs_start_sample)

                            len_src_available = len(s_audio) - src_slice_start
                            len_target_available = total_samples - target_slice_start

                            len_to_copy = min(len_src_available, len_target_available)
                            
                            if len_to_copy > 0:
                                src_slice_end = src_slice_start + len_to_copy
                                target_slice_end = target_slice_start + len_to_copy
                                
                                final_audio[target_slice_start:target_slice_end] = s_audio[src_slice_start:src_slice_end]
                            
                        results.append(final_audio)
                        sampling_rates.append(target_sr)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        imglens: Optional[list[int]] = None,
    ) -> dict[str, "torch.Tensor"]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor: BaseImageProcessor = getattr(
                processor, "video_processor", getattr(processor, "image_processor", None)
            )
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 384 * 384), #256 * 256
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 1024),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[list[dict[str, str]]], #list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
        messages,
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios)

        return self._get_mm_inputs(images, videos, audios, processor, messages)


@dataclass
class Gemma3Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token

        do_pan_and_scan: bool = getattr(processor, "image_do_pan_and_scan", False)
        if do_pan_and_scan:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if do_pan_and_scan:
                    image_placeholder_str = (
                        "Here is the original image {{image}} and here are some crops to help you see better "
                        + " ".join(["{{image}}"] * mm_inputs["num_crops"][0][num_image_tokens])
                    )
                else:
                    image_placeholder_str = "{{image}}"

                content = content.replace(IMAGE_PLACEHOLDER, image_placeholder_str, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", image_str)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("num_crops", None)
        mm_inputs["token_type_ids"] = _get_gemma3_token_type_ids(batch_ids, processor)
        return mm_inputs


@dataclass
class InternVLPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "ProcessorMixin",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        image_processor_kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            image_processor_kwargs.update(
                {
                    "crop_to_patches": True,
                    "max_patches": 12,
                    "min_patches": 1,
                }
            )

        mm_inputs = {}
        image_video_patches = []

        if len(images) != 0 and isinstance(images[0], str):
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]

        if len(videos) != 0 and isinstance(videos[0], str):
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]

        if len(images) != 0:
            images = make_flat_list_of_images(images)
            image_inputs = image_processor(images=images, return_tensors="pt", **image_processor_kwargs)
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)

        if len(videos) != 0:
            videos = make_batched_videos(videos)
            num_frames_per_video = [len(video) for video in videos]
            patch_indices = np.cumsum(num_frames_per_video)
            image_processor_kwargs["crop_to_patches"] = False
            video_inputs = image_processor(images=videos, return_tensors="pt", **image_processor_kwargs)
            video_num_patches = video_inputs.pop("num_patches")
            video_pixel_values = video_inputs.pop("pixel_values")
            video_num_patches_indices = np.cumsum(video_num_patches)

        # NOT SUPPORT IMAGE VIDEO INTERLEAVED
        if len(images) != 0 and image_pixel_values is not None:
            for i in range(len(images)):
                start_index = image_num_patches_indices[i - 1] if i > 0 else 0
                end_index = image_num_patches_indices[i]
                image_video_patches.append(image_pixel_values[start_index:end_index])

        if len(videos) != 0 and video_pixel_values is not None:
            patch_indices_with_prefix = [0] + list(patch_indices)
            for i in range(len(videos)):
                current_patch_index = patch_indices_with_prefix[i]
                end_patch_index = patch_indices_with_prefix[i + 1]
                start_index = video_num_patches_indices[current_patch_index - 1] if i > 0 else 0
                end_index = video_num_patches_indices[end_patch_index - 1]
                image_video_patches.append(video_pixel_values[start_index:end_index])

        if len(images) != 0 or len(videos) != 0:
            mm_inputs["pixel_values"] = torch.cat(image_video_patches, dim=0)

        if len(images) != 0:
            mm_inputs.update({"image_num_patches": image_num_patches})

        if len(videos) != 0:
            mm_inputs.update({"video_patch_indices": patch_indices})
            mm_inputs.update({"video_num_patches": video_num_patches})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["ProcessorMixin"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        image_seqlen = getattr(processor, "image_seq_length") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        image_pixel_patch_list = mm_inputs.get("image_num_patches")  # pathes of images
        video_num_patches = mm_inputs.get("video_num_patches")  # all patches for frames of videos
        video_patch_indices = mm_inputs.get("video_patch_indices")  # num frames of per video

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<img>{'<IMG_CONTEXT>' * image_seqlen * image_pixel_patch_list[num_image_tokens]}</img>",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                current_patch_index = video_patch_indices[num_video_tokens - 1] if num_video_tokens > 0 else 0
                end_patch_index = video_patch_indices[num_video_tokens]
                num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                video_replaced_prompt = "\n".join(
                    f"Frame{i + 1}: <img>{'<IMG_CONTEXT>' * image_seqlen * num_patches[i]}</img>"
                    for i in range(len(num_patches))
                )
                content = content.replace(VIDEO_PLACEHOLDER, video_replaced_prompt, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("image_num_patches", None)
        mm_inputs.pop("video_patch_indices", None)
        mm_inputs.pop("video_num_patches", None)
        return mm_inputs


class KimiVLPlugin(BasePlugin):
    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        image_grid_hws = mm_inputs.get("image_grid_hws", [])
        num_image_tokens = 0
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        merge_length = math.prod(image_processor.merge_kernel_size)
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_hws[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|media_start|>image<|media_content|>{self.image_token * image_seqlen}<|media_end|>",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Llama4Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_height, image_width = mm_inputs["pixel_values"][0].shape[-2:]
                num_patches_per_chunk = int(
                    (image_height // processor.patch_size)
                    * (image_width // processor.patch_size)
                    // processor.downsample_ratio
                )
                aspect_ratios = mm_inputs.pop("aspect_ratios")

        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            if self.expand_mm_tokens:
                placeholder_count = content.count(IMAGE_PLACEHOLDER)
                prompt_splits = content.split(IMAGE_PLACEHOLDER)
                new_content = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_content.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = processor._prompt_split_image(
                            aspect_ratios[num_image_tokens], num_patches_per_chunk
                        )
                        num_image_tokens += 1
                        new_content.append(tokens_for_this_image)

                content = "".join(new_content)
            else:
                content = content.replace(IMAGE_PLACEHOLDER, self.image_token)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("aspect_ratios", None)
        return mm_inputs


@dataclass
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen = 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if self.expand_mm_tokens:
            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                video_seqlen = image_seqlen // 4 * num_frames  # divide by 4 needed for avg pooling layer
        else:
            video_seqlen = 1

        for message in messages:
            content = message["content"]
            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


@dataclass
class MiniCPMVPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if "valid_image_nums_ls" in kwargs:
                valid_image_nums_ls = kwargs["valid_image_nums_ls"]
                new_images = []
                idx = 0
                for valid_image_nums in valid_image_nums_ls:
                    new_images.append(images[idx : idx + valid_image_nums])
                    idx += valid_image_nums

                images = new_images

            image_inputs = image_processor(
                images, do_pad=True, max_slice_nums=image_processor.max_slice_nums, return_tensors="pt"
            )
            mm_inputs.update(image_inputs)

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            video_inputs = image_processor(videos, do_pad=True, max_slice_nums=2, return_tensors="pt")
            mm_inputs.update(video_inputs)

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            if "valid_audio_nums_ls" in kwargs:
                valid_audio_nums_ls = kwargs["valid_audio_nums_ls"]
                audios_ls = []
                idx = 0
                for valid_audio_nums in valid_audio_nums_ls:
                    audios_ls.append(audios[idx : idx + valid_audio_nums])
                    idx += valid_audio_nums
            else:
                audios_ls = [audios]

            audio_features, audio_feature_lens, audio_phs = processor.audio_feature_extract(
                audios_ls,
                chunk_input=True,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )
            audio_feature_lens = [torch.tensor(audio_feature_len) for audio_feature_len in audio_feature_lens]
            mm_inputs.update({"audio_features": audio_features, "audio_feature_lens": audio_feature_lens})
            if kwargs.get("ret_phs", False):
                mm_inputs.update({"audio_phs": audio_phs})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs, audio_inputs = {}, {}
        if len(images) != 0 and len(videos) != 0:
            raise ValueError("MiniCPM-V model does not support input images and videos at the same time.")

        if len(videos) != 0:
            max_slice_nums = 2
            use_image_id = False
            mm_inputs = self._get_mm_inputs([], videos, [], processor)
        else:
            max_slice_nums = image_processor.max_slice_nums
            use_image_id = image_processor.use_image_id

        for i, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = len(mm_inputs["pixel_values"][num_video_tokens]) if self.expand_mm_tokens else 1
                content = content.replace(VIDEO_PLACEHOLDER, "{{image}}" * video_seqlen, 1)
                num_video_tokens += 1

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, "{{audio}}", 1)
                num_audio_tokens += 1

            message["content"] = content.replace("{{image}}", "(<image>./</image>)").replace(
                "{{audio}}", "(<audio>./</audio>)"
            )

        if len(images):
            mm_inputs = self._get_mm_inputs(images, [], [], processor)

        if len(audios):
            audio_inputs = self._get_mm_inputs([], [], audios, processor, ret_phs=True)

        if self.expand_mm_tokens and mm_inputs:
            pattern = "(<image>./</image>)"
            image_sizes = mm_inputs["image_sizes"]
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                image_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(image_tags)):
                    final_text = (
                        final_text
                        + text_chunks[i]
                        + image_processor.get_slice_image_placeholder(
                            image_sizes[0][idx], idx, max_slice_nums, use_image_id
                        )
                    )
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        if self.expand_mm_tokens and audio_inputs:
            pattern = "(<audio>./</audio>)"
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                audio_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(audio_tags)):
                    audio_placeholder = audio_inputs["audio_phs"][0][idx]
                    final_text = final_text + text_chunks[i] + audio_placeholder
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        # image bound
        image_bounds_list = []
        valid_image_nums_ls = []
        for i, input_ids in enumerate(batch_ids):
            input_ids_ = torch.tensor(input_ids)
            start_cond = (input_ids_ == processor.tokenizer.im_start_id) | (
                input_ids_ == processor.tokenizer.slice_start_id
            )
            end_cond = (input_ids_ == processor.tokenizer.im_end_id) | (input_ids_ == processor.tokenizer.slice_end_id)
            image_start_tokens = torch.where(start_cond)[0]
            image_start_tokens += 1
            image_end_tokens = torch.where(end_cond)[0]
            valid_image_nums_ls.append(imglens[i])
            image_bounds = torch.hstack(
                [
                    image_start_tokens.unsqueeze(-1),
                    image_end_tokens.unsqueeze(-1),
                ]
            )
            image_bounds_list.append(image_bounds)

        mm_inputs = self._get_mm_inputs(images, videos, [], processor, valid_image_nums_ls=valid_image_nums_ls)
        if "tgt_sizes" not in mm_inputs:
            dummy_data = [torch.empty(0) for _ in range(len(batch_ids))]
            mm_inputs.update({"tgt_sizes": dummy_data, "pixel_values": dummy_data, "image_sizes": dummy_data})

        mm_inputs.update({"image_bound": image_bounds_list})

        if len(audios) > 0:
            # audio bound
            audio_bounds_ls = []
            spk_bounds_ls = []
            valid_audio_nums_ls = []

            for input_ids, audiolen in zip(batch_ids, audlens):
                input_ids_ = torch.tensor(input_ids)
                audio_start_idx = torch.where(input_ids_ == processor.tokenizer.audio_start_id)[0]
                audio_end_idx = torch.where(input_ids_ == processor.tokenizer.audio_end_id)[0]
                assert len(audio_start_idx) == len(audio_end_idx)
                audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])
                audio_bounds_ls.append(audio_bounds)
                valid_audio_nums_ls.append(audiolen)

                spk_start_idx = torch.where(input_ids_ == processor.tokenizer.spk_start_id)[0]
                spk_end_idx = torch.where(input_ids_ == processor.tokenizer.spk_end_id)[0]
                assert len(spk_start_idx) == len(spk_end_idx)
                spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
                spk_bounds_ls.append(spk_bounds)

            audio_inputs = self._get_mm_inputs([], [], audios, processor, valid_audio_nums_ls=valid_audio_nums_ls)
            mm_inputs.update(audio_inputs)
            mm_inputs.update({"audio_bounds": audio_bounds_ls, "spk_bounds": spk_bounds_ls})

        return mm_inputs


@dataclass
class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor, imglens)
        if mm_inputs:
            num_tiles = mm_inputs.pop("num_tiles")
            image_token_id: int = getattr(processor, "image_token_id")
            max_image_tiles: int = getattr(processor.image_processor, "max_image_tiles")
            cross_attention_token_mask = [
                get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
            ]
            mm_inputs["cross_attention_mask"] = torch.from_numpy(
                convert_sparse_cross_attention_mask_to_dense(
                    cross_attention_token_mask,
                    num_tiles=num_tiles,
                    max_num_tiles=max_image_tiles,
                    length=max(len(input_ids) for input_ids in batch_ids),
                )
            )  # shape: (batch_size, length, max_num_images, max_num_tiles)

        return mm_inputs


@dataclass
class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "", 1)
                num_image_tokens += 1

            message["content"] = content

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        self._validate_input(processor, images, videos, audios)
        num_images = len(images)
        image_seqlen = processor.image_seq_length if self.expand_mm_tokens else 0  # skip mm token
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * num_images * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * num_images * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


@dataclass
class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                # BC for transformers < 4.49.0
                if isinstance(mm_inputs["image_sizes"], list):
                    image_sizes = iter(mm_inputs["image_sizes"][0])
                else:
                    image_sizes = iter(mm_inputs["image_sizes"].tolist())

                image_break_token: str = getattr(processor, "image_break_token")
                image_end_token: str = getattr(processor, "image_end_token")

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    height, width = next(image_sizes)
                    num_height_tokens = height // processor.patch_size
                    num_width_tokens = width // processor.patch_size
                    replace_tokens = [[self.image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                    replace_tokens[-1] = image_end_token
                    replace_str = "".join(replace_tokens)
                else:
                    replace_str = self.image_token

                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        # ref to this commit https://github.com/huggingface/transformers/pull/35122
        # after transformers 4.49.0, the `image_sizes` is mandatory as an input parameter for Pixtral VisionEncoder forwarding.
        # it can be passed into `LlavaConditionalGeneration` as a parameter.
        if not is_transformers_version_greater_than("4.49.0"):
            mm_inputs.pop("image_sizes", None)
        return mm_inputs


@dataclass
class Qwen2AudioPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        bos_token: str = getattr(processor, "audio_bos_token")
        eos_token: str = getattr(processor, "audio_eos_token")
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs([], [], audios, processor)
            if "feature_attention_mask" in mm_inputs:
                audio_lengths = mm_inputs["feature_attention_mask"].sum(-1).tolist()

        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    audio_seqlen = (input_length - 2) // 2 + 1
                else:
                    audio_seqlen = 1

                content = content.replace(
                    AUDIO_PLACEHOLDER, f"{bos_token}{self.audio_token * audio_seqlen}{eos_token}", 1
                )

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Qwen2VLPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))
        return image

    @override
    def _regularize_videos(
        self, max_length, videos: list["VideoInput"], **kwargs
    ) -> dict[str, Union[list[list["ImageObject"]], list[float]]]:
        results, fps_per_video = [], []
        for video, max_time_for_single_video in zip(videos, max_length):
            frames: list[ImageObject] = []
            if len(video) == 1 and video[0].endswith('.mp4'): # 是一个单一video路径
                container = av.open(video[0], "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                #print("processing_video_path:",video[0])
                
                all_frames = list(container.decode(video_stream))
                total_frames = len(all_frames)
                # for frame_idx, frame in enumerate(container.decode(video_stream)):
                #     if frame_idx in sample_indices:
                #         frames.append(frame.to_image())

                sample_indices = self._get_video_sample_indices_2fps(container, total_frames, video[0] ,**kwargs) #self._get_video_sample_indices(video_stream, **kwargs)
                frames = [all_frames[idx].to_image() for idx in sample_indices if idx < total_frames]
                container.seek(0)

                if container:
                    container.close()
            else : # already a bunch of frames
                #print("processing_video_path:",video[0])
                for image in video:
                    try:
                        if isinstance(image, (str, BinaryIO)):
                            image = Image.open(image)
                        elif isinstance(image, bytes):
                            image = Image.open(BytesIO(image))
                        elif isinstance(image, dict):
                            if image["bytes"] is not None:
                                image = Image.open(BytesIO(image["bytes"]))
                            else:
                                image = Image.open(image["path"])
                    except Exception as e:
                        print(f"Error processing image {image}: {e}")
                        continue
                    frames.append(image)
                video_stream = None
            #start_time = time.time()
            frames = self._regularize_images(frames, **kwargs)["images"]
            #end_time = time.time()
            while len(frames)/2 < max_time_for_single_video: # 每次加0.5s的进去
                frames.append(frames[-1])
            if len(frames) % 2 != 0:  # qwen2-vl requires even number of frames
                frames.append(frames[-1])
            #print('video中的_regularize_images耗时：', end_time - start_time, 's')
            #print("len_frames:",len(frames))
            results.append(frames)
            if video_stream is None or video_stream.duration is None:
                fps_per_video.append(2.0)
            else:
                fps_per_video.append(2.0) #fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))
        
        #print("[DEBUG] returning videos, len:", len(frames))
        return {"videos": results, "fps_per_video": fps_per_video}

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(image_processor(images=None, videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )
                num_video_tokens += 1

            message["content"] = content

        return messages


class Qwen2OmniPlugin(Qwen2VLPlugin):
    def chunk_audio_by_seconds(self, audio, sampling_rate=16000, chunk_duration=1.0):
        chunk_size = int(sampling_rate * chunk_duration)
        total_length = len(audio)
        chunks = []
        for start in range(0, total_length, chunk_size):
            end = start + chunk_size
            chunk = audio[start:end]
            chunks.append(chunk)
        return chunks
    
    def extract_features_chunked_batch(self, audios, processor, feature_extractor, sampling_rate=16000, chunk_duration=1.0):
        all_input_features = []
        all_attention_masks = []

        for audio in audios:
            audio_chunks = self.chunk_audio_by_seconds(audio, sampling_rate, chunk_duration)
            #print("audio_chunks_len:",len(audio_chunks),"audio_chunks:",audio_chunks)
            features = feature_extractor(
                audio_chunks,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                max_length = 16000,
                return_tensors="pt",
            )
            input_feat = features["input_features"]       # shape: [num_chunks, 128, T]
            attn_mask = features["attention_mask"]        # shape: [num_chunks, T]

            # 拼接成一条长序列
            input_feat = input_feat.transpose(1, 2).reshape(1, -1, 128).transpose(1, 2)  # [1, 128, total_T]
            attn_mask = attn_mask.reshape(1, -1)                                         # [1, total_T]

            all_input_features.append(input_feat[0]) # [0].shape: [128,t*100]
            all_attention_masks.append(attn_mask[0])

        max_len = max(feat.shape[-1] for feat in all_input_features)

        # padding对齐
        padded_features = []
        padded_attn_masks = []

        for feat, mask in zip(all_input_features, all_attention_masks):
            pad_len = max_len - feat.shape[-1]
            padded_feat = F.pad(feat, (0, pad_len), value=0.0)         # [128, max_len]
            padded_mask = F.pad(mask, (0, pad_len), value=0)           # [max_len]
            padded_features.append(padded_feat)
            padded_attn_masks.append(padded_mask)

        batch_input_features = torch.stack(padded_features, dim=0)     # [B, 128, max_T]
        batch_attention_masks = torch.stack(padded_attn_masks, dim=0)  # [B, max_T]

        return batch_input_features, batch_attention_masks
    
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        messages = [[[],[]]]
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}
        # 每条数据的每个messages都算
        ## 提前先算出来哪个stream最长，video,query,ans
        
        max_length = []
        for i, mes in enumerate(messages):
            video_length = 0
            last_query_end = 0
            last_ans_end = 0
            #video
            if len(videos)!=0 and len(videos[i]) == 1 and videos[i][0].endswith('.mp4'): #单独的一个video路径
                clip = VideoFileClip(videos[i][0])
                video_length = math.ceil(clip.duration)
                clip.close()
                #print('视频长度：',video_length)
            elif len(videos)!=0:
                video_length = math.ceil(len(videos[i])/2.0) #是已经提取完的一系列images，2fps
                #print('视频长度：',video_length)

            # query
            if mes[0][-1]['audio'] is not None: #messages是[[[][]]] 取最后一个query
                last_query = mes[0][-1] # 
                # if isinstance(last_query['audio'], list):
                #     segment_audio = last_query['audio'][0]
                #     original_sr = 16000
                # else:
                #     segment_audio, original_sr = librosa.load(last_query.get('audio'), sr=None, mono=True)
                last_query_start = last_query['time']
                #last_query_end = math.ceil(last_query_start + len(segment_audio) / original_sr)
                last_query_end = math.ceil(last_query_start + last_query['duration'])

                #print('query_audio最后结束时间是：',last_query_end)
            # ans
            if mes[-1][-1]['text'] is not None: #找到最后一个ans
                last_ans = mes[-1][-1]
                last_ans_start = last_ans['time']
                last_ans_dur = len(processor.tokenizer(last_ans.get('text'))['input_ids']) * 0.04
                last_ans_end = math.ceil(math.ceil(last_ans_start) + last_ans_dur)-1
                #print('ans最后结束时间是：',last_ans_end)

            max_time = max(video_length, last_query_end, last_ans_end)
            #print("max_time:",max_time)
            
            max_length.append(max_time)

        #####################
        # if len(images) != 0: #
        #     images = self._regularize_images(
        #         images,
        #         image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
        #         image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
        #     )["images"]
        #     mm_inputs.update(image_processor(images, return_tensors="pt"))
        time_start = time.time()
        if len(videos) != 0:
            video_dict = self._regularize_videos(
                max_length, #[8]
                videos, 
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 800),
            )
            #print("这里判断len(videos)!=0,然后处理videos:",len(video_dict["videos"]))
            mm_inputs.update(image_processor(images=None, videos=video_dict["videos"], return_tensors="pt"))
            #image_processor(images=None, videos=video_dict["videos"], return_tensors="pt")
            
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            )

        if messages[0][0][0]['audio'] is not None: # messages[0][0]是query，query里有audio
            audios = self._regularize_audios(
                messages,
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                max_length = max_length,
            )["audios"]

            input_features_list, attention_mask_list = self.extract_features_chunked_batch(
                audios=audios,
                processor=processor,
                feature_extractor=feature_extractor,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                chunk_duration=1.0,
            )
            mm_inputs.update({
                "input_features": input_features_list, #[1,128,30*100]
                "attention_mask": attention_mask_list #[1,30*100]
            })

            #print(mm_inputs['input_features'].shape)
            # mm_inputs.update(
            #     feature_extractor(
            #         audios,
            #         sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            #         return_attention_mask=True,
            #         padding="max_length",
                    
            #         return_tensors="pt",
            #     )
            # ) #['input_features].shape = [2,128,30000] ['feature_attention_mask'].shape=[2,30000]
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")

        return mm_inputs #如果video audio 都有，会有dict_keys(['pixel_values_videos', 'video_grid_thw', 'video_second_per_grid', 'input_features', 'feature_attention_mask'])
    

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
        mode = "train",
    ) -> list[dict[str, str]]:
        time_start = time.time()    
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None) #Qwen2VLImageProcessor

        merge_length = processor.image_processor.merge_size**2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False) # True
        audio_lengths_from_mm_inputs = []

        if self.expand_mm_tokens: # True
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor, messages)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", []) #tensor([[64,6,12]])
            if "feature_attention_mask" in mm_inputs:
                input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                audio_lengths = (input_lengths - 2) // 2 + 1
                audio_lengths_from_mm_inputs = [audio_lengths] #30s->750
        else:
            mm_inputs = {}
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)
            audio_lengths_from_mm_inputs = [None] * len(audios)

        final_messages_for_template = []
        query_list = messages[0][0]
        ans_list = messages[0][1]

        MODEL_TIME_UNITS_PER_SECOND = 25 

        # --- 1. 音频总模型时间单元数 ---
        total_audio_model_time_units = 0
        if self.expand_mm_tokens and audio_lengths_from_mm_inputs and audio_lengths_from_mm_inputs[0] is not None:
            total_audio_model_time_units = audio_lengths_from_mm_inputs[0][0]
        elif "feature_attention_mask" in mm_inputs and mm_inputs["feature_attention_mask"] is not None:
             # Alternative if audio_lengths not directly from the formula above
             total_audio_model_time_units = int(mm_inputs["feature_attention_mask"].sum().item())
            
        #print(f"mm_plugin_1692: Total audio model time units: {total_audio_model_time_units}")

        audio_t_index_full = torch.arange(total_audio_model_time_units)
        # --- 2. 视频准备 ---
        has_video = False
        video_t_index_full = None
        # 检查 mm_inputs 中是否有视频相关key
        if query_list and query_list[0].get("text", "").count(VIDEO_PLACEHOLDER) > 0 and \
           mm_inputs.get("video_grid_thw") is not None and \
           mm_inputs.get("video_second_per_grid") is not None:
            video_grid_thw = mm_inputs['video_grid_thw'][0] # tensor([T, H, W]) for the first video
            video_sec_per_grid = mm_inputs["video_second_per_grid"][0].item() #3.7755

            if video_sec_per_grid > 0 and image_processor: # 需要processor提供merge_size
                has_video = True
                T_video_frames = video_grid_thw[0].item() #64
                H_grid_final_tokens = video_grid_thw[1].item() // image_processor.merge_size # 3
                W_grid_final_tokens = video_grid_thw[2].item() // image_processor.merge_size # 6
                video_t_index_full = (
                    torch.arange(T_video_frames) # [0, 1, ..., T-1]
                    .view(-1, 1, 1) # Shape (T, 1, 1)
                    .expand(
                        -1, # T
                        H_grid_final_tokens,
                        W_grid_final_tokens,
                    ) # Shape (T, H_final, W_final), values are t_idx for all spatial tokens in that t_idx
                    .flatten() # Shape (T * H_final * W_final), values are [0...0, 1...1, ...]
                    * video_sec_per_grid  # Convert T_frame_idx to real seconds for that frame
                    * MODEL_TIME_UNITS_PER_SECOND # Convert real seconds to model time units
                ).long()
            else:
                has_video = False
        # --- 3. 准备答案 ---
        answers_at_second = {}

        if ans_list:
            for ans_item in ans_list:
                if ans_item.get('time') is None:
                    final_messages_for_template.append({"role": "user", "content": "Narration History"})
                    final_messages_for_template.append({"role": "assistant", "content": ans_item.get('text', '')})
                    continue
                if not isinstance(ans_item, dict): continue
                insert_second = int(np.ceil(float(ans_item.get('time', 0.0))))
                text_to_add = ans_item.get('text', '')
                current_text = answers_at_second.get(insert_second, "")
                answers_at_second[insert_second] = (text_to_add).strip() #(current_text + " " + text_to_add + '<|im_end|>').strip()

        # --- 4. 分块 ---
        t_ntoken_per_chunk_1s = MODEL_TIME_UNITS_PER_SECOND

        audio_chunk_indices_1s_list = []
        if total_audio_model_time_units > 0:
            audio_chunk_indices_1s_list = processor.get_chunked_index(audio_t_index_full, t_ntoken_per_chunk_1s)
        
        video_chunk_indices_1s_list = []
        if has_video and video_t_index_full is not None:
            video_chunk_indices_1s_list = processor.get_chunked_index(video_t_index_full, t_ntoken_per_chunk_1s)
        
        num_chunks = 0
        num_chunks = max(len(audio_chunk_indices_1s_list), len(video_chunk_indices_1s_list))
        #print(f"DEBUG: Num audio chunks: {len(audio_chunk_indices_1s_list)}, Num video chunks: {len(video_chunk_indices_1s_list)}, Max chunks: {num_chunks}")

        # --- 5. 构建交替的 messages 列表 ---
        # if audio_chunk_indices_1s_list:
        #     num_chunks = len(audio_chunk_indices_1s_list) # 默认audio是>=video长度的，有audio用audio
        # elif video_chunk_indices_1s_list and not audio_chunk_indices_1s_list: # 如果只有视频（其实不存在这情况）
        #     num_chunks = len(video_chunk_indices_1s_list)
        
        leftover_assistant_tokens = []  # 存储未说完的助手回答的token ID
        MAX_ASSISTANT_TOKENS_PER_CHUNK = 25        # 每个助手回合最大输出的文本token数
        

        if num_chunks == 0:
            print('mm_plugin_1818: num_chunks=0')
            return []
        for chunk_idx in range(num_chunks):
            media_content_this_chunk = ""
            video_tokens_str_this_chunk = ""
            audio_tokens_str_this_chunk = ""
            has_actual_video_content_this_chunk = False
            has_actual_audio_content_this_chunk = False

            if has_video and chunk_idx < len(video_chunk_indices_1s_list):
                video_chunk_range = video_chunk_indices_1s_list[chunk_idx]
                num_video_tok_this_chunk = video_chunk_range[1] - video_chunk_range[0]
                if num_video_tok_this_chunk > 0:
                    video_tokens_str_this_chunk = self.video_token * num_video_tok_this_chunk
                    has_actual_video_content_this_chunk = True

            if chunk_idx < len(audio_chunk_indices_1s_list):
                audio_chunk_range = audio_chunk_indices_1s_list[chunk_idx]
                num_audio_tok_this_chunk = audio_chunk_range[1] - audio_chunk_range[0]
                if num_audio_tok_this_chunk > 0:
                    audio_tokens_str_this_chunk = self.audio_token * num_audio_tok_this_chunk
                    has_actual_audio_content_this_chunk = True

            if has_actual_video_content_this_chunk and has_actual_audio_content_this_chunk:
                # 同时有视频和音频内容 -> 使用新的组合格式
                media_content_this_chunk = (
                    f"<|vision_bos|><|audio_bos|>"
                    f"{video_tokens_str_this_chunk}"
                    f"{audio_tokens_str_this_chunk}"
                    f"<|audio_eos|><|vision_eos|>"
                )
            elif has_actual_video_content_this_chunk: # 只有视频内容
                print('only_video=========')
                #print(messages[-1][-1][-1])
                #print('===================')
                media_content_this_chunk = (
                    f"<|vision_bos|>"
                    f"{video_tokens_str_this_chunk}"
                    f"<|vision_eos|>"
                )
            elif has_actual_audio_content_this_chunk: # 只有音频内容
                media_content_this_chunk = (
                    f"<|audio_bos|>"
                    f"{audio_tokens_str_this_chunk}"
                    f"<|audio_eos|>"
                )
                print('only_audio=========')
            final_messages_for_template.append({"role": "user", "content": media_content_this_chunk.strip()})

            # --- 助手回合 ---
            assistant_response_time_key = chunk_idx + 1 
            scheduled_answer_text  = answers_at_second.get(assistant_response_time_key, "")
            current_turn_assistant_token_ids = []
            if scheduled_answer_text:
                if leftover_assistant_tokens:
                    #print(f"INFO (chunk_idx {chunk_idx}): New answer scheduled ('{scheduled_answer_text[:30]}...') "
                    #      f"at time key {assistant_response_time_key}. Discarding previous leftover tokens "
                    #      f"and processing the new answer.")
                    leftover_assistant_tokens = []
                all_new_answer_tokens = processor.tokenizer(scheduled_answer_text)['input_ids']
                
                if len(all_new_answer_tokens) > MAX_ASSISTANT_TOKENS_PER_CHUNK:
                    current_turn_assistant_token_ids = all_new_answer_tokens[:MAX_ASSISTANT_TOKENS_PER_CHUNK]
                    current_turn_assistant_token_ids.extend(processor.tokenizer("<|endoftext|>")['input_ids'])
                    # 此时 leftover_assistant_tokens 存储的是 *新答案处理后* 的剩余部分
                    leftover_assistant_tokens = all_new_answer_tokens[MAX_ASSISTANT_TOKENS_PER_CHUNK:]
                else:
                    current_turn_assistant_token_ids = all_new_answer_tokens
            elif leftover_assistant_tokens: # 没有新的预设答案，但有上一轮的剩余
                if len(leftover_assistant_tokens) > MAX_ASSISTANT_TOKENS_PER_CHUNK:
                    current_turn_assistant_token_ids = leftover_assistant_tokens[:MAX_ASSISTANT_TOKENS_PER_CHUNK]
                    current_turn_assistant_token_ids.extend(processor.tokenizer("<|endoftext|>")['input_ids'])
                    leftover_assistant_tokens = leftover_assistant_tokens[MAX_ASSISTANT_TOKENS_PER_CHUNK:]
                else:
                    current_turn_assistant_token_ids = leftover_assistant_tokens
                    leftover_assistant_tokens = []
            # if leftover_assistant_tokens:
            #     if scheduled_answer_text: 

            #         # 如果有新的预设答案，并且有未说完的助手回答，打印警告
            #         print(f"WARNING (chunk_idx {chunk_idx}): New answer scheduled ('{scheduled_answer_text[:30]}...') "
            #             f"at time key {assistant_response_time_key} while previous answer has leftovers. "
            #             f"Prioritizing and continuing with leftover text.")
                    
            #     if len(leftover_assistant_tokens) > MAX_ASSISTANT_TOKENS_PER_CHUNK:
            #         current_turn_assistant_token_ids = leftover_assistant_tokens[:MAX_ASSISTANT_TOKENS_PER_CHUNK]
            #         leftover_assistant_tokens = leftover_assistant_tokens[MAX_ASSISTANT_TOKENS_PER_CHUNK:]
            #     else:
            #         current_turn_assistant_token_ids = leftover_assistant_tokens
            #         leftover_assistant_tokens = []
            # elif scheduled_answer_text:  # 2. 没有剩下的，但有新的预设答案
            #     # 使用 processor.tokenizer 对预设答案文本进行编码
            #     # add_special_tokens=False 确保只编码内容，不添加额外的bos/eos等（模板的encode_multiturn会处理）
            #     all_new_answer_tokens = processor.tokenizer(scheduled_answer_text)['input_ids']
                
            #     if len(all_new_answer_tokens) > MAX_ASSISTANT_TOKENS_PER_CHUNK:
            #         current_turn_assistant_token_ids = all_new_answer_tokens[:MAX_ASSISTANT_TOKENS_PER_CHUNK]
            #         leftover_assistant_tokens = all_new_answer_tokens[MAX_ASSISTANT_TOKENS_PER_CHUNK:]
            #     else:
            #         current_turn_assistant_token_ids = all_new_answer_tokens
                    # leftover_assistant_tokens 保持为空 []
            # else: 3. 既没有剩下的，也没有新的预设答案，current_turn_assistant_token_ids 保持为空 []
            # 将当前回合的助手token ID解码成文本
            if mode == "train":
                final_assistant_content_this_turn = "<|silence|>" #"<|im_end|>"
            else:
                final_assistant_content_this_turn = "<|silence|>"
            if current_turn_assistant_token_ids:
                final_assistant_content_this_turn = processor.tokenizer.decode(current_turn_assistant_token_ids)

            final_messages_for_template.append({"role": "assistant", "content": final_assistant_content_this_turn})
        
        #time_end = time.time()
        #print(f"process_messages: {time_end - time_start} seconds")
        return final_messages_for_template
        # for message in messages: #[query,ans]
        #     # 只check query的第一个text里有没有video。
        #     # content = message["content"]
        #     while IMAGE_PLACEHOLDER in content:
        #         image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
        #         content = content.replace(
        #             IMAGE_PLACEHOLDER, f"<|vision_bos|>{self.image_token * image_seqlen}<|vision_eos|>", 1
        #         )
        #         num_image_tokens += 1

        #     if (
        #         use_audio_in_video and len(audios) and len(videos)
        #     ):  # if use the audio of video # deal video token and audio token togather
        #         if len(videos) != len(audios):
        #             raise ValueError(
        #                 f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
        #             )

        #         while VIDEO_PLACEHOLDER in content:
        #             video_pos = content.find(VIDEO_PLACEHOLDER)
        #             audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
        #             if audio_pos == -1 or audio_pos < video_pos:
        #                 raise ValueError(
        #                     f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
        #                 )

        #             audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
        #             video_t_index = (
        #                 torch.arange(video_grid_thw[num_video_tokens][0])
        #                 .view(-1, 1, 1)
        #                 .expand(
        #                     -1,
        #                     video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
        #                     video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
        #                 )
        #                 .flatten()
        #                 * mm_inputs["video_second_per_grid"][num_video_tokens]
        #                 * 25  # FIXME hardcode of position_id_per_seconds=25
        #             ).long() 
        #             t_ntoken_per_chunk = 50  # FIXME hardcode: 2s
        #             video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
        #             audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
        #             placeholder_string = ""
        #             placeholder_string += "<|vision_bos|>" + "<|audio_bos|>"
        #             for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
        #                 video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
        #                 audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
        #                 if video_chunk_index is not None:
        #                     placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

        #                 if audio_chunk_index is not None:
        #                     placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

        #             placeholder_string += "<|audio_eos|>" + "<|vision_eos|>"
        #             content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
        #             content = content.replace(AUDIO_PLACEHOLDER, "", 1)
        #             num_audio_tokens += 1
        #             num_video_tokens += 1
        #     else:
        #         while AUDIO_PLACEHOLDER in content:
        #             audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
        #             content = content.replace(
        #                 AUDIO_PLACEHOLDER, f"<|audio_bos|>{self.audio_token * audio_seqlen}<|audio_eos|>", 1
        #             )
        #             num_audio_tokens += 1

        #         while VIDEO_PLACEHOLDER in content:
        #             video_seqlen = (
        #                 video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
        #             )
        #             content = content.replace(
        #                 VIDEO_PLACEHOLDER, f"<|vision_bos|>{self.video_token * video_seqlen}<|vision_eos|>", 1
        #             )
        #             num_video_tokens += 1

        #     message["content"] = content ### 这里就得到了一个role不变，但是content覆盖成<|vision_bos|><|audio_bos|><|VIDEO|>*50<|audio|>*50<|audio_eos|><|vision_eos|>What is the video describing?

        return messages


@dataclass
class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        num_frames = 0
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values_images" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values_images"][0]))
                num_frames = 1

            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs["pixel_values_videos"][0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim

            if "pixel_values_images" in mm_inputs or "pixel_values_videos" in mm_inputs:
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                video_seqlen = image_seqlen * num_frames
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen, video_seqlen = 1, 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)
                num_video_tokens += 1

            content = content.replace("{{image}}", self.image_token)
            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


PLUGINS = {
    "base": BasePlugin,
    "gemma3": Gemma3Plugin,
    "intern_vl": InternVLPlugin,
    "kimi_vl": KimiVLPlugin,
    "llama4": Llama4Plugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "minicpm_v": MiniCPMVPlugin,
    "mllama": MllamaPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2_audio": Qwen2AudioPlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "video_llava": VideoLlavaPlugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token)
