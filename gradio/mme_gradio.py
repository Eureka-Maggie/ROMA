import gradio as gr
import time
import os
import torch
import json
import math
import soundfile as sf
from argparse import Namespace
from transformers import Qwen2_5OmniModel, AutoTokenizer, AutoProcessor
from transformers.cache_utils import DynamicCache
from src.llamafactory.model.loader import patch_processor
from src.llamafactory.data.template import get_template_and_fix_tokenizer
from qwen_vl_utils import process_vision_info
from moviepy.editor import VideoFileClip

# ================= Configuration =================
MODEL_PATH = "whole_model/model"
DEFAULT_VIDEO_PATH = "videos_w_audio/uoJDGnaVuTg_290_silent.mp4"
DEFAULT_AUDIO_PATH = "audios/290.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH} on {DEVICE}...")

# ================= Model Loading =================
model = Qwen2_5OmniModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    split_special_tokens=False,
    padding_side="left",
    trust_remote_code=True,
    cache_dir = None,
    revision = 'main',
    token = None
)

processor_args_dict = {
    "image_max_pixels": 262144,
    "image_min_pixels": 1024,
    "image_do_pan_and_scan": False,
    "crop_to_patches": False,
    "video_max_pixels": 65536,
    "video_min_pixels": 256,
    "video_fps": 2.0,
    "video_maxlen": 14400,
    "audio_sampling_rate": 16000,
    "use_audio_in_video": True
}
processor_args = Namespace(**processor_args_dict)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
patch_processor(processor, tokenizer, processor_args)

template_args = Namespace(**{"template": "streaming_mix", "train_on_prompt": False, "tool_format": None})
template = get_template_and_fix_tokenizer(tokenizer, template_args)

print("Model Loaded Successfully!")

# ================= Helper Functions =================
def endswith_tensor(seq, pattern):
    if isinstance(seq, torch.Tensor):
        return torch.equal(seq[-len(pattern):], pattern.to(seq.device))
    else:
        return seq[-len(pattern):] == pattern

def transform_example_format(example: dict) -> dict:
    return {
        "_prompt": example.get("query", []), 
        "_response": example.get("ans", []),  
        "_system": "",                       
        "_tools": example.get("tools", "") if example.get("tools") else "",                                                                 
        "_images": example.get("images") if len(example.get("images"))!=0 else None,
        "_videos": example.get("videos") if len(example.get("videos"))!=0 else None,
        "_audios": []
    }

def get_multimodal_input_ids(prompt, response, system, tools, images, videos, audios):
    messages = template.mm_plugin.process_messages(
        [[prompt, response]], images, videos, audios, processor, mode="infer"
    )
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    inputs_list = []
    for input_multimodal, _ in encoded_pairs:
        inputs_list.append(input_multimodal)
    return inputs_list

def get_media_duration(video_path, audio_path):
    try:
        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        clip.close()
        audio_info = sf.info(audio_path)
        audio_duration = audio_info.duration
        return video_duration, audio_duration
    except Exception as e:
        print(f"Error reading media duration: {e}")
        return 0, 0

# ================= Core Inference Logic (保持不变) =================
def model_inference(video_path, audio_path, history):
    if not video_path or not audio_path:
        history = history or []
        history.append((None, "❌ 请先上传视频和音频文件。"))
        yield history
        return

    print(f"Processing - Video: {video_path}, Audio: {audio_path}")
    
    # 确保 history 是列表
    history = history or []
    
    # 1. 计算时间参数
    v_dur, a_dur = get_media_duration(video_path, audio_path)
    calc_time = float(math.floor(v_dur - a_dur))
    if calc_time < 0: calc_time = 0 

    # 2. 构造数据
    data =  {
        "task": "basic",
        "id": 291,
        "ans": [{"text": "", "time": 0.0}],
        "query": [
        {
            "text": "<video>How does the right one become at the end?",
            "audio": "audios/290.wav",
            "time": 14.0,
            "duration": 1.4833125
        }
        ],
        "videos": [[
            "videos_w_audio/uoJDGnaVuTg_290_silent.mp4"
        ]],
        "images": [],
        "answer": {"segment": [], "text": "Small."}
    }

    data_formated = transform_example_format(data)
    multimodal_input_id_list = get_multimodal_input_ids(
        prompt=data_formated["_prompt"],
        response=data_formated["_response"],
        system="",
        tools="",
        images=data_formated["_images"] or [],
        videos=data_formated["_videos"] or [],
        audios=[],
    )

    batch_images = []
    batch_videos = [data['videos'][0]]
    batch_audios = []
    batch_imglens = [0]
    batch_vidlens = []
    batch_audlens = [1]
    batch_input_ids = [multimodal_input_id_list[0]]
    messages = [[data_formated["_prompt"], data_formated["_response"]]]
    
    mm_inputs = template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_audios, batch_imglens, batch_vidlens, batch_audlens,
            batch_input_ids, processor, messages=messages,
    )
    
    features = {}
    input_ids = []
    sum_video_token = 0
    sum_audio_token = 0
    past_key_values = None
    last_rope_delta = None
    
    ask_time = messages[0][0][0]['time'] + math.ceil(messages[0][0][0]['duration'])
    assistant_prefix = [151645, 198, 151644, 77091, 198] 
    user_prefix = [198, 151644, 872, 198]               

    current_response = "" 

    # === 循环处理 ===
    flag = False
    for i, chunk in enumerate(multimodal_input_id_list):
        if i >= ask_time:
            flag = True
            break
            
        if endswith_tensor(chunk, [151645,198,151644,77091,198]):
            chunk = chunk[:-5]
            
        if isinstance(chunk, torch.Tensor):
            input_ids.extend(chunk.tolist())
            sum_video_token += int((chunk == 151656).sum().item())
            sum_audio_token += int((chunk == 151646).sum().item())
        else:
            input_ids.extend(chunk)
            sum_video_token += chunk.count(151656)
            sum_audio_token += chunk.count(151646)
            
        num_video_features = sum_video_token * 4
        
    input_ids.extend(assistant_prefix)
    
    if not flag and (i == len(multimodal_input_id_list)-1):
        i = i + 1
        
    features['input_ids'] = torch.tensor([input_ids]).to(model.device)
    features['attention_mask'] = torch.ones([1, len(input_ids)], dtype=torch.int64).to(model.device)
    features['video_grid_thw'] = mm_inputs['video_grid_thw'].clone().to(model.device)
    features['video_grid_thw'][0, 0] = i
    features['pixel_values_videos'] = mm_inputs['pixel_values_videos'][:num_video_features, :].to(model.dtype).to(model.device)
    features['input_features'] = mm_inputs['input_features'][:, :, :(i)*100].to(model.dtype).to(model.device)
    features['feature_attention_mask'] = mm_inputs['feature_attention_mask'][:, :(i)*100].to(model.device)
    features['video_second_per_grid'] = mm_inputs['video_second_per_grid'].to(model.dtype).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **features,
            thinker_max_new_tokens=25,
            use_audio_in_video=True,
            return_audio=False,
            streaming=True,
            past_key_values=past_key_values,
            output_scores=True,
            mode="infer",
            rope_deltas=last_rope_delta
        )
        last_rope_delta = output["rope_deltas"]
        past_key_values = output.past_key_values
        
        newly_generated_ids = output.sequences[0, len(input_ids):]
        if newly_generated_ids.numel() > 0:
            ids_list = newly_generated_ids.tolist()
            input_ids.extend(ids_list)
            if ids_list[-1] != 151645:
                input_ids.extend([151643, 151645])
            newly_generated_text = processor.decode(newly_generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        input_ids.extend(user_prefix)
        current_response += newly_generated_text
        print("Progress:", newly_generated_text)
    
    # === 循环彻底结束，一次性输出 ===
    if not current_response:
        final_text = "⚠️ 模型未生成有效回复"
    else:
        final_text = current_response.strip("<|im_end|>")

    print("Final Output:", final_text)
    
    history.append((None, final_text))
    yield history
    
    del features, output, mm_inputs, data_formated, multimodal_input_id_list
    torch.cuda.empty_cache()

# ================= Gradio UI (保持不变) =================
custom_css = """
#submit-btn {
    background-color: #FF7C00 !important;
    border: 1px solid #E66A00 !important;
    color: white !important;
    height: 60px !important;
    font-size: 20px !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    margin-top: 20px !important;
}
#submit-btn:hover { background-color: #E66A00 !important; }

#clean-video .upload-container, #clean-audio .upload-container {
    height: 100% !important;
    min-height: 100% !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    cursor: pointer !important;
}

#clean-audio {
    height: 50px !important;
    min-height: auto !important;
    background-color: #f9f9f9;
    border: 2px dashed #e5e7eb;
}
#clean-audio .upload-container { padding: 0 !important; overflow: hidden; }

#clean-video {
    background-color: #f9f9f9;
    border: 2px dashed #e5e7eb;
}

#roma-chatbot .avatar {
    width: 70px !important;
    height: 70px !important;
    margin-right: 15px !important;
}
#roma-chatbot .avatar img {
    width: 70px !important;
    height: 70px !important;
}
"""

theme = gr.themes.Soft(primary_hue="orange", radius_size="md")

default_avatar = "https://api.iconify.design/noto:robot.svg"
local_avatar = "bot.png"
bot_avatar = local_avatar if os.path.exists(local_avatar) else default_avatar

start_video = DEFAULT_VIDEO_PATH if os.path.exists(DEFAULT_VIDEO_PATH) else None
start_audio = DEFAULT_AUDIO_PATH if os.path.exists(DEFAULT_AUDIO_PATH) else None

with gr.Blocks(theme=theme, css=custom_css, title="Reactive QA") as demo:
    
    gr.Markdown("### Demo Video of ROMA's Omni-Modality Reactive Video Understanding")
    
    with gr.Row():
        with gr.Column(scale=5):
            video_input = gr.Video(
                label="Upload Video", 
                elem_id="clean-video", 
                height=380, 
                interactive=True,
                sources=["upload"],
                value=start_video 
            )
            
            audio_input = gr.Audio(
                label="Upload Audio",
                elem_id="clean-audio",
                height=50,
                type="filepath", 
                interactive=True,
                sources=["upload"],
                value=start_audio
            )

        with gr.Column(scale=5):
            chatbot = gr.Chatbot(
                label="Roma's Output", 
                elem_id="roma-chatbot",
                height=450, 
                bubble_full_width=False,
                avatar_images=(None, bot_avatar)
            )

    with gr.Row():
        submit_btn = gr.Button(
            "▶ Submit to ROMA", 
            elem_id="submit-btn", 
            scale=1
        )

    submit_btn.click(
        fn=model_inference, 
        inputs=[video_input, audio_input, chatbot], 
        outputs=[chatbot],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.queue().launch()