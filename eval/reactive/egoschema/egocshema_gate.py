import soundfile as sf
import os
###os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from io import BytesIO
from urllib.request import urlopen
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniModel, AutoTokenizer, AutoProcessor, Qwen2_5OmniThinkerModel
from src.llamafactory.model.loader import patch_tokenizer, patch_processor
from src.llamafactory.data.template import get_template_and_fix_tokenizer
from argparse import Namespace, ArgumentParser
import torch
import json
from tqdm import tqdm
import time
import math
from transformers.cache_utils import DynamicCache
import glob
from safetensors import safe_open

def endswith_tensor(seq, pattern):
    if isinstance(seq, torch.Tensor):
        return torch.equal(seq[-len(pattern):], pattern.to(seq.device))
    else:  # list
        return seq[-len(pattern):] == pattern
    
def truncate_kv_cache(past_key_values, n_tokens_to_remove=4):
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_key_values):
        # k/v shape: [batch, num_heads, seq_len, head_dim]
        new_k = k[:, :, :-n_tokens_to_remove, :].contiguous()
        new_v = v[:, :, :-n_tokens_to_remove, :].contiguous()
        new_cache.update(new_k, new_v, layer_idx)
    return new_cache

# 添加命令行参数解析
parser = ArgumentParser(description="egoschema")
parser.add_argument("--model_path", type=str, help="模型文件夹的路径。")
parser.add_argument("--test_data_path", type=str, help="测试数据的JSONL文件路径。")
parser.add_argument("--output_path", type=str, help="结果输出的JSONL文件路径。")

args = parser.parse_args()


model = Qwen2_5OmniModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",  
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
            split_special_tokens = False,
            padding_side="left",
            trust_remote_code = True,
            cache_dir = None,
            revision = 'main',
            token = None
        )
newline_token_id = tokenizer.encode("\n", add_special_tokens=False)
processor_args_dict = {
    "image_max_pixels": 262144,
    "image_min_pixels": 1024,
    "image_do_pan_and_scan": False,
    "crop_to_patches": False,
    "video_max_pixels": 65536, #147456,
    "video_min_pixels": 256,
    "video_fps": 2.0,
    "video_maxlen": 14400, #2h
    "audio_sampling_rate": 16000,
    "use_audio_in_video": True
}

processor_args = Namespace(**processor_args_dict)
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code = True, cache_dir = None, revision = 'main', token = None)

patch_processor(processor,tokenizer, processor_args)

args_dict = {
    "template": "streaming_mix",
    "train_on_prompt": False,
    "tool_format": None
}
template_args= Namespace(**args_dict)
template = get_template_and_fix_tokenizer(tokenizer, template_args)
###################
def transform_example_format(example: dict[str, any]) -> dict[str, any]:
    if not isinstance(example, dict):
        raise ValueError(" Input 'example' must be a dictionary.")

    output: dict[str, any] = {
        "_prompt": example.get("query", []), 
        "_response": example.get("ans", []),  
        "_system": "",                       
        "_tools": example.get("tools", "") if example.get("tools") else "",                                                                 
        "_images": example.get("images") if len(example.get("images"))!=0 else None,
        "_videos": example.get("videos") if len(example.get("videos"))!=0 else None,
        "_audios": []
    }
    return output
##################
def get_multimodal_input_ids(prompt,response,system,tools,images,videos,audios):
    messages = template.mm_plugin.process_messages(
        [[prompt,response]], images, videos, audios, processor,mode = "infer"
    )
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    inputs_list = []
    for input_multimodal, _ in encoded_pairs:
        inputs_list.append(input_multimodal)
    return inputs_list
#################
# process data
if args.test_data_path.endswith(".jsonl"):
    with open(args.test_data_path, "r") as f:
        conversations = [json.loads(line) for line in f]
else:
    with open(args.test_data_path, "r") as f:
        conversations = json.load(f)

processed_ids = set()
if os.path.exists(args.output_path):
    try:
        with open(args.output_path, "r", encoding="utf-8") as f_out:
            for line in f_out:
                if not line.strip():
                    continue
                try:
                    processed_data = json.loads(line)
                    if 'id' in processed_data:
                        processed_ids.add(processed_data['id'])
                except json.JSONDecodeError:
                    print(f"警告：跳过无法解析的行: {line.strip()}")
    except Exception as e:
        print(f"错误：读取输出文件时发生错误: {e}")

original_total = len(conversations)
tasks_to_process = [item for item in conversations if item.get('id') not in processed_ids]
remaining_total = len(tasks_to_process)

print("-" * 50)
print(f"总任务数: {original_total}")
print(f"已完成任务数: {len(processed_ids)}")
print(f"待处理任务数: {remaining_total}")
print("-" * 50)

all_probs = []
import numpy as np

if remaining_total == 0:
    print("所有任务均已完成！程序退出。")
    exit()
with torch.no_grad():
    for data in tqdm(tasks_to_process):
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
        batch_videos = []
        batch_videos.append(data['videos'][0])
        batch_audios = []

        batch_imglens = []
        batch_imglens.append(0)
        batch_vidlens = []
        batch_audlens = []
        batch_audlens.append(1)

        batch_input_ids = []
        batch_input_ids.append(multimodal_input_id_list[0])
        messages = []
        messages.append([data_formated["_prompt"], data_formated["_response"]])
        print(messages)
        mm_inputs = template.mm_plugin.get_mm_inputs(
                batch_images,
                batch_videos,
                batch_audios,
                batch_imglens,
                batch_vidlens,
                batch_audlens,
                batch_input_ids,
                processor,
                messages = messages,
            )
        features = {} 
        ### 
        input_ids = []
        sum_video_token = 0
        sum_audio_token = 0
        final_answer = {}

        # 1. 初始化状态变量
        past_key_values = None
        generated_ids_since_last_chunk = torch.tensor([], dtype=torch.long, device=model.device)
        final_answers = []
        final_answer_text = []
        final_answer_time = None
        last_rope_delta = None
    ######################
        flag = False
        ask_time = messages[0][0][0]['time'] + math.ceil(messages[0][0][0]['duration'])

        for i, chunk in enumerate(multimodal_input_id_list):     
            if i >= ask_time:
                flag = True
                break
            if endswith_tensor(chunk,[151645,198,151644,77091,198]):
                chunk = chunk[:-5]
            
            if isinstance(chunk, torch.Tensor):
                input_ids.extend(chunk.tolist())
                sum_video_token += int((chunk == 151656).sum().item()) # <|VIDEO|>
                sum_audio_token += int((chunk == 151646).sum().item()) # <|AUDIO|>
            else:
                input_ids.extend(chunk)
                sum_video_token += chunk.count(151656) #<|VIDEO|>
                sum_audio_token += chunk.count(151646) #<|AUDIO|>
            num_video_features = sum_video_token * 4
            num_audio_features = sum_audio_token        
        
        assistant_prefix = [151645, 198, 151644, 77091, 198]    #<|im_end|>\n<|im_start|>assistant\n
        user_prefix = [198, 151644, 872, 198]    #\n<|im_start|>user\n

        input_ids.extend(assistant_prefix)
        
        if not flag and (i == len(multimodal_input_id_list)-1):
            i = i+1
        features['input_ids'] = torch.tensor([input_ids]).to(model.device) 
        features['attention_mask'] = torch.ones([1,len(input_ids)],dtype=torch.int64).to(model.device) 
        features['video_grid_thw'] = mm_inputs['video_grid_thw'].clone().to(model.device)
        features['video_grid_thw'][0, 0] = i
        features['pixel_values_videos'] = mm_inputs['pixel_values_videos'][:num_video_features,:].to(model.dtype).to(model.device) 
        features['input_features'] = mm_inputs['input_features'][:, : , :(i)*100].to(model.dtype).to(model.device)
        features['feature_attention_mask'] = mm_inputs['feature_attention_mask'][:, :(i)*100].to(model.device)
        features['video_second_per_grid'] = mm_inputs['video_second_per_grid'].to(model.dtype).to(model.device)
        
        output = model.generate(
            **features, 
            thinker_max_new_tokens=25, 
            use_audio_in_video=True, 
            return_audio=False, 
            streaming=True,
            past_key_values=past_key_values,
            output_scores=True,
            mode="infer",
            rope_deltas = last_rope_delta
        )
        last_rope_delta = output["rope_deltas"]
        past_key_values = output.past_key_values

        newly_generated_ids = output.sequences[0,len(input_ids):]
        
        if newly_generated_ids.numel() > 0:
            ids_list = newly_generated_ids.tolist()
            input_ids.extend(ids_list)
            if ids_list[-1] != 151645: 
                input_ids.extend([151643,151645])
            newly_generated_text = processor.decode(newly_generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            newly_generated_text = ""
        
        input_ids.extend(user_prefix)

        print("newly_generated:", [newly_generated_text])

        stop_token_str = "<|im_end|>"
        is_answer_finished = False
        chunk_to_process = newly_generated_text

        if stop_token_str in newly_generated_text:
            stop_index = newly_generated_text.index(stop_token_str)
            chunk_to_process = newly_generated_text[:stop_index]
            
            is_answer_finished = True
        if chunk_to_process:
            if final_answer_time is None:
                final_answer_time = i+1
            
            final_answer_text.append(chunk_to_process)
        if is_answer_finished and final_answer_text:
            final_answer = {
                'time': final_answer_time,
                'text': "".join(final_answer_text)
            }
            final_answers.append(final_answer)
            
            with open(args.output_path, "a") as f:
                final_answer = {
                    'id': data['id'],
                    'question' : data['query'][0]['text'],
                    'prediction': final_answers,
                    "gt": data['answer']
                }
                f.write(json.dumps(final_answer, ensure_ascii=False) + "\n")   
            del data_formated, multimodal_input_id_list, mm_inputs, features, output, final_answers
            torch.cuda.empty_cache()
            continue
    #######
        for i, chunk in enumerate(multimodal_input_id_list):
            
            if i <= ask_time:
                continue
            
            if isinstance(chunk, torch.Tensor):
                input_ids.extend(chunk.tolist())
                prev_sum_video_token = sum_video_token
                sum_video_token += int((chunk == 151656).sum().item())
                sum_audio_token += int((chunk == 151646).sum().item())
            else:
                input_ids.extend(chunk) 
                prev_sum_video_token = sum_video_token
                sum_video_token += chunk.count(151656)
                sum_audio_token += chunk.count(151646)
            num_video_features = sum_video_token * 4
            num_audio_features = sum_audio_token
            video_features_before_this_chunk = prev_sum_video_token * 4
            
            features = {}
            if i != len(multimodal_input_id_list)-1:
                
                input_ids.extend(assistant_prefix)
            features['input_ids'] = torch.tensor([input_ids]).to(model.device)
            features['attention_mask'] = torch.ones([1,len(input_ids)],dtype=torch.int64).to(model.device)
            # 将 mm_inputs 中的所有 tensor 移动到指定设备
            features['video_grid_thw'] = mm_inputs['video_grid_thw'].clone().to(model.device)
            features['video_grid_thw'][0, 0] = 1
            features['pixel_values_videos'] = mm_inputs['pixel_values_videos'][video_features_before_this_chunk:num_video_features,:].to(model.dtype).to(model.device)
            features['input_features'] = mm_inputs['input_features'][:, : , (i)*100:(i+1)*100].to(model.dtype).to(model.device)
            features['feature_attention_mask'] = mm_inputs['feature_attention_mask'][:, (i)*100:(i+1)*100].to(model.device)
            features['video_second_per_grid'] = mm_inputs['video_second_per_grid'].to(model.dtype).to(model.device)
            
            output = model.generate(
                **features, 
                thinker_max_new_tokens=25, 
                use_audio_in_video=True, 
                return_audio=False, 
                streaming=True,
                past_key_values=past_key_values,
                output_scores=True,
                mode="infer",
                rope_deltas = last_rope_delta
            )
            last_rope_delta = output["rope_deltas"]
            
            past_key_values = output.past_key_values
            newly_generated_ids = output.sequences[0,len(input_ids):]
        
            if newly_generated_ids.numel() > 0:
                ids_list = newly_generated_ids.tolist()
                input_ids.extend(ids_list)
                if ids_list[-1] != 151645:
                    input_ids.extend([151643,151645])
                newly_generated_text = processor.decode(newly_generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            else:
                newly_generated_text = ""
            
            input_ids.extend(user_prefix)

            print("newly_generated:", [newly_generated_text])

            stop_token_str = "<|im_end|>"
            is_answer_finished = False
            chunk_to_process = newly_generated_text

            if stop_token_str in newly_generated_text:
                stop_index = newly_generated_text.index(stop_token_str)
                chunk_to_process = newly_generated_text[:stop_index]
                is_answer_finished = True
            if chunk_to_process:
                if final_answer_time is None:
                    final_answer_time = i+1
                
                final_answer_text.append(chunk_to_process)
            if is_answer_finished and final_answer_text:
                final_answer = {
                    'time': final_answer_time,
                    'text': "".join(final_answer_text)
                }
                final_answers.append(final_answer)
                break

       
        if (not final_answers) and final_answer_text:
            final_answers.append({
                'time': final_answer_time or len(multimodal_input_id_list),
                'text': "".join(final_answer_text)
            })

        with open(args.output_path, "a") as f:
            final_answer = {
                'id': data['id'],
                'question' : data['query'][0]['text'],
                'prediction': final_answers,
                "gt": data['answer']
            }
            f.write(json.dumps(final_answer, ensure_ascii=False) + "\n")   
        del data_formated, multimodal_input_id_list, mm_inputs, features, output, final_answers
        torch.cuda.empty_cache()