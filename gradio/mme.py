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
parser = ArgumentParser(description="在指定CUDA设备上运行Qwen2_5Omni模型进行视频问答。")
parser.add_argument("--model_path", type=str, default="whole_model/model", help="模型文件夹的路径。")
parser.add_argument("--test_data_path", type=str, default="egoschema/Subset/test_egoschema.jsonl", help="测试数据的JSONL文件路径。")
parser.add_argument("--output_path", type=str, default="eval/basic/egoschema/test_result.jsonl", help="结果输出的JSONL文件路径。")

args = parser.parse_args()

# 不再使用 os.environ，直接创建 torch.device 对象
#device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
#print(f"正在使用设备: {device}")

print(f"从 '{args.model_path}' 加载模型")
model = Qwen2_5OmniModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",  
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
# if not hasattr(model.thinker, "gate_head"):
#     H = model.thinker.config.get_text_config().hidden_size
#     device = next(model.parameters()).device
#     dtype = next(model.parameters()).dtype

#     class GateMixer(torch.nn.Module):
#         def __init__(self, K, device, dtype):
#             super().__init__()
#             self.logits = torch.nn.Parameter(torch.zeros(K, device=device, dtype=dtype))
#         def weights(self): return torch.softmax(self.logits, dim=0)

#     model.thinker.gate_head = torch.nn.Linear(H, 1, bias=True).to(device=device, dtype=dtype)
#     model.thinker.gate_mixer = GateMixer(K=4, device=device, dtype=dtype)  # K按你训练时的值
#     model.thinker.gate_layer_ids = [-4, -3, -2, -1] 

# state = {}
# for shard in sorted(glob.glob(os.path.join(args.model_path, "model-*.safetensors"))):
#     with safe_open(shard, framework="pt", device="cpu") as f:
#         for k in f.keys():
#             if k.startswith("thinker.gate_head.") or k.startswith("thinker.gate_mixer."):
#                 state[k] = f.get_tensor(k)

# head_sd  = {k.split("thinker.gate_head.", 1)[1]:  v for k,v in state.items() if k.startswith("thinker.gate_head.")}
# mixer_sd = {k.split("thinker.gate_mixer.",1)[1]:  v for k,v in state.items() if k.startswith("thinker.gate_mixer.")}

# model.thinker.gate_head.load_state_dict(head_sd, strict=True)
# model.thinker.gate_mixer.load_state_dict(mixer_sd, strict=True)
# print("✅ gate_head / gate_mixer 权重已恢复")

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
    "template": "streaming_turn",
    "train_on_prompt": False,
    "tool_format": None
}
template_args= Namespace(**args_dict)
template = get_template_and_fix_tokenizer(tokenizer, template_args)
###################
def transform_example_format(example: dict[str, any]) -> dict[str, any]:
    if not isinstance(example, dict):
        raise ValueError("输入 'example' 必须是一个字典。")

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
# 开始处理数据
if args.test_data_path.endswith(".jsonl"):
    with open(args.test_data_path, "r") as f:
        conversations = [json.loads(line) for line in f]
else:
    with open(args.test_data_path, "r") as f:
        conversations = json.load(f)

processed_ids = set()
if os.path.exists(args.output_path):
    print(f"发现已存在的输出文件: {args.output_path}。正在读取已处理的ID...")
    try:
        with open(args.output_path, "r", encoding="utf-8") as f_out:
            for line in f_out:
                # 跳过空行
                if not line.strip():
                    continue
                try:
                    # 解析每一行JSON
                    processed_data = json.loads(line)
                    # 确保'id'键存在
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

# 如果所有任务都已完成，则直接退出
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
        ### 在这里进行每个chunk的拼接
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
        # import random
        # ask_time = random.randint(1, ask_time-1)
        for i, chunk in enumerate(multimodal_input_id_list):     
            if i >= ask_time:
                flag = True
                break
            if endswith_tensor(chunk,[151645,198,151644,77091,198]):
                chunk = chunk[:-5]
            # 确保 chunk 的类型与操作安全
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
        # 关键改动：不要把 Tensor 直接 extend 到 list
        input_ids.extend(assistant_prefix)
        
        if not flag and (i == len(multimodal_input_id_list)-1):
            i = i+1
        features['input_ids'] = torch.tensor([input_ids]).to(model.device) 
        features['attention_mask'] = torch.ones([1,len(input_ids)],dtype=torch.int64).to(model.device) 
        # 将 mm_inputs 中的所有 tensor 移动到指定设备
        features['video_grid_thw'] = mm_inputs['video_grid_thw'].clone().to(model.device)
        features['video_grid_thw'][0, 0] = i
        features['pixel_values_videos'] = mm_inputs['pixel_values_videos'][:num_video_features,:].to(model.dtype).to(model.device) 
        features['input_features'] = mm_inputs['input_features'][:, : , :(i)*100].to(model.dtype).to(model.device)
        features['feature_attention_mask'] = mm_inputs['feature_attention_mask'][:, :(i)*100].to(model.device)
        features['video_second_per_grid'] = mm_inputs['video_second_per_grid'].to(model.dtype).to(model.device)
        
########################### probing

#         probe_inputs = {
#             **features,
#             "input_ids":          features["input_ids"],
#             "attention_mask":     features["attention_mask"],
#             "use_cache":          True,
#             "output_hidden_states": True,
#             "return_dict":        True,
#             "past_key_values":    past_key_values,
#             "rope_deltas":        last_rope_delta,
#         }

#         out = model.thinker(**probe_inputs)
#         hs_all = out.hidden_states[1] if (isinstance(out.hidden_states, tuple) and isinstance(out.hidden_states[1], (list, tuple))) else out.hidden_states
#         B, T, H = hs_all[-1].shape

#         anchor_pos = features['input_ids'].size(1) - 1
#         anchor_idx = torch.tensor([[anchor_pos]], device=features['input_ids'].device)
#         idx = anchor_idx.unsqueeze(-1).expand(B, 1, H)

#         # 层混合
#         layer_ids = getattr(model.thinker, "gate_layer_ids", [-4, -3, -2, -1])
#         mix_w = model.thinker.gate_mixer.weights() 
#         print("mix_w:",mix_w)
#         h_mix = 0.0
#         L = len(hs_all)
#         for w, lid in zip(mix_w, layer_ids):
#             lid = lid if lid >= 0 else L + lid
#             lid = int(max(0, min(L - 1, lid)))
#             h_l = hs_all[lid]                 # [B,T,H]
#             h_anchor = torch.gather(h_l, 1, idx)    # [B,1,H]
#             h_mix = h_mix + w * h_anchor

#         logit = model.thinker.gate_head(h_mix).squeeze(-1).squeeze(-1)   # [B]
#         prob  = torch.sigmoid(logit).item()
#         gate_decision = (prob >= 0.1)
#         all_probs.append(prob)
#         print(prob)

# print("平均概率:", np.mean(all_probs), "概率方差:", np.var(all_probs))

######################################## end probing
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
        # 关键改动：空输出保护 + list[int] 保持
        if newly_generated_ids.numel() > 0:
            ids_list = newly_generated_ids.tolist()
            input_ids.extend(ids_list)
            if ids_list[-1] != 151645: 
                input_ids.extend([151643,151645])
            newly_generated_text = processor.decode(newly_generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            newly_generated_text = ""
        # 关键改动：不要把 Tensor 直接 extend
        input_ids.extend(user_prefix)

        print("newly_generated:", [newly_generated_text])

        stop_token_str = "<|im_end|>"
        is_answer_finished = False
        chunk_to_process = newly_generated_text

        if stop_token_str in newly_generated_text:
            stop_index = newly_generated_text.index(stop_token_str)
            chunk_to_process = newly_generated_text[:stop_index]
            # if chunk_to_process == "<|silence|>":
            #     chunk_to_process = ""
            is_answer_finished = True
        if chunk_to_process:
            if final_answer_time is None:
                final_answer_time = i+1
            # 关键改动：append 而不是 extend（避免按字符拆）
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
            #print("进入到第二个循环")
            if i <= ask_time:
                continue
            # 将 chunk 移动到指定设备（关键改动：类型安全）
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
                # 关键改动：不要 extend Tensor
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
            
            start_time = time.time()
            
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
            end_time = time.time()
            print("完成一个chunk的耗时：", end_time - start_time)
            
            past_key_values = output.past_key_values
            newly_generated_ids = output.sequences[0,len(input_ids):]
        
            # 关键改动：空输出保护 + list[int] 保持
            if newly_generated_ids.numel() > 0:
                ids_list = newly_generated_ids.tolist()
                input_ids.extend(ids_list)
                if ids_list[-1] != 151645:
                    input_ids.extend([151643,151645])
                newly_generated_text = processor.decode(newly_generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            else:
                newly_generated_text = ""
            # 关键改动：不要把 Tensor 直接 extend
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
                # 关键改动：append 而不是 extend
                final_answer_text.append(chunk_to_process)
            if is_answer_finished and final_answer_text:
                final_answer = {
                    'time': final_answer_time,
                    'text': "".join(final_answer_text)
                }
                final_answers.append(final_answer)
                break

        # 关键改动：写盘前兜底提交（即使没有 <|im_end|> 也写当前累积）
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
