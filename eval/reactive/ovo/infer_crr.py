import soundfile as sf
import os
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

def get_past_length(past_key_values):
    if past_key_values is None:
        return 0
    # DynamicCache case
    if isinstance(past_key_values, DynamicCache):
        # New version of transformers has this interface; if versions are inconsistent, can fallback to the one below
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        # Fallback: get the shape of the first layer's k
        k0, v0 = past_key_values[0]
        return k0.shape[2]
    # Old-style tuple(list) case: ((k, v), (k, v), ...)
    k0 = past_key_values[0][0]
    return k0.shape[2]

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

def min_max_normalize(scores, eps: float = 1e-8):
    """Perform per-video min-max normalization on a sequence."""
    if not scores:
        return []
    s_min = min(scores)
    s_max = max(scores)
    if s_max - s_min < eps:
        # All scores are the same, directly set to all 0
        return [0.0 for _ in scores]
    return [(s - s_min) / (s_max - s_min + eps) for s in scores]


parser = ArgumentParser(description="Run Qwen2_5Omni model for video question answering on specified CUDA device.")
parser.add_argument("--model_path", type=str, default="whole_model/model", help="Path to the model folder.")
parser.add_argument("--test_data_path", type=str, default="OVO-Bench/ovobench.jsonl", help="Path to the test data JSONL file.")
parser.add_argument("--score_output_path", type=str, default="result.jsonl", help="Path to output gate probabilities and smoothed scores for each sample.")
parser.add_argument("--head", type=str, default="mlp", help="Choose to use mlp or gate as head.")
parser.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards to split into.")
parser.add_argument("--shard_id", type=int, default=0,
                    help="Which shard the current process handles, [0, num_shards-1].")

args = parser.parse_args()

print(f"Loading model from '{args.model_path}'")
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
        raise ValueError("Input 'example' must be a dictionary.")

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
# data processing
if args.test_data_path.endswith(".jsonl"):
    with open(args.test_data_path, "r") as f:
        conversations = [json.loads(line) for line in f]
else:
    with open(args.test_data_path, "r") as f:
        conversations = json.load(f)

# ====== split shard ======
if args.num_shards > 1:
    shard_conversations = []
    for idx, item in enumerate(conversations):
        if idx % args.num_shards == args.shard_id:
            shard_conversations.append(item)
    print(f"[Shard {args.shard_id}/{args.num_shards}] 负责 {len(shard_conversations)} 条样本")
    conversations = shard_conversations
# ==================================

processed_ids = set()
if os.path.exists(args.score_output_path):
    print(f"发现已存在的输出文件: {args.score_output_path}。正在读取已处理的ID...")
    try:
        with open(args.score_output_path, "r", encoding="utf-8") as f_out:
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
                    print(f"Warning: Skipping unparseable line: {line.strip()}")
    except Exception as e:
        print(f"Error: Failed to read output file: {e}")

original_total = len(conversations)
tasks_to_process = [item for item in conversations if item.get('id') not in processed_ids]
remaining_total = len(tasks_to_process)

print("-" * 50)
print(f"Total tasks: {original_total}")
print(f"Completed tasks: {len(processed_ids)}")
print(f"Remaining tasks: {remaining_total}")
print("-" * 50)

# If all tasks are completed, exit directly.
if remaining_total == 0:
    print("All tasks have been completed! Exiting the program.")
    exit()

with torch.no_grad():
    # Append the score output file line by line.
    with open(args.score_output_path, "a", encoding="utf-8") as score_f:
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
            ### Perform chunk concatenation here
            input_ids = []
            sum_video_token = 0
            sum_audio_token = 0
            final_answer = {}

            # 1. Initialize state variables
            past_key_values = None
            generated_ids_since_last_chunk = torch.tensor([], dtype=torch.long, device=model.device)
            final_answers = []
            final_answer_text = []
            final_answer_time = None
            last_rope_delta = None

            assistant_prefix = [151645, 198, 151644, 77091, 198]    #<|im_end|>\n<|im_start|>assistant\n
            user_prefix = [198, 151643, 872, 198]    #\n<|im_start|>user\n

            # Record gate probabilities for each chunk in this video segment
            chunk_probs = []

#################        
            ask_time = messages[0][0][0]['time']
            print("ask_time:", ask_time)
            question_ids = tokenizer.encode(data['query'][0]['text'].strip("<video>"))
            for i, chunk in enumerate(multimodal_input_id_list):
                
                input_ids.extend(chunk) 
                if i == ask_time-1:
                    input_ids.extend(question_ids)

                prev_sum_video_token = sum_video_token
                sum_video_token += chunk.count(151656)
                sum_audio_token += chunk.count(151646)
                num_video_features = sum_video_token * 4
                num_audio_features = sum_audio_token
                video_features_before_this_chunk = prev_sum_video_token * 4
                
                features = {}
                features['input_ids'] = torch.tensor([input_ids]).to(model.device)
                features['attention_mask'] = torch.ones([1,len(input_ids)],dtype=torch.int64).to(model.device)
                
                features['video_grid_thw'] = mm_inputs['video_grid_thw'].clone().to(model.device)
                features['video_grid_thw'][0, 0] = 1
                features['pixel_values_videos'] = mm_inputs['pixel_values_videos'][video_features_before_this_chunk:num_video_features,:].to(model.dtype).to(model.device)
                features['input_features'] = mm_inputs['input_features'][:, : , (i)*100:(i+1)*100].to(model.dtype).to(model.device)
                features['feature_attention_mask'] = mm_inputs['feature_attention_mask'][:, (i)*100:(i+1)*100].to(model.device)
                features['video_second_per_grid'] = mm_inputs['video_second_per_grid'].to(model.dtype).to(model.device)

                audio_feature_lengths = torch.sum(features['feature_attention_mask'], dim=1)
                position_ids, rope_deltas = model.thinker.get_interleaved_rope_index(
                    features['input_ids'][:,-len(chunk):],
                    None,
                    features['video_grid_thw'],
                    features["attention_mask"][:,-len(chunk):],
                    use_audio_in_video=True,
                    audio_seqlens = audio_feature_lengths,
                )
                final_rope_delta = rope_deltas
                cache_position = torch.arange(0, len(input_ids), dtype=torch.int64).to(model.device)[-len(chunk):]

                if last_rope_delta is not None and cache_position is not None:
                    shift = cache_position[0] + last_rope_delta
                    position_ids += shift
                    final_rope_delta += last_rope_delta

                probe_inputs = {
                    **features,
                    "input_ids":          features['input_ids'][:,-len(chunk):],
                    "attention_mask":     features["attention_mask"],
                    "use_cache":          True,
                    "output_hidden_states": True,
                    "return_dict":        True,
                    "past_key_values":    past_key_values,
                    "rope_deltas":        final_rope_delta,
                    "position_ids":       position_ids, #[3,1,len(chunk)]
                    "cache_position":     cache_position
                }

                out = model.thinker(**probe_inputs)

                hs_all = out.hidden_states[1] if (isinstance(out.hidden_states, tuple) and isinstance(out.hidden_states[1], (list, tuple))) else out.hidden_states
                B, T, H = hs_all[-1].shape #batch,seq_len,hidden_size

                
                layer_ids = getattr(model.thinker, "gate_layer_ids", [-4, -3, -2, -1])
                mix_w = model.thinker.gate_mixer.weights() 
                
                h_mix = 0.0
                L = len(hs_all)
                for w, lid in zip(mix_w, layer_ids):#layer_ids
                    lid = lid if lid >= 0 else L + lid
                    lid = int(max(0, min(L - 1, lid)))
                    h_l = hs_all[lid]                 # [B,T,H]
                    anchor_pos = T - 1
                    anchor_idx = torch.tensor([[anchor_pos]], device=h_l.device, dtype=torch.long)
                    idx = anchor_idx.unsqueeze(-1).expand(B, 1, H)
                    h_anchor = torch.gather(h_l, 1, idx)    # [B,1,H]
                    h_mix = h_mix + w * h_anchor
                    

                if args.head == "mlp":
                    logit = model.thinker.gate_head_pro_fc2(model.thinker.gate_head_pro_act(model.thinker.gate_head_pro_fc1(h_mix))).squeeze(-1).squeeze(-1)
                else:
                    logit = model.thinker.gate_head(h_mix).squeeze(-1).squeeze(-1)   # [B]
                prob  = torch.sigmoid(logit).item()
                print(prob)
                chunk_probs.append(prob)

                past_key_values = out.past_key_values
                last_rope_delta = out["rope_deltas"]

            # ---------- Perform per-video min-max + smoothing here, and write to file ----------
            if len(chunk_probs) > 0:
                norm_scores = min_max_normalize(chunk_probs)

                result = {
                    "id": data.get("id"),
                    "video": data.get("videos"),
                    "raw_probs": chunk_probs,
                    "norm_scores": norm_scores
                }
                score_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                score_f.flush()