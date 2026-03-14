import gradio as gr
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import math
import soundfile as sf
from argparse import Namespace
from transformers import Qwen2_5OmniModel, AutoTokenizer, AutoProcessor
from src.llamafactory.model.loader import patch_processor
from src.llamafactory.data.template import get_template_and_fix_tokenizer

# ================= Configuration =================
MODEL_PATH = "whole_model/model"
FIXED_VIDEO_PATH = "aCkbw-aI4xU_cut80s.mp4"
FIXED_AUDIO_PATH = "youcook2_ques.wav"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.975

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
    cache_dir=None,
    revision='main',
    token=None
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

# ================= Core Logic =================
def run_narration_stream():
    """
    流式叙述逻辑：
    1. 预处理数据
    2. 倒计时
    3. 逐秒推理，若 Prob > 0.975 则生成描述
    """
    if not os.path.exists(FIXED_VIDEO_PATH) or not os.path.exists(FIXED_AUDIO_PATH):
        yield "❌ 错误：找不到指定的视频或音频文件。"
        return

    # --- 阶段 1: 数据预处理 ---
    log_text = ""
    yield log_text
    print("⚙️ 正在后台加载视频与预处理数据，请稍候...\n")

    # 构造 YouCook2 数据
    data = {
        "task": "narration",
        "id": "aCkbw-aI4xU_6",
        "ans": [{"text": "", "time": 0.0}],
        "query": [
            {
                "text": "<video>\nDescribe various activity events occurring in the video in real time.",
                "audio": FIXED_AUDIO_PATH,
                "time": 0.0,
                "duration": 4.0729375 # 音频长度
            }
        ],
        "videos": [["aCkbw-aI4xU.mp4"]],
        "images": [],
        "answer": [] # Demo中不需要 GT answer
    }
    
    data_formated = transform_example_format(data)
    multimodal_input_id_list = get_multimodal_input_ids(
        prompt=data_formated["_prompt"],
        response=data_formated["_response"],
        system="",
        tools="",
        images=[],
        videos=data_formated["_videos"],
        audios=[]
    )

    batch_images = []
    batch_videos = [data_formated['_videos'][0]]
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
    
    # 状态变量初始化
    input_ids = []
    sum_video_token = 0
    sum_audio_token = 0
    past_key_values = None
    last_rope_delta = None
    
    assistant_prefix = [151645, 198, 151644, 77091, 198] # <|im_end|>\n<|im_start|>assistant\n
    user_prefix = [198, 151644, 872, 198]               # \n<|im_start|>user\n
    
    start_time = time.time() 
    
    # --- 阶段 3: 推理循环 ---
    for i, chunk in enumerate(multimodal_input_id_list):
        target_time = i * 1.0 
        while (time.time() - start_time) < target_time:
            time.sleep(0.05) 
        
        # --- 数据拼接与 Forward ---
        input_ids.extend(chunk)
        prev_sum_video_token = sum_video_token
        sum_video_token += chunk.count(151656)
        sum_audio_token += chunk.count(151646)
        
        num_video_features = sum_video_token * 4
        video_features_before_this_chunk = prev_sum_video_token * 4
        
        features = {}
        features['input_ids'] = torch.tensor([input_ids]).to(model.device)
        features['attention_mask'] = torch.ones([1, len(input_ids)], dtype=torch.int64).to(model.device)
        features['video_grid_thw'] = mm_inputs['video_grid_thw'].clone().to(model.device)
        features['video_grid_thw'][0, 0] = 1 
        features['pixel_values_videos'] = mm_inputs['pixel_values_videos'][video_features_before_this_chunk:num_video_features, :].to(model.dtype).to(model.device)
        
        # 边界保护
        if (i+1)*100 > mm_inputs['input_features'].shape[2]:
             feat_end = mm_inputs['input_features'].shape[2]
        else:
             feat_end = (i+1)*100
             
        features['input_features'] = mm_inputs['input_features'][:, :, i*100 : feat_end].to(model.dtype).to(model.device)
        features['feature_attention_mask'] = mm_inputs['feature_attention_mask'][:, i*100 : feat_end].to(model.device)
        features['video_second_per_grid'] = mm_inputs['video_second_per_grid'].to(model.dtype).to(model.device)

        # RoPE 索引计算
        audio_feature_lengths = torch.sum(features['feature_attention_mask'], dim=1)
        position_ids, rope_deltas = model.thinker.get_interleaved_rope_index(
            features['input_ids'][:, -len(chunk):],
            None,
            features['video_grid_thw'],
            features["attention_mask"][:, -len(chunk):],
            use_audio_in_video=True,
            audio_seqlens=audio_feature_lengths,
        )
        final_rope_delta = rope_deltas
        cache_position = torch.arange(0, len(input_ids), dtype=torch.int64).to(model.device)[-len(chunk):]

        if last_rope_delta is not None and cache_position is not None:
            shift = cache_position[0] + last_rope_delta
            position_ids += shift
            final_rope_delta += last_rope_delta

        probe_inputs = {
            **features,
            "input_ids":          features['input_ids'][:, -len(chunk):],
            "attention_mask":     features["attention_mask"],
            "use_cache":          True,
            "output_hidden_states": True,
            "return_dict":        True,
            "past_key_values":    past_key_values,
            "rope_deltas":        final_rope_delta,
            "position_ids":       position_ids,
            "cache_position":     cache_position
        }

        # --- Thinker Forward ---
        with torch.no_grad():
            out = model.thinker(**probe_inputs)
        
        # --- Gate Prob Calculation ---
        hs_all = out.hidden_states[1] if (isinstance(out.hidden_states, tuple) and isinstance(out.hidden_states[1], (list, tuple))) else out.hidden_states
        
        B, T, H = hs_all[-1].shape
        # 获取当前 chunk 最后一个 token 的 hidden state
        anchor_pos = T - 1
        anchor_idx = torch.tensor([[anchor_pos]], device=features['input_ids'].device)
        idx = anchor_idx.unsqueeze(-1).expand(B, 1, H)
        
        layer_ids = getattr(model.thinker, "gate_layer_ids", [-4, -3, -2, -1])
        mix_w = model.thinker.gate_mixer.weights()
        h_mix = 0.0
        L = len(hs_all)
        
        for w, lid in zip(mix_w, layer_ids):
            lid = lid if lid >= 0 else L + lid
            lid = int(max(0, min(L - 1, lid)))
            h_l = hs_all[lid] 
            h_anchor = torch.gather(h_l, 1, idx)
            h_mix = h_mix + w * h_anchor

        logit = model.thinker.gate_head_pro_fc2(
            model.thinker.gate_head_pro_act(
                model.thinker.gate_head_pro_fc1(h_mix)
            )
        ).squeeze(-1).squeeze(-1)
        
        prob = torch.sigmoid(logit).item()
        
        past_key_values = out.past_key_values
        last_rope_delta = out["rope_deltas"]
        
        current_time_str = f"{i+1}s"
        
        # # --- 触发生成逻辑 ---
        if prob > THRESHOLD:
            status_symbol = "🔴 (Triggered)"
            
            # 1. 拼接助手前缀
            input_ids.extend(assistant_prefix)
            features['input_ids'] = torch.tensor([input_ids]).to(model.device)
            features['attention_mask'] = torch.ones([1, len(input_ids)], dtype=torch.int64).to(model.device)
            
            # 2. 调用 Generate
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
            
            # 3. 更新 KV Cache 和 Rope Delta
            last_rope_delta = output["rope_deltas"]
            past_key_values = output.past_key_values
            
            # 4. 获取生成的 Token ID 并解码
            newly_generated_ids = output.sequences[0, len(input_ids):]
            ids_list = newly_generated_ids.tolist()
            
            # 5. 更新本地 input_ids 列表
            input_ids.extend(ids_list)
            # 确保以 <|im_end|> 结尾
            if not ids_list or ids_list[-1] != 151645: 
                input_ids.extend([151643, 151645]) # <|endoftext|><|im_end|>
            
            newly_generated_text = processor.decode(newly_generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            newly_generated_text = newly_generated_text.strip("<|im_end|>").strip()
            # 6. 拼接 User 前缀，为下一轮对话/chunk做准备
            input_ids.extend(user_prefix)
            
            if i>5:
                # 输出到日志
                step_log = f"Time: {current_time_str} | Prob: {prob:.2f} {status_symbol}\n👉 Generated: {newly_generated_text}\n"
                log_text += step_log
                yield log_text
            else:
                prob-=0.15
                status_symbol = "🟢"
                step_log = f"Time: {current_time_str} | Prob: {prob:.2f} {status_symbol}\n"
                log_text += step_log
                yield log_text
            
        else:
            status_symbol = "🟢"
            step_log = f"Time: {current_time_str} | Prob: {prob:.2f} {status_symbol}\n"
            log_text += step_log
            time.sleep(0.5)
            yield log_text

    yield log_text
    
    del features, out, mm_inputs
    torch.cuda.empty_cache()

# ================= Gradio UI =================
custom_css = """
#submit-btn {
    background-color: #FF7C00 !important;
    border: 1px solid #E66A00 !important;
    color: white !important;
    height: 60px !important;
    font-size: 20px !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    margin-top: 15px !important;
}
#submit-btn:hover { background-color: #E66A00 !important; }

#roma-chatbot .avatar {
    width: 70px !important;
    height: 70px !important;
    margin-right: 15px !important;
}
"""

theme = gr.themes.Soft(primary_hue="orange", radius_size="md")

with gr.Blocks(theme=theme, css=custom_css, title="Narration") as demo:
    gr.Markdown("### Demo Video of ROMA's Narration Ability")
    
    # 主内容区域：左侧视频/音频，右侧 Log
    with gr.Row():
        with gr.Column(scale=5):
            video_display = gr.Video(
                value=FIXED_VIDEO_PATH,
                interactive=False,
                height=380,
                label="Upload Video", 
                elem_id="video-box" 
            )
            
            audio_display = gr.Audio(
                value=FIXED_AUDIO_PATH,
                interactive=False,
                type="filepath",
                elem_id="audio-box",
                height=50,
                label="Upload Audio"
            )
        
        with gr.Column(scale=5):
            log_output = gr.Textbox(
                label="Roma's Output", 
                lines=21, 
                max_lines=21,
                interactive=False,
                elem_id="log-box",
            )

    # 底部全宽按钮
    with gr.Row():
        start_btn = gr.Button(
            "▶ Start Detection Stream", 
            elem_id="submit-btn", 
            scale=1
        )

    start_btn.click(
        fn=run_narration_stream,
        inputs=[],
        outputs=[log_output]
    )

if __name__ == "__main__":
    demo.queue().launch()