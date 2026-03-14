import gradio as gr
import time
import os
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
FIXED_VIDEO_PATH = "videos/XzxRMH7G8Lk_360.0_510.0.mp4"
FIXED_AUDIO_PATH = "audio/pa_audio/154.wav"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.6

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

template_args = Namespace(**{"template": "streaming_turn", "train_on_prompt": False, "tool_format": None})
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
def run_detection_stream():
    """
    1. 预处理
    2. 倒计时
    3. 同步推理
    """
    if not os.path.exists(FIXED_VIDEO_PATH) or not os.path.exists(FIXED_AUDIO_PATH):
        yield "❌ 错误：找不到指定的视频或音频文件。"
        return

    # --- 阶段 1: 数据预处理 ---
    # 为了让前端看到状态，这里必须 yield log_text
    log_text = ""
    yield log_text
    print("⚙️ 正在后台加载视频与预处理数据，请稍候...\n")

    data = {
        "task": "action_prediction",
        "id": "3",
        "videos": [[FIXED_VIDEO_PATH]], # 使用配置的路径
        "query": [
            {
                "text": "<video>Find the part of the video where a man in a yellow bananas shirt is speaking in a room decorated with plants.",
                "audio": FIXED_AUDIO_PATH, # 使用配置的路径
                "time": 0.0,
                "duration": 4.4167 #3.05
            }
        ],
        "images": [],
        "answer": [{"segment": [10, 28]}],
        "ans": [{"text": "", "time": 0.0}]
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
    
    input_ids = []
    sum_video_token = 0
    sum_audio_token = 0
    past_key_values = None
    last_rope_delta = None
    last_prob = 0.0
    
    # --- 阶段 2: 倒计时同步 ---
    # 恢复 yield 以便在网页文本框中显示倒计时
    #log_text += "✅ 数据就绪！请将鼠标放在视频播放键上...\n"
    #yield log_text
    #time.sleep(1)
    
    #log_text += "3...\n"
    print("7...\n")
    time.sleep(1)

    print("6...\n")
    time.sleep(1)

    print("5...\n")
    time.sleep(1)
    print("4...\n")
    time.sleep(1)

    print("3...\n")
    #yield log_text
    time.sleep(1)
    
    #log_text += "2...\n"
    print("2...\n")
    #yield log_text
    time.sleep(1)
    
    #log_text += "🎬 1... GO! 请点击播放! 🎬\n"
    #log_text += f"{'-'*40}\n"
    print("🎬 1...")
    time.sleep(1)

    print("GO! 请点击播放! 🎬\n")
    time.sleep(1)
    #yield log_text
    
    # 记录开始时间（假设用户在看到GO时点击了播放）
    start_time = time.time() 
    
    # --- 阶段 3: 推理循环 ---
    for i, chunk in enumerate(multimodal_input_id_list):
        # 同步逻辑
        target_time = i * 1.0 
        while (time.time() - start_time) < target_time:
            time.sleep(0.05) 
        
        # --- 推理 ---
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
        
        if (i+1)*100 > mm_inputs['input_features'].shape[2]:
             feat_end = mm_inputs['input_features'].shape[2]
        else:
             feat_end = (i+1)*100
             
        features['input_features'] = mm_inputs['input_features'][:, :, i*100 : feat_end].to(model.dtype).to(model.device)
        features['feature_attention_mask'] = mm_inputs['feature_attention_mask'][:, i*100 : feat_end].to(model.device)
        features['video_second_per_grid'] = mm_inputs['video_second_per_grid'].to(model.dtype).to(model.device)

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

        with torch.no_grad():
            out = model.thinker(**probe_inputs)
        
        hs_all = out.hidden_states[1] if (isinstance(out.hidden_states, tuple) and isinstance(out.hidden_states[1], (list, tuple))) else out.hidden_states
        
        layer_ids = getattr(model.thinker, "gate_layer_ids", [-4, -3, -2, -1])
        mix_w = model.thinker.gate_mixer.weights()
        h_mix = 0.0
        L = len(hs_all)
        
        for w, lid in zip(mix_w, layer_ids):
            lid = lid if lid >= 0 else L + lid
            lid = int(max(0, min(L - 1, lid)))
            h_l = hs_all[lid] 
            h_anchor = h_l[:, -1:, :] 
            h_mix = h_mix + w * h_anchor

        logit = model.thinker.gate_head_pro_fc2(
            model.thinker.gate_head_pro_act(
                model.thinker.gate_head_pro_fc1(h_mix)
            )
        ).squeeze(-1).squeeze(-1)
        
        prob = torch.sigmoid(logit).item()
        
        past_key_values = out.past_key_values
        last_rope_delta = out["rope_deltas"]
        
        status_symbol = "🟢"
        if prob > THRESHOLD:
            status_symbol = "🔴 [Alert]"
        
        current_time_str = f"{i+1}s"
        step_log = f"Time: {current_time_str} | Prob: {prob:.2f} {status_symbol}\n"
        log_text += step_log
        #end_time = time.time()
        #print("time-consume:", end_time - start_time)
        time.sleep(0.6)
        yield log_text 

        # if last_prob > THRESHOLD and prob > THRESHOLD:
        #     log_text += "\n" + "="*30 + "\n"
        #     log_text += f"❗❗❗ DETECTED at {current_time_str} (Consecutive > {THRESHOLD}) ❗❗❗\n"
        #     log_text += "="*30 + "\n"
        #     yield log_text
        #     break 
        
        last_prob = prob

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

with gr.Blocks(theme=theme, css=custom_css, title="Proactive Alert") as demo:
    gr.Markdown("### Demo Video of ROMA's Event-Triggered Alert")
    
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
        fn=run_detection_stream,
        inputs=[],
        outputs=[log_output]
    )

if __name__ == "__main__":
    demo.queue().launch()

