# Copyright 2025 the LlamaFactory team.
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

import os
import glob
import shutil
import fire
import torch

from peft import PeftModel
from transformers import AutoModel, AutoProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)

def merge_lora(
    base_model_path: str,
    lora_checkpoint_path: str,
    extra_file: str = "spk_dict.pt",
    submodule_name: str = "thinker",
    save_path: str = "./merged_model_checkpoint",
):
    """Load the original model, tokenizer, and processor configuration, merge the LoRA weights.

    For a specified submodule, and save the final merged model along with its configurations.

    Args:
        base_model_path (str): Path to the original model directory.
        lora_checkpoint_path (str): Path to the directory containing LoRA weights.
        extra_file (str): Name of the extra file to be copied (default: "spk_dict.pt").
        submodule_name (str): Name of the submodule to merge (default: "thinker").
        save_path (str): Directory where the merged model and configurations will be saved.
    """
    # 1. Load the original model, tokenizer, and processor
    model = AutoModel.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    processor = AutoProcessor.from_pretrained(base_model_path)
    print("Successfully loaded the original model and tokenizer.")

    # 2. Extract the submodule to be merged (e.g., model.thinker)
    if not hasattr(model, submodule_name):
        raise AttributeError(f"The model does not have a submodule named '{submodule_name}'.")

    base_submodule = getattr(model, submodule_name)
    print(f"Successfully extracted submodule: {submodule_name}.")

    # 3. Load the LoRA weights onto the extracted submodule
    lora_model = PeftModel.from_pretrained(base_submodule, lora_checkpoint_path)
    print("LoRA weights loaded successfully.")

    # 4. Merge the LoRA weights into the submodule and unload the LoRA modules
    merged_submodule = lora_model.merge_and_unload()
    print("LoRA weights merged successfully.")

    # 5. Replace the original submodule with the merged submodule in the model
    setattr(model, submodule_name, merged_submodule)

    # 6. Save the final merged model along with the tokenizer and processor configuration
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Merged model and tokenizer saved to {save_path}.")

    source_file = os.path.join(base_model_path, extra_file)
    target_file = os.path.join(save_path, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"File '{extra_file}' copied from {base_model_path} to {save_path}.")
    else:
        print(f"File '{extra_file}' not found in {base_model_path}, skipping copy.")


# ------------------------------
# 改进后的 save_full（适用于 full+ZeRO-3）
# ------------------------------
def _ensure_gate_exists(thinker):
    """若类定义默认不带 gate_*，先补挂空模块，便于后续从分片回填权重。"""
    if hasattr(thinker, "gate_head") and hasattr(thinker, "gate_mixer"):
        return

    H = thinker.config.get_text_config().hidden_size
    device = next(thinker.parameters()).device
    dtype = next(thinker.parameters()).dtype

    class GateMixer(torch.nn.Module):
        def __init__(self, K, device, dtype):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(K, device=device, dtype=dtype))
        def weights(self): return torch.softmax(self.logits, dim=0)

    layer_ids = getattr(thinker.config, "gate_layer_ids", [-4, -3, -2, -1])
    K = len(layer_ids)

    thinker.gate_head = torch.nn.Linear(H, 1, bias=True).to(device=device, dtype=dtype)
    thinker.gate_mixer = GateMixer(K=K, device=device, dtype=dtype)
    thinker.gate_layer_ids = layer_ids


def _try_load_gate_from_shards(thinker, path):
    """从 safetensors 分片/单文件中抽取 gate_* 权重并加载到 thinker（若存在）。"""
    try:
        from safetensors import safe_open
    except Exception:
        print("⚠️ safetensors 未安装或不可用，跳过从分片回填 gate_*。")
        return

    # 收集可能存在的 safetensors 文件：分片或单体
    files = []
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        files.append(single)
    else:
        files.extend(sorted(glob.glob(os.path.join(path, "model-*.safetensors"))))

    if not files:
        print("⚠️ 未发现 safetensors 权重文件，跳过 gate_* 回填（from_pretrained 可能已加载）。")
        return

    head_sd, mixer_sd = {}, {}
    for fp in files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("gate_head."):
                    head_sd[k.split("gate_head.", 1)[1]] = f.get_tensor(k)
                elif k.startswith("gate_mixer."):
                    mixer_sd[k.split("gate_mixer.", 1)[1]] = f.get_tensor(k)
                elif k.startswith("thinker.gate_head."):
                    head_sd[k.split("thinker.gate_head.", 1)[1]] = f.get_tensor(k)
                elif k.startswith("thinker.gate_mixer."):
                    mixer_sd[k.split("thinker.gate_mixer.", 1)[1]] = f.get_tensor(k)

    if head_sd:
        thinker.gate_head.load_state_dict(head_sd, strict=False)
    if mixer_sd:
        thinker.gate_mixer.load_state_dict(mixer_sd, strict=False)

    if head_sd or mixer_sd:
        print("✅ 从分片/单体 safetensors 回填 gate_head / gate_mixer 完成")
    else:
        print("ℹ️ 未在分片中找到 gate_* 权重键，可能已由 from_pretrained 自动加载。")


def save_full_model(
    saved_thinker_path: str,
    base_model_path: str,
    save_path: str = "./merged_model_checkpoint",
    extra_file: str = "spk_dict.pt",
):
    """加载你 full 训练得到的 thinker（包含新增 gate_*），替换 base 顶层的 thinker，并整体保存。"""
    # 1) 加载 full finetune 产出的 thinker
    print("加载 thinker（full finetune 权重）...")
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        saved_thinker_path,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,  # Qwen Omni 需要
    )

    # 2) 兜底：若类定义默认无 gate_*，先补挂后尝试从 safetensors 回填
    _ensure_gate_exists(thinker)
    _try_load_gate_from_shards(thinker, saved_thinker_path)

    # 3) 加载顶层 base 模型，替换其 thinker
    print("加载 base 顶层模型...")
    base_model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,  # 顶层同样需要
    )
    if not hasattr(base_model, "thinker"):
        raise AttributeError("base 模型不包含 'thinker' 子模块，检查 base_model_path 是否为 Qwen2.5-Omni 顶层。")

    base_model.thinker = thinker

    # 4) 处理器沿用 base（若未改 tokenizer/processor，这是最稳）
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # 5) 保存（safetensors 分片）
    os.makedirs(save_path, exist_ok=True)
    base_model.save_pretrained(
        save_path,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    processor.save_pretrained(save_path)
    print(f"✅ 模型与处理器已保存至 {save_path}")

    # 6) 复制可选文件
    src = os.path.join(base_model_path, extra_file)
    dst = os.path.join(save_path, extra_file)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"已复制额外文件 '{extra_file}' 到保存目录。")
    else:
        print(f"未在 base_model_path 找到 '{extra_file}'，跳过复制。")

    # 7) 简单自检：索引文件中是否包含 gate_* 关键词
    index_fp = os.path.join(save_path, "model.safetensors.index.json")
    if os.path.exists(index_fp):
        try:
            text = open(index_fp, "r", encoding="utf-8").read()
            hit = ("thinker.gate_head." in text) or ("thinker.gate_mixer." in text) or \
                  ("gate_head." in text) or ("gate_mixer." in text)
            print(f"自检：索引中 gate_* 路径检测 {'✅' if hit else '⚠️ 未检出（可能键名不同，建议实际加载验证）'}")
        except Exception as e:
            print(f"索引自检跳过：{e}")


if __name__ == "__main__":
    fire.Fire({"save_full": save_full_model, "merge_lora": merge_lora})
