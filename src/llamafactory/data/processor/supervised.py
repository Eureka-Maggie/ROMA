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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    SIL_TOKENS: list[int] | None = None
    def _encode_data_example(
        self,
        task: str,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        cls = type(self)
        sil = type(self).SIL_TOKENS
        if sil is None:
            # "<|silence|>" 的 token 序列（不是 special token）
            sil = self.tokenizer.encode("<|silence|>", add_special_tokens=False)
            type(self).SIL_TOKENS = sil
        messages = self.template.mm_plugin.process_messages([[prompt,response]], images, videos, audios, self.processor) # Qwen2_5OmniProcessor #### 需要改！！！
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools) #[{'content': '<|vision_bos|><|audio_bos|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|VIDEO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|AUDIO|><|audio_eos|><|vision_eos|>What is the video describing?', 'role': 'user'}, {'content': 'A girl who is drawing a picture of a guitar and feel nervous.', 'role': 'assistant'}]
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history: #false
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns
        input_ids = []
        labels = []         
        anchor_idx_list = []    
        gate_label_list = []    
        def startswith_silence(toks, sil=sil):
            return len(toks) >= len(sil) and toks[:len(sil)] == sil
# # ################# 混合 ################
        if cls.SIL_TOKENS is None:
            cls.SIL_TOKENS = self.tokenizer.encode("<|silence|>", add_special_tokens=False)
        SIL = cls.SIL_TOKENS

        if getattr(cls, "VISION_EOS_TOKENS", None) is None:
            cls.VISION_EOS_TOKENS = self.tokenizer.encode("<|vision_eos|>", add_special_tokens=False)
        VEOS = cls.VISION_EOS_TOKENS
        VEOS_L = len(VEOS)

        if getattr(cls, "ASSIST_TOKENS", None) is None:
            cls.ASSIST_TOKENS = self.tokenizer.encode(
                "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
            )
        ASSIST = cls.ASSIST_TOKENS  

        if getattr(cls, "USER_TOKENS", None) is None:
            cls.USER_TOKENS = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        USER = cls.USER_TOKENS 
        def find_last_veos_position(source_ids: list[int], veos: list[int]) -> int:
            """只返回最后一次出现的“pattern 末尾”索引；找不到则返回 -1"""
            L = len(veos)
            if L == 0 or len(source_ids) < L:
                return -1
            # 反向扫描，第一次命中就是最后一次出现
            for i in range(len(source_ids) - L, -1, -1):
                if source_ids[i:i+L] == veos:
                    return i + L - 1
            return -1    
        def is_alert_example(prompt_msg, resp_msg) -> bool:
            if len(resp_msg) > 0 and isinstance(resp_msg[0].get("text", ""), str):
                return resp_msg[0]["text"].lstrip().startswith("alert")
            return False

        is_alert = is_alert_example(prompt, response)
        chunk_gate_labels: list[float] = []
        for _src_ids, tgt_ids in encoded_pairs:
            t = list(tgt_ids)
            should_speak = (len(t) > 0) and (not startswith_silence(t))
            chunk_gate_labels.append(1.0 if should_speak else 0.0)

        if is_alert:
            new_pairs: list[tuple[list[int], list[int]]] = []
            for i, (src_ids, tgt_ids) in enumerate(encoded_pairs):
                src = list(src_ids)
                tgt = list(tgt_ids)

                if chunk_gate_labels[i] == 1.0 and len(tgt) > 0:
                    if ASSIST and len(src) >= len(ASSIST):
                        if src[-len(ASSIST) :] == ASSIST:
                            src = src[: -len(ASSIST)]

                    tgt = []

                    if i + 1 < len(encoded_pairs):
                        next_src, next_tgt = encoded_pairs[i + 1]
                        next_src = list(next_src)
                        if USER and len(next_src) >= len(USER):
                            if next_src[: len(USER)] == USER:
                                next_src = next_src[len(USER) :]
                        encoded_pairs[i + 1] = (next_src, list(next_tgt))

                new_pairs.append((src, tgt))

            encoded_pairs = new_pairs
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            # 截断长度
            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = list(source_ids[:source_len])
            target_ids = list(target_ids[:target_len])

            # ---------- source_label ----------
            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            # ---------- target_label----------
            if self.data_args.mask_history and turn_idx != 0:
                target_label = [IGNORE_INDEX] * target_len
            # else:
            #     # gate-only：无论 alert 还是 narration，都不训 LM
            #     target_label = [IGNORE_INDEX] * target_len
            else:
                if 151643 in target_ids: 
                    if task == "turn-taking":
                        target_label = target_ids[:-3] + [IGNORE_INDEX] * 3
                    else:
                        target_label = [IGNORE_INDEX] * target_len
                else:
                    # Narration：若以 <|silence|> 开头，该轮不训练 LM
                    ##target_label = [IGNORE_INDEX] * target_len if startswith_silence(target_ids) else target_ids
                    if task == "turn-taking":
                        target_label = [IGNORE_INDEX] * target_len if startswith_silence(target_ids) else target_ids
                    else:
                        target_label = [IGNORE_INDEX] * target_len

            # ---------- flatten ----------
            base_len = len(input_ids)
            input_ids += source_ids + target_ids
            labels += source_label + target_label
            total_length += len(source_ids) + len(target_ids)

            # ---------- time head：anchor + gate ----------
            veos_pos = find_last_veos_position(source_ids, VEOS)
            if veos_pos != -1:
                anchor_abs = base_len + veos_pos
                anchor_idx_list.append(anchor_abs)
                gate_label_list.append(chunk_gate_labels[turn_idx])

        # efficient_eos 收尾
        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        if task == "turn-taking":
            pos_anchor_idxs = [a for a, y in zip(anchor_idx_list, gate_label_list) if y == 1.0]
            pos_gate_label_list = [1.0] * len(pos_anchor_idxs)
            return input_ids, labels, pos_anchor_idxs, pos_gate_label_list, [[prompt, response]]

        return input_ids, labels, anchor_idx_list, gate_label_list, [[prompt, response]]

################# 训 narration 的###################        
        # # "<|silence|>" 的 token 序列（非 special token，按文本编码）
        # if getattr(cls, "SIL_TOKENS", None) is None:
        #     cls.SIL_TOKENS = self.tokenizer.encode("<|silence|>", add_special_tokens=False)
        # SIL = cls.SIL_TOKENS

        # # "<|vision_eos|>" 的 token 序列
        # if getattr(cls, "VISION_EOS_TOKENS", None) is None:
        #     cls.VISION_EOS_TOKENS = self.tokenizer.encode("<|vision_eos|>", add_special_tokens=False)
        # VEOS = cls.VISION_EOS_TOKENS
        # VEOS_L = len(VEOS)

        # if self.data_args.mask_history:
        #     encoded_pairs = encoded_pairs[::-1]
        # def find_last_veos_position(source_ids: list[int], veos: list[int]) -> int:
        #     """只返回最后一次出现的“pattern 末尾”索引；找不到则返回 -1"""
        #     L = len(veos)
        #     if L == 0 or len(source_ids) < L:
        #         return -1
        #     # 反向扫描，第一次命中就是最后一次出现
        #     for i in range(len(source_ids) - L, -1, -1):
        #         if source_ids[i:i+L] == veos:
        #             return i + L - 1
        #     return -1
        
        # for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        #     if total_length >= self.data_args.cutoff_len:
        #         break

        #     # 截断本轮可用长度
        #     source_len, target_len = infer_seqlen(
        #         len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
        #     )
        #     source_ids = list(source_ids[:source_len])
        #     target_ids = list(target_ids[:target_len])

        #     # -------- SFT: 构造 labels --------
        #     if self.data_args.train_on_prompt:
        #         source_label = source_ids
        #     elif self.template.efficient_eos:
        #         source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        #     else:
        #         source_label = [IGNORE_INDEX] * source_len

        #     if self.data_args.mask_history and turn_idx != 0:
        #         # 仅训练最后一轮（若启用）
        #         target_label = [IGNORE_INDEX] * target_len
        #     else:
        #         # 一些你的自定义规则保留
        #         if 151643 in target_ids:  # 你之前的 “没生成完句子” 特判
        #             #target_label = target_ids[:-3] + [IGNORE_INDEX] * 3
        #             target_label = [IGNORE_INDEX] * target_len
        #         # elif turn_idx == 0 and (messages[0]["content"] == "Narration History"):
        #         #     target_label = [IGNORE_INDEX] * target_len
        #         else:
        #             # Narration：若以 <|silence|> 开头，该轮不训练 LM
        #             ##target_label = [IGNORE_INDEX] * target_len if startswith_silence(target_ids) else target_ids
        #             target_label = [IGNORE_INDEX] * target_len

        #     # 将本轮拼到扁平序列
        #     base_len = len(input_ids)
        #     input_ids += source_ids + target_ids
        #     labels += source_label + target_label
        #     total_length += len(source_ids) + len(target_ids)

        #     # -------- Time Head: 记录锚点与标签 --------
        #     # 该 turn 是否“应该说话”：target 非空 且 不以 <|silence|> 开头
        #     should_speak = (target_len > 0) and (not startswith_silence(target_ids))

        #     # 找出本轮 source 内最后一个 <|vision_eos|> 的位置（相对 source_ids）
        #     veos_pos = find_last_veos_position(source_ids,VEOS)

        #     # 将这些位置映射到扁平后的 input_ids 绝对下标
        #     anchor_abs = base_len + veos_pos  # pos 落在 source 段
        #     anchor_idx_list.append(anchor_abs)
        #     gate_label_list.append(1.0 if should_speak else 0.0)

        # # 可选：efficient_eos 末尾补 eos
        # if self.template.efficient_eos:
        #     input_ids += [self.tokenizer.eos_token_id]
        #     labels += [self.tokenizer.eos_token_id]

        # # 返回给上层（collator 负责把 anchor/label pad 成等长，并生成 gate_mask）
        # return input_ids, labels, anchor_idx_list, gate_label_list, [[prompt, response]]        
################# 训 narration 的###################

################# 训 time head 的（proactive）###################
        # 1) 先记录“原始”每个 chunk 是否有 target（有 target = 需要开口）
        # chunk_gate_labels: list[float] = []
        # for (src_ids, tgt_ids) in encoded_pairs:
        #     chunk_gate_labels.append(1.0 if (len(tgt_ids) > 0 and not startswith_silence(tgt_ids)) else 0.0)

        # cls = type(self)
        # # 用 tokenizer 直接拿模板 token 序列，避免硬编码 ID
        # if getattr(cls, "ASSIST_TOKENS", None) is None:
        #     # 对应 "<|im_end|>\n<|im_start|>assistant\n"
        #     cls.ASSIST_TOKENS = self.tokenizer.encode(
        #         "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        #     )
        # if getattr(cls, "USER_TOKENS", None) is None:
        #     # 对应 "<|im_start|>user\n"
        #     cls.USER_TOKENS = self.tokenizer.encode(
        #         "<|im_start|>user\n", add_special_tokens=False
        #     )
        # if getattr(cls, "VISION_EOS_TOKENS", None) is None:
        #     # 对应 "<|vision_eos|>"，用来确定 anchor 的位置
        #     cls.VISION_EOS_TOKENS = self.tokenizer.encode(
        #         "<|vision_eos|>", add_special_tokens=False
        #     )

        # assist_tokens = cls.ASSIST_TOKENS
        # user_tokens = cls.USER_TOKENS
        # vision_tokens = cls.VISION_EOS_TOKENS

        # # 2) 对 encoded_pairs 做一次 pass：
        # #    - 对于原来有 target 的 chunk：
        # #         * 从 source 末尾切掉 "<|im_end|>\n<|im_start|>assistant\n"
        # #         * 把 target_ids 清空（不训练 LM）
        # #         * 把下一条 source 开头的 "<|im_start|>user\n" 切掉
        # new_pairs: list[tuple[list[int], list[int]]] = []
        # for i, (src_ids, tgt_ids) in enumerate(encoded_pairs):
        #     src_ids = list(src_ids)
        #     tgt_ids = list(tgt_ids)

        #     #if chunk_gate_labels[i] == 1.0 and tgt_ids:  # 原本在这个 chunk 有 assistant 文本 → label=1
        #         # 切掉 source 末尾的 "<|im_end|>\n<|im_start|>assistant\n"
        #         # if assist_tokens and len(src_ids) >= len(assist_tokens):
        #         #     if src_ids[-len(assist_tokens):] == assist_tokens:
        #         #         src_ids = src_ids[:-len(assist_tokens)]

        #         # 只训练 time head，不训练 LM head → target 清空
        #         # tgt_ids = []

        #         # 下一条 pair[i+1]，如果以 "<|im_start|>user\n" 开头，就切掉这个前缀
        #         # if i + 1 < len(encoded_pairs):
        #         #     next_src, next_tgt = encoded_pairs[i + 1]
        #         #     next_src = list(next_src)
        #         #     if user_tokens and len(next_src) >= len(user_tokens):
        #         #         if next_src[:len(user_tokens)] == user_tokens:
        #         #             next_src = next_src[len(user_tokens):]
        #         #     encoded_pairs[i + 1] = (next_src, list(next_tgt))

        #     new_pairs.append((src_ids, tgt_ids))

        # encoded_pairs = new_pairs

        # # ===============================
        # #       flatten + label 构造
        # # ===============================
        # for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        #     if total_length >= self.data_args.cutoff_len:
        #         break

        #     source_len, target_len = infer_seqlen(
        #         len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
        #     )
        #     source_ids = source_ids[:source_len]
        #     target_ids = target_ids[:target_len]

        #     if self.data_args.train_on_prompt:  # prompt部分也要预测，也就是预训练的那种模式
        #         source_label = source_ids
        #     elif self.template.efficient_eos:
        #         source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        #     else:
        #         source_label = [IGNORE_INDEX] * source_len

        #     if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
        #         target_label = [IGNORE_INDEX] * target_len
        #     else:
        #         if 151643 in target_ids:  ##### 没生成完句子。
        #             target_label = target_ids[:-3] + [IGNORE_INDEX] * 3
        #         elif turn_idx == 0 and (messages[0]["content"] == "Narration History"):
        #             target_label = [IGNORE_INDEX] * target_len
        #         else:
        #             # 这里即便 startswith_silence 逻辑还在，对 proactive_gate 来说，
        #             # 上面我们已经把所有有文本的 target 清空了，所以不会再训练 LM。
        #             target_label = target_ids   #[IGNORE_INDEX] * target_len if startswith_silence(target_ids) else target_ids

        #     total_length += len(source_ids) + len(target_ids)

        #     if self.data_args.mask_history:  # false # reversed sequences
        #         base_len = len(input_ids)
        #         input_ids = source_ids + target_ids + input_ids
        #         labels = source_label + target_label + labels
        #     else:
        #         base_len = len(input_ids)
        #         input_ids += source_ids + target_ids
        #         labels += source_label + target_label

        #     # # ===============================
        #     # #       Time head anchor & label
        #     # # ===============================
        #     # # 在当前 chunk 的 source 里找到最后一个 "<|vision_eos|>"，以它作为 anchor；
        #     # # label 由 chunk_gate_labels[turn_idx] 决定：
        #     # #   - 原来有 target 的 chunk → 1.0（需要开口）
        #     # #   - 原来没有 target 的 chunk → 0.0（不需要开口）
        #     # if vision_tokens and source_len >= len(vision_tokens):
        #     #     L = len(vision_tokens)
        #     #     pos = -1
        #     #     # 从后往前找，可以保证拿到最后一个 vision_eos
        #     #     for idx in range(source_len - L, -1, -1):
        #     #         if source_ids[idx : idx + L] == vision_tokens:
        #     #             pos = idx + L - 1  # pattern 最后一个 token 的位置
        #     #             break

        #     #     if pos != -1:
        #     #         anchor_idx = base_len + pos
        #     #         anchor_idx_list.append(anchor_idx)
        #     #         gate_label_list.append(chunk_gate_labels[turn_idx])

        # if self.template.efficient_eos:  # false
        #     input_ids += [self.tokenizer.eos_token_id]
        #     labels += [self.tokenizer.eos_token_id]

        # return input_ids, labels, [[prompt, response]] #anchor_idx_list, gate_label_list,
################# 训 time head 的###################

################# 训正常SFT的(turn-taking)###################
        # for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        #     if total_length >= self.data_args.cutoff_len:
        #         break

        #     source_len, target_len = infer_seqlen(
        #         len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
        #     )
        #     source_ids = source_ids[:source_len]
        #     target_ids = target_ids[:target_len]
        #     ####total_length += source_len + target_len

        #     if self.data_args.train_on_prompt: #prompt部分也要预测，也就是预训练的那种模式
        #         source_label = source_ids
        #     elif self.template.efficient_eos:
        #         source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        #     else:
        #         source_label = [IGNORE_INDEX] * source_len #######

        #     if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
        #         target_label = [IGNORE_INDEX] * target_len
        #     else:
        #         if 151643 in target_ids: ##### 没生成完句子。
        #             target_label = target_ids[:-3]+[IGNORE_INDEX]*3
        #         else:
        #             target_label = target_ids #[IGNORE_INDEX] * target_len if startswith_silence(target_ids) else target_ids 
        #     total_length += len(source_ids)+len(target_ids)
        #     if self.data_args.mask_history:  # false # reversed sequences
        #         input_ids = source_ids + target_ids + input_ids
        #         labels = source_label + target_label + labels
        #     else:
        #         base_len = len(input_ids)
        #         input_ids += source_ids + target_ids
        #         labels += source_label + target_label

        #     is_assistant_turn = (target_len > 0)
        #     if is_assistant_turn:
        #         # 锚点 = 本轮 source 的最后一个 token（assistant\n 的 '\n' 位）
        #         #anchor_idx = base_len + source_len - 1
        #         # 锚点 = 本轮source中最后一个<vision_eos>的位置
        #         anchor_idx = base_len + source_len - 6
        #         # gate 标签：仅当 target 以 <|silence|> 开头时视为沉默
        #         #is_sil = startswith_silence(target_ids)
        #         #anchor_idx_list.append(anchor_idx)
        #         #gate_label_list.append(0.0 if is_sil else 1.0)

        # if self.template.efficient_eos: #false
        #     input_ids += [self.tokenizer.eos_token_id]
        #     labels += [self.tokenizer.eos_token_id]
        # #print('prompt:',prompt,'input_ids:',input_ids)
        # return input_ids, labels, [[prompt,response]] #input_ids, labels, anchor_idx_list, gate_label_list, [[prompt,response]]
################# 训正常SFT的###################

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            #     logger.warning_rank0(
            #         "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            #     )
            #     continue

            input_ids, labels, anchor_idx_list, gate_label_list, messages = self._encode_data_example( ###### tokenize
                task=examples["_task"][i],
                prompt=examples["_prompt"][i], #query
                response=examples["_response"][i], #ans
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids) #[151544, 8948, xxxxx]
            model_inputs["attention_mask"].append([1] * len(input_ids)) #[[1,1,1,.....]]
            model_inputs["labels"].append(labels) #[-100,xxxxx]
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            model_inputs['messages'].append(messages)
            model_inputs['anchor_idx_list'].append(anchor_idx_list)
            model_inputs['gate_label_list'].append(gate_label_list)
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        #print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        #print("label_ids:\n{}".format(example["labels"]))
        #print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        print('PackedSupervisedDatasetProcessor')
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels,_ = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
