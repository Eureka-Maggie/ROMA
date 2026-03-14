#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

IN_PATH = "sqa.jsonl"


LETTER_RE = re.compile(r"(?i)(?:^|[\s:：\(\[\{<]+)([ABCD])(?:[\s\.\):：\]\}>]|$)")

def parse_options(question_text: str) -> Dict[str, str]:
    """
    从 question 字符串中解析 (A) ... (B) ... (C) ... (D) ... 的选项文本
    返回 {'A': '...', 'B': '...', ...}（已 strip）
    """
    if not question_text:
        return {}
    # 允许 (A) 或 A) 或 A. 等常见形式；用 non-greedy 拉到下一个选项或结尾
    pat = re.compile(
        r"(?is)(?:\(\s*([ABCD])\s*\)|^\s*([ABCD])[\)\.])\s*(.*?)\s*(?=(?:\(\s*[ABCD]\s*\)|^\s*[ABCD][\)\.]|\Z))",
        re.MULTILINE
    )
    opts: Dict[str, str] = {}
    for g1, g2, body in pat.findall(question_text):
        k = (g1 or g2).upper()
        if k and k not in opts:
            opts[k] = " ".join(body.strip().split())
    return opts

def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def extract_choice_from_pred(pred_text: str, options: Dict[str, str], gt_choice: Optional[str]) -> Optional[str]:
    """
    1) 优先从 pred_text 里抽 A/B/C/D（支持 A, A., A:..., (A) 等）
    2) 若没抽到，尝试用“包含选项文本/gt文本”进行匹配（复述内容也算对上）
    """
    if pred_text is None:
        return None
    t = pred_text.strip()
    if not t:
        return None

    # 1) 直接抽字母
    m = LETTER_RE.search(t)
    if m:
        return m.group(1).upper()

    # 额外：有些模型会输出 "A)" / "A." 开头
    m2 = re.match(r"(?i)^\s*([ABCD])\s*[\)\.\:：]\s*", t)
    if m2:
        return m2.group(1).upper()

    # 2) 文本包含匹配：如果 pred 复述了某个选项内容，则映射回该选项字母
    norm_pred = normalize_text(t)
    if options:
        # 先匹配最长选项（避免短句误匹配）
        items = sorted(options.items(), key=lambda kv: len(kv[1]), reverse=True)
        for k, opt_text in items:
            norm_opt = normalize_text(opt_text)
            if norm_opt and (norm_opt in norm_pred or norm_pred in norm_opt):
                return k

    # 3) 兜底：如果 pred 直接把 gt 的文本重复出来（有些数据 gt.text）
    # 这里不需要完整等于，包含即可（同样走字符串归一）
    # 注意：gt_choice 不一定有对应文本，所以只能靠 options 或者 pred 自己的字母形式
    return None

def get_gt_choice(obj: dict) -> Optional[str]:
    gt = obj.get("gt")
    if isinstance(gt, dict):
        c = gt.get("choice")
        if isinstance(c, str) and c.strip():
            return c.strip().upper()
    # 兼容 answer 字段
    ans = obj.get("answer")
    if isinstance(ans, dict):
        c = ans.get("choice")
        if isinstance(c, str) and c.strip():
            return c.strip().upper()
    return None

def get_pred_text(obj: dict) -> Optional[str]:
    pred = obj.get("prediction")
    # 常见：prediction 是 list[{"time":..., "text":...}]，取最后一个
    if isinstance(pred, list) and pred:
        last = pred[-1]
        if isinstance(last, dict):
            t = last.get("text")
            return t if isinstance(t, str) else None
        if isinstance(last, str):
            return last
    # 兜底：prediction 是 str
    if isinstance(pred, str):
        return pred
    return None

def main():
    path = Path(IN_PATH)
    total = 0
    correct = 0
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scoring", unit="lines"):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            gt_choice = get_gt_choice(obj)
            pred_text = get_pred_text(obj)
            q = obj.get("question", "")
            options = parse_options(q) if isinstance(q, str) else {}

            pred_choice = extract_choice_from_pred(pred_text or "", options, gt_choice)

            if not gt_choice or gt_choice not in {"A", "B", "C", "D"}:
                skipped += 1
                continue

            total += 1
            if pred_choice == gt_choice:
                correct += 1

    acc = (correct / total) if total else 0.0
    print(f"File: {IN_PATH}")
    print(f"Total (scored): {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Skipped (empty/invalid/no-gt/bad-json): {skipped}")

if __name__ == "__main__":
    main()
