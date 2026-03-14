import json
import os
from typing import Dict, List, Tuple, Optional, Iterable
from tqdm import tqdm

# ================= Configuration =================
PRED_PATH = "eval/proactive/streaming_po/new_results/spo_resut.jsonl"
GT_PATH = "spo_gt.jsonl"

THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
WINDOW_SIZE = 5
# =================================================

def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def load_json_or_jsonl(path: str) -> List[dict]:
    """Support both JSON array and JSONL formats"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            return json.load(f)
        else:
            return load_jsonl(path)

def build_gt_index(gt_items: Iterable[dict]) -> Dict[str, Tuple[int, int]]:
    """Build GT index {id: (start, end)}"""
    gt = {}
    for it in gt_items:
        vid = str(it.get("id"))
        seg = None
        ans = it.get("answer", [])
        
        # Parsing logic for your dataset structure
        if isinstance(ans, dict):  # Sometimes it may be a direct dictionary
            ans_list = [ans]
        elif isinstance(ans, list):
            ans_list = ans
        else:
            continue

        if ans_list:
            # Try to obtain segment
            # Support both {"segment": [s,e]} and {"segment": [[s,e]]}
            first_ans = ans_list[0]
            seg_field = first_ans.get("segment")
            
            if isinstance(seg_field, list) and len(seg_field) > 0:
                if isinstance(seg_field[0], list):  # Nested format [[s, e]]
                    s, e = seg_field[0]
                elif len(seg_field) == 2 and isinstance(seg_field[0], (int, float)):  # Direct format [s, e]
                    s, e = seg_field
                else:
                    continue  # Invalid format
                seg = (int(s), int(e))
        
        if seg is not None:
            gt[vid] = seg
    return gt

def get_sliding_triggers(raw: List[float], thr: float, w: int) -> List[int]:
    """
    Return all trigger points filtered by sliding window (using the window end index).
    Logic: if w consecutive points >= threshold, record the index of the w-th point.
    """
    if w <= 0 or w > len(raw):
        return []
    
    triggers = []
    count = 0
    
    # 1. First window [0, w-1]
    for i in range(w):
        if raw[i] >= thr:
            count += 1
    if count == w:
        triggers.append(w - 1)
        
    # 2. Sliding
    for end in range(w, len(raw)):
        # Remove left element
        if raw[end - w] >= thr:
            count -= 1
        # Add right element
        if raw[end] >= thr:
            count += 1
            
        if count == w:
            triggers.append(end)
            
    return triggers

def count_distinct_groups(indices: List[int]) -> int:
    """
    Count the number of independent trigger groups.
    [10, 11, 12] -> 1 group
    [10, 11, 50, 51] -> 2 groups
    """
    if not indices:
        return 0
    groups = 1
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > 1:
            groups += 1
    return groups

def main():
    print(f"Loading predictions from: {PRED_PATH}")
    preds = load_jsonl(PRED_PATH)

    print(f"Loading ground-truth from: {GT_PATH}")
    gt_items = load_json_or_jsonl(GT_PATH)
    gt = build_gt_index(gt_items)

    # Statistics container
    # Structure: {thr: {'total': 0, 'correct': 0, 'early': 0, 'late': 0, 'miss': 0, 'repeated': 0}}
    stats = {thr: {
        'total': 0,
        'correct': 0,
        'early': 0,
        'late': 0,
        'miss': 0,
        'repeated': 0  # Number of videos where multiple independent alarms occurred
    } for thr in THRESHOLDS}

    missing_gt_count = 0
    used_count = 0

    for item in tqdm(preds, desc="Evaluating"):
        vid = str(item.get("id"))
        raw = item.get("raw_probs", [])
        if not isinstance(raw, list) or not raw:
            continue

        seg = gt.get(vid)
        if seg is None:
            missing_gt_count += 1
            continue

        used_count += 1
        start, end = seg

        for thr in THRESHOLDS:
            stats[thr]['total'] += 1
            
            # Use sliding window to obtain all trigger points
            # If sliding window is not desired, set WINDOW_SIZE = 1
            triggers = get_sliding_triggers(raw, thr, WINDOW_SIZE)
            
            # 1. Determine Missed
            if not triggers:
                stats[thr]['miss'] += 1
                continue
            
            # 2. Get the first trigger time (First Trigger)
            # Used to determine Correct / Early / Late
            first_idx = triggers[0]
            
            if first_idx < start:
                stats[thr]['early'] += 1
            elif first_idx > end:
                stats[thr]['late'] += 1
            else:
                stats[thr]['correct'] += 1
                
            # 3. Determine Repeated triggers (verbosity level)
            # Count how many independent alarm groups exist
            num_groups = count_distinct_groups(triggers)
            if num_groups > 1:
                stats[thr]['repeated'] += 1

    print("\n" + "=" * 95)
    print(f"ERROR ANALYSIS REPORT (Window Size = {WINDOW_SIZE})")
    print(f"Total Preds: {len(preds)} | Used (Has GT): {used_count} | Missing GT: {missing_gt_count}")
    print("=" * 95)

    # Print table
    # Early Trigger | Late Trigger | Correct | Missed | Repeated Rate
    header = f"{'Thr':<5} | {'Acc/Correct':<12} | {'Early':<10} | {'Late':<10} | {'Missed':<10} | {'Repeated%':<10}"
    print(header)
    print("-" * len(header))

    for thr in THRESHOLDS:
        s = stats[thr]
        total = s['total']
        if total == 0:
            continue

        p_corr = s['correct'] / total * 100
        p_early = s['early'] / total * 100
        p_late = s['late'] / total * 100
        p_miss = s['miss'] / total * 100
        p_rep = s['repeated'] / total * 100

        print(f"{thr:<5.1f} | {p_corr:<12.1f} | {p_early:<10.1f} | {p_late:<10.1f} | {p_miss:<10.1f} | {p_rep:<10.1f}")
    
    print("=" * 95)
    print("Metrics Definition:")
    print("1. Correct: First trigger (after window) falls inside [Start, End].")
    print("2. Early:   First trigger falls before Start.")
    print("3. Late:    First trigger falls after End.")
    print("4. Missed:  No trigger detected throughout the video.")
    print("5. Repeated%: Percentage of videos where distinct triggers occurred more than once.")

if __name__ == "__main__":
    main()