import json
import os
from typing import Dict, List, Tuple, Optional, Iterable
from tqdm import tqdm

# ================= Configuration =================
PRED_PATH = "result.jsonl"

GT_PATH = "gt.jsonl"

THRESHOLDS = [0.7, 0.8, 0.9]

WINDOW_SIZE = 1 
# ===========================================================

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
    """Support both JSON and JSONL formats"""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)
        else:
            return load_jsonl(path)

def build_gt_index(gt_items: Iterable[dict]) -> Dict[str, Tuple[int, int]]:
    """Build {video_id: (start, end)} mapping"""
    gt = {}
    for item in gt_items:
        vid = str(item.get("id"))
        
        # Adapt to different GT structures
        seg = None
        ans = item.get("answer", [])
        if ans:
            # Try to obtain the segment field
            seg_field = ans.get("segment")  # Could be [start, end] or [[s,e]]
            if isinstance(seg_field, list) and len(seg_field) > 0:
                if isinstance(seg_field[0], list):  # Nested list [[s, e]]
                    s, e = seg_field[0]
                elif len(seg_field) == 2:  # Direct list [s, e]
                    s, e = seg_field
                else:
                    continue
                seg = (int(s), int(e))
        
        if seg:
            gt[vid] = seg
    return gt

def get_trigger_events(probs: List[float], threshold: float, window: int) -> List[int]:
    """
    Return all trigger timestamps (indices).
    Logic: must have window consecutive frames >= threshold.
    The trigger timestamp is defined as the last frame of the window.
    """
    triggers = []
    
    # If window is 1, directly check
    if window <= 1:
        return [i for i, p in enumerate(probs) if p >= threshold]
    
    # Window-based logic
    if len(probs) < window:
        return []

    for i in range(len(probs) - window + 1):
        chunk = probs[i : i + window]
        if all(p >= threshold for p in chunk):
            triggers.append(i + window - 1)
            
    return triggers

def count_distinct_groups(indices: List[int]) -> int:
    """
    Count how many independent trigger groups exist.
    Example: indices = [10, 11, 12, 50, 51]
    [10,11,12] is group 1 (continuous)
    [50,51] is group 2
    Return 2. This better reflects how "verbose" the model is.
    """
    if not indices:
        return 0
        
    groups = 1
    for i in range(1, len(indices)):
        # If the difference between indices is greater than 1, a new group begins
        if indices[i] - indices[i-1] > 1:
            groups += 1
    return groups

def main():
    print(f"Loading predictions from: {os.path.basename(PRED_PATH)}")
    preds = load_jsonl(PRED_PATH)
    
    print(f"Loading Ground Truth from: {os.path.basename(GT_PATH)}")
    gt_raw = load_json_or_jsonl(GT_PATH)
    gt_index = build_gt_index(gt_raw)
    
    print(f"Total Preds: {len(preds)} | Total GT: {len(gt_index)}")
    
    stats = {thr: {
        'total': 0,
        'correct': 0,
        'early': 0,
        'late': 0,
        'miss': 0,
        'repeated_count': 0  # number of samples with multiple independent alarms
    } for thr in THRESHOLDS}
    
    missing_gt_count = 0
    
    for item in tqdm(preds, desc="Analyzing"):
        vid = str(item.get("id"))
        probs = item.get("raw_probs", [])
        
        if vid not in gt_index:
            missing_gt_count += 1
            continue
            
        start, end = gt_index[vid]
        
        for thr in THRESHOLDS:
            stats[thr]['total'] += 1
            
            # 1. Get all trigger points
            trigger_indices = get_trigger_events(probs, thr, WINDOW_SIZE)
            
            if not trigger_indices:
                # No trigger during the entire sequence -> Miss
                stats[thr]['miss'] += 1
                continue
                
            # 2. Get the first trigger time
            first_idx = trigger_indices[0]
            
            # 3. Determine Early / Late / Correct
            # If the first trigger lies within the interval, it counts as Correct
            if first_idx < start:
                stats[thr]['early'] += 1
            elif first_idx > end:
                stats[thr]['late'] += 1
            else:
                stats[thr]['correct'] += 1
                
            # 4. Determine Repeated Triggers (whether the model is verbose)
            # Count how many independent alarm segments exist
            num_groups = count_distinct_groups(trigger_indices)
            if num_groups > 1:
                stats[thr]['repeated_count'] += 1

    # ================= Output Report =================
    print("\n" + "="*95)
    print(f"TRIGGER ERROR ANALYSIS (Window Size = {WINDOW_SIZE})")
    print(f"Valid Samples: {stats[THRESHOLDS[0]]['total']} | Missing GT: {missing_gt_count}")
    print("="*95)
    
    # Table header
    headers = ["Thr", "Correct(Hit)", "Early", "Late", "Missed", "FP(E+L)", "Repeated%"]
    row_fmt = "{:<6} | {:<12} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    
    print(row_fmt.format(*headers))
    print("-" * 95)
    
    for thr in THRESHOLDS:
        s = stats[thr]
        total = s['total']
        if total == 0: 
            continue
        
        # Calculate percentages
        p_corr = s['correct'] / total * 100
        p_early = s['early'] / total * 100
        p_late = s['late'] / total * 100
        p_miss = s['miss'] / total * 100
        p_fp = p_early + p_late  # False Positive = Early + Late
        p_rep = s['repeated_count'] / total * 100
        
        print(row_fmt.format(
            f"{thr}",
            f"{p_corr:.1f}%",
            f"{p_early:.1f}%",
            f"{p_late:.1f}%",
            f"{p_miss:.1f}%",
            f"{p_fp:.1f}%",
            f"{p_rep:.1f}%"
        ))
    
    print("="*95)
    print("Metrics Definition:")
    print("1. Correct: First trigger occurs inside [Start, End].")
    print("2. Early:   First trigger occurs before Start.")
    print("3. Late:    First trigger occurs after End.")
    print("4. FP:      False Positives (Early + Late).")
    print("5. Repeated%: Percentage of videos that triggered distinct alarms more than once.")
    print("              (e.g., alarm at 10s... stop... alarm again at 50s).")

if __name__ == "__main__":
    main()