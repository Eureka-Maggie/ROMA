#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm

PRED_PATH = "eval/proactive/charades/results/test.jsonl"
GT_PATH = "xxxx/proactive/charades-sta/gt.jsonl"

THRESHOLDS = [0.4,0.5,0.6,0.7]

WINDOW_SIZES = [2, 3, 4, 5]

IOU_THRESHOLDS = [0.5, 0.7]


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_ground_truth(path):
    """
    Return: {id_str: segments}
    segments: [[start, end], ...], which are interpreted directly as frame indices.
    """
    gt = {}
    for item in load_jsonl(path):
        vid = str(item["id"])
        segments = []
        if "answer" in item and item["answer"]:
            segs = item["answer"][0].get("segment", [])
            segments = segs
        gt[vid] = segments
    return gt


def minmax_normalize(nums):
    if not nums:
        return []
    mn = min(nums)
    mx = max(nums)
    if mx == mn:
        # All scores are identical, cannot distinguish them; set all to 0
        return [0.0 for _ in nums]
    return [(x - mn) / (mx - mn) for x in nums]


def smooth_scores(scores, w):
    """
    For each frame i, take the mean score of frames within [i-w, i+w]
    as the smoothed score. Boundary conditions are automatically clipped.
    """
    n = len(scores)
    out = [0.0] * n
    for i in range(n):
        l = max(0, i - w)
        r = min(n - 1, i + w)
        window = scores[l : r + 1]
        out[i] = sum(window) / len(window)
    return out


def segments_to_frame_labels(segments, num_frames):
    """
    Generate per-frame 0/1 labels from segment annotations.
    Here [start, end] is treated as a closed interval (both ends included).
    """
    labels = [0] * num_frames
    for seg in segments:
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            continue
        s, e = seg
        s = int(s)
        e = int(e)
        if e < 0 or s >= num_frames:
            continue
        s = max(0, s)
        e = min(num_frames - 1, e)
        for i in range(s, e + 1):
            labels[i] = 1
    return labels


def frame_iou(pred, labels):
    """
    Frame-level IoU: treat all frames with value 1 as a set
    and compute IoU (intersection / union).
    """
    assert len(pred) == len(labels)
    inter = 0
    union = 0
    for p, l in zip(pred, labels):
        if p == 1 and l == 1:
            inter += 1
        if p == 1 or l == 1:
            union += 1
    if union == 0:
        # No positive samples and the model also predicts none; set IoU=1.0
        return 1.0
    return inter / union


def safe_mean(arr):
    return sum(arr) / len(arr) if arr else float("nan")


def main():
    print("Loading ground truth...")
    gt_dict = load_ground_truth(GT_PATH)

    print("Loading predictions...")
    pred_items = load_jsonl(PRED_PATH)

    # 1) Without smoothing: different thresholds -> R@0.5 / R@0.7
    recall_no_smooth = {
        thr: {alpha: [] for alpha in IOU_THRESHOLDS} for thr in THRESHOLDS
    }

    # 2) With smoothing: for each w and each threshold -> R@0.5 / R@0.7
    recall_smooth = {
        w: {thr: {alpha: [] for alpha in IOU_THRESHOLDS} for thr in THRESHOLDS}
        for w in WINDOW_SIZES
    }

    missing_gt = 0
    used_samples = 0

    print("Evaluating...")
    for item in tqdm(pred_items):
        vid = str(item["id"])
        raw_probs = item.get("raw_probs", [])
        num_frames = len(raw_probs)
        if num_frames == 0:
            continue

        gt_segments = gt_dict.get(vid)
        if gt_segments is None:
            missing_gt += 1
            continue

        labels = segments_to_frame_labels(gt_segments, num_frames)
        norm_scores = minmax_normalize(raw_probs)

        # ---------- 1) No smoothing ----------
        for thr in THRESHOLDS:
            pred_bin = [1 if s >= thr else 0 for s in norm_scores]
            iou = frame_iou(pred_bin, labels)
            for alpha in IOU_THRESHOLDS:
                hit = 1.0 if iou >= alpha else 0.0
                recall_no_smooth[thr][alpha].append(hit)

        # ---------- 2) With smoothing: w=2..6, threshold sweep 0.2..0.5 ----------
        for w in WINDOW_SIZES:
            sm = smooth_scores(norm_scores, w)
            for thr in THRESHOLDS:
                pred_bin = [1 if s >= thr else 0 for s in sm]
                iou = frame_iou(pred_bin, labels)
                for alpha in IOU_THRESHOLDS:
                    hit = 1.0 if iou >= alpha else 0.0
                    recall_smooth[w][thr][alpha].append(hit)

        used_samples += 1

    print("\n===== Summary =====")
    print(f"Total prediction items: {len(pred_items)}")
    print(f"Used items with GT    : {used_samples}")
    print(f"Missing GT items      : {missing_gt}")

    # Output results without smoothing
    print("\nCharades-STA R@α (no smoothing, varying threshold on min-max scores):")
    for thr in THRESHOLDS:
        line = [f"thr={thr:.2f}"]
        for alpha in IOU_THRESHOLDS:
            r = safe_mean(recall_no_smooth[thr][alpha])
            line.append(f"R@{alpha:.1f}={r:.4f}")
        print("  " + ", ".join(line))

    # Output results with smoothing
    print("\nCharades-STA R@α (smoothed scores, varying window size w and threshold):")
    for w in WINDOW_SIZES:
        print(f"\n  w = {w}:")
        for thr in THRESHOLDS:
            line = [f"    thr={thr:.2f}"]
            for alpha in IOU_THRESHOLDS:
                r = safe_mean(recall_smooth[w][thr][alpha])
                line.append(f"R@{alpha:.1f}={r:.4f}")
            print("  " + ", ".join(line))


if __name__ == "__main__":
    main()