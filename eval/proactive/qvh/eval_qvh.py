import json
from tqdm import tqdm

PRED_PATH = "eval/proactive/qvh/results/qvh.jsonl"
GT_PATH = "QVH/qvh_val_proactive_tts_merged.jsonl"


THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

WINDOW_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_ground_truth(path):
    """
    Return: {id_str: segments}
    segments: [[start, end], ...], here they are directly interpreted as frame indices.
    """
    gt = {}
    for item in load_jsonl(path):
        vid = str(item["id"])
        segments = []
        # According to the provided format: "answer": [{"segment": [[0, 2], [52, 74], ...]}]
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
        # All scores are identical; cannot distinguish them, set all to 0
        return [0.0 for _ in nums]
    return [(x - mn) / (mx - mn) for x in nums]


def smooth_scores(scores, w):
    """
    For each frame i, take the mean of all frames within [i-w, i+w]
    as the smoothed score.
    Boundary positions are automatically truncated.
    """
    n = len(scores)
    out = [0.0] * n
    for i in range(n):
        l = max(0, i - w)
        r = min(n - 1, i + w)
        window = scores[l: r + 1]
        out[i] = sum(window) / len(window)
    return out


def segments_to_frame_labels(segments, num_frames):
    """
    Generate per-frame 0/1 labels based on segment annotations.
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
    Frame-level IoU: treat all frames with value 1 as a set,
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
        # Case where there are no positive samples and the model predicts none
        return 1.0
    return inter / union


def average_precision(scores, labels):
    """
    Frame-level AP within a single video:
    - Sort frames by score
    - Compute AP for that video
    Return None if the video contains no positive samples (skip it).
    """
    assert len(scores) == len(labels)
    n_pos = sum(labels)
    if n_pos == 0:
        return None

    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0], reverse=True)

    hit = 0
    sum_prec = 0.0
    for idx, (_, l) in enumerate(pairs, start=1):
        if l == 1:
            hit += 1
            sum_prec += hit / idx
    return sum_prec / n_pos


def hit_at_1(scores, labels):
    """
    HIT@1: check whether the frame with the highest score
    in the video is a positive frame.
    Return None if the video contains no positive samples.
    """
    assert len(scores) == len(labels)
    if not scores:
        return None
    if sum(labels) == 0:
        return None
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return 1.0 if labels[best_idx] == 1 else 0.0


def safe_mean(arr):
    return sum(arr) / len(arr) if arr else float("nan")


def main():
    print("Loading ground truth...")
    gt_dict = load_ground_truth(GT_PATH)

    print("Loading predictions...")
    pred_items = load_jsonl(PRED_PATH)

    # IoU statistics: raw thresholds + smoothed threshold 0.5
    iou_raw = {thr: [] for thr in THRESHOLDS}
    iou_smooth = {w: [] for w in WINDOW_SIZES}

    # mAP / HIT@1 statistics by score variant
    # One variant is min-max normalized raw scores (raw_norm),
    # others are smoothed scores smooth_w{w}.
    ap_by_variant = {"raw_norm": []}
    hit_by_variant = {"raw_norm": []}
    for w in WINDOW_SIZES:
        name = f"smooth_w{w}"
        ap_by_variant[name] = []
        hit_by_variant[name] = []

    missing_gt = 0
    used_samples = 0

    print("Evaluating...")
    for item in tqdm(pred_items):
        vid = str(item["id"])
        raw_probs = item["raw_probs"]
        num_frames = len(raw_probs)

        gt_segments = gt_dict.get(vid)
        if gt_segments is None:
            # Sample exists in predictions but not in GT, skip it
            missing_gt += 1
            continue

        labels = segments_to_frame_labels(gt_segments, num_frames)
        norm_scores = minmax_normalize(raw_probs)

        # 1) IoU: raw min-max scores with multiple thresholds
        for thr in THRESHOLDS:
            pred_bin = [1 if s >= thr else 0 for s in norm_scores]
            iou = frame_iou(pred_bin, labels)
            iou_raw[thr].append(iou)

        # 2) IoU after smoothing, threshold fixed at 0.5
        smooth_cache = {}
        for w in WINDOW_SIZES:
            sm = smooth_scores(norm_scores, w)
            smooth_cache[w] = sm
            pred_bin = [1 if s >= 0.5 else 0 for s in sm]
            iou = frame_iou(pred_bin, labels)
            iou_smooth[w].append(iou)

        # 3) mAP / HIT@1: evaluated directly on scores (without thresholding)
        ap = average_precision(norm_scores, labels)
        hit1 = hit_at_1(norm_scores, labels)
        if ap is not None:
            ap_by_variant["raw_norm"].append(ap)
        if hit1 is not None:
            hit_by_variant["raw_norm"].append(hit1)

        for w in WINDOW_SIZES:
            sm = smooth_cache[w]
            name = f"smooth_w{w}"
            ap = average_precision(sm, labels)
            hit1 = hit_at_1(sm, labels)
            if ap is not None:
                ap_by_variant[name].append(ap)
            if hit1 is not None:
                hit_by_variant[name].append(hit1)

        used_samples += 1

    print("\n===== Summary =====")
    print(f"Total prediction items: {len(pred_items)}")
    print(f"Used items with GT    : {used_samples}")
    print(f"Missing GT items      : {missing_gt}")

    print("\nFrame-level IoU (min-max raw, different thresholds):")
    for thr in THRESHOLDS:
        print(f"  thr={thr:.2f}: mean IoU = {safe_mean(iou_raw[thr]):.4f}")

    print("\nFrame-level IoU (smoothed, threshold=0.5):")
    for w in WINDOW_SIZES:
        print(f"  w={w}: mean IoU = {safe_mean(iou_smooth[w]):.4f}")

    print("\nFrame-level mAP by score variant:")
    for name, vals in ap_by_variant.items():
        print(f"  {name}: mAP = {safe_mean(vals):.4f}")

    print("\nFrame-level HIT@1 by score variant:")
    for name, vals in hit_by_variant.items():
        print(f"  {name}: HIT@1 = {safe_mean(vals):.4f}")


if __name__ == "__main__":
    main()


# mAP: rank all clips using saliency scores and compute AP over positive clips ("highlight").
# HIT@1: check whether the highest-scoring clip in each video belongs to the positive clips.