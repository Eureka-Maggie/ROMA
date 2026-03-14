import json

PRED_PATH = "eval/narration/youcook2/results/yc2.jsonl"
GT_PATH   = "xxxx/YouCook2/data/youcook2_ourtest.json"

def load_predictions_jsonl(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            vid = obj["id"]
            # predictions come from generated_outputs
            # format: {"time": xxx, "text": "..."}
            pred_points = []
            for g in obj.get("generated_outputs", []):
                t = g.get("time", None)
                if t is not None:
                    pred_points.append(float(t))

            preds[vid] = sorted(pred_points)
    return preds


def load_groundtruth(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
        for item in arr:
            vid = item["id"]
            segs = item.get("answer", [])
            # list of {"segment":[s,e], "text":...}
            # ensure float
            segments = []
            for s in segs:
                st, ed = s["segment"]
                segments.append((float(st), float(ed)))
            gt[vid] = segments
    return gt


def compute_f1(preds_dict, gts_dict):
    total_TP = 0
    total_FP = 0
    total_FN = 0

    for vid, gt_segments in gts_dict.items():
        pred_times = preds_dict.get(vid, [])
        used_pred_indices = set()

        TP = 0
        FN = 0

        # 1) For each GT segment → check if hit
        for (start, end) in gt_segments:
            hits = [i for i, t in enumerate(pred_times) if start <= t < end]
            if len(hits) == 0:
                FN += 1
            else:
                # take the earliest hit as the TP
                first_hit = hits[0]
                used_pred_indices.add(first_hit)
                TP += 1

        # 2) FP = all preds that are NOT used as TP matches
        FP = len(pred_times) - len(used_pred_indices)

        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Compute metrics
    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1, total_TP, total_FP, total_FN


def main():
    preds = load_predictions_jsonl(PRED_PATH)
    gts = load_groundtruth(GT_PATH)

    precision, recall, f1, TP, FP, FN = compute_f1(preds, gts)

    print("======= OVO-Bench SSR — Time-Only Per-Segment F1 =======")
    print(f"Total TP = {TP}")
    print(f"Total FP = {FP}")
    print(f"Total FN = {FN}")
    print("---------------------------------------------------------")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1        = {f1:.4f}")
    print("=========================================================")


if __name__ == "__main__":
    main()
