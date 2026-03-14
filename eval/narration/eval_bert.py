#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional


IM_END_PATTERN = re.compile(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>", re.IGNORECASE)
WS_PATTERN = re.compile(r"\s+")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"JSONL parse error at line {line_no}: {e}")
    return data


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    """Light cleaning: remove special tokens, trim, and merge whitespace. You can optionally add lower()."""
    if s is None:
        return ""
    s = IM_END_PATTERN.sub(" ", s)
    s = s.replace("\n", " ")
    s = WS_PATTERN.sub(" ", s).strip()
    return s


def join_generated(gen_outputs: List[Dict[str, Any]]) -> str:
    """Sort by time and then concatenate."""
    if not gen_outputs:
        return ""
    gen_sorted = sorted(gen_outputs, key=lambda x: float(x.get("time", 0)))
    parts = [normalize_text(x.get("text", "")) for x in gen_sorted]
    parts = [p for p in parts if p]  # remove empty
    return " ".join(parts).strip()


def join_gt_all_segments(gt_answer: List[Dict[str, Any]]) -> str:
    parts = [normalize_text(x.get("text", "")) for x in (gt_answer or [])]
    parts = [p for p in parts if p]
    return " ".join(parts).strip()


def build_pred_map(pred_jsonl: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    The prediction JSONL should contain id and generated_outputs.
    If your JSONL uses a different id field name, modify it here.
    """
    mp = {}
    for obj in pred_jsonl:
        _id = obj.get("id")
        if _id is None:
            # Some frameworks may use 'sample_id' / 'uid', etc.
            _id = obj.get("sample_id") or obj.get("uid")
        if _id is None:
            raise KeyError(f"Cannot find id field in pred item keys={list(obj.keys())[:20]}")
        mp[str(_id)] = obj.get("generated_outputs", []) or []
    return mp


def build_gt_map(gt_json: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mp = {}
    for obj in gt_json:
        _id = obj.get("id")
        if _id is None:
            raise KeyError(f"Cannot find id in gt item keys={list(obj.keys())[:20]}")
        mp[str(_id)] = obj.get("answer", []) or []
    return mp


# -----------------------
# BLEU
# -----------------------
def compute_bleu(cands: List[str], refs: List[str]) -> float:
    """
    corpus BLEU. Prefer sacrebleu, fallback to nltk.
    refs: single reference per candidate
    """
    assert len(cands) == len(refs)
    try:
        import sacrebleu  # type: ignore

        # sacrebleu expects list of references as list-of-lists: [refs]
        bleu = sacrebleu.corpus_bleu(cands, [refs])
        return float(bleu.score)
    except Exception:
        pass

    try:
        import nltk  # type: ignore
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore

        smoothie = SmoothingFunction().method1
        ref_tokens = [[r.split()] for r in refs]
        cand_tokens = [c.split() for c in cands]
        score = corpus_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie) * 100.0
        return float(score)
    except Exception as e:
        raise RuntimeError(
            "BLEU requires sacrebleu or nltk. "
            "Try: pip install sacrebleu nltk"
        ) from e


# -----------------------
# CIDEr (fallback: self-implemented)
# -----------------------
def _ngrams(tokens: List[str], n: int) -> Counter:
    c = Counter()
    if n <= 0:
        return c
    for i in range(0, len(tokens) - n + 1):
        c[tuple(tokens[i:i+n])] += 1
    return c


def _tfidf_vector(ngram_counts: Counter, idf: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, ...], float]:
    # tf: raw count; idf: precomputed
    v = {}
    for g, tf in ngram_counts.items():
        w = idf.get(g, 0.0)
        if w > 0:
            v[g] = float(tf) * w
    return v


def _cosine(v1: Dict[Tuple[str, ...], float], v2: Dict[Tuple[str, ...], float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = 0.0
    for k, a in v1.items():
        b = v2.get(k)
        if b is not None:
            dot += a * b
    n1 = math.sqrt(sum(a*a for a in v1.values()))
    n2 = math.sqrt(sum(b*b for b in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def compute_cider(cands: List[str], refs: List[str]) -> float:
    """
    Prefer evaluate or pycocoevalcap; otherwise use a custom CIDEr implementation
    (approximation of original CIDEr TF-IDF cosine, 1–4 gram average, *10).
    Note: this is a reasonable approximation and does not include the CIDEr-D length penalty.
    """
    assert len(cands) == len(refs)

    # Try evaluate
    try:
        import evaluate  # type: ignore
        cider = evaluate.load("cider")
        # evaluate expects references as List[List[str]]
        out = cider.compute(predictions=cands, references=[[r] for r in refs])
        return float(out["cider"])
    except Exception:
        pass

    # Try pycocoevalcap
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore

        scorer = Cider()
        # scorer expects dict: {id: [sent]} and refs dict: {id: [ref1, ref2...]}
        gts = {i: [refs[i]] for i in range(len(refs))}
        res = {i: [cands[i]] for i in range(len(cands))}
        score, _ = scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        pass

    # Fallback: custom CIDEr
    N = len(refs)
    # Precompute document frequency for each n-gram in references
    df = [defaultdict(int) for _ in range(4)]  # for n=1..4
    ref_ng = []
    for r in refs:
        toks = r.lower().split()
        per_n = []
        for n in range(1, 5):
            c = _ngrams(toks, n)
            per_n.append(c)
            # df uses presence in document
            for g in c.keys():
                df[n-1][g] += 1
        ref_ng.append(per_n)

    # IDF: log((N+1)/(df+1)) + 1
    idf = []
    for n in range(1, 5):
        idf_n = {}
        for g, d in df[n-1].items():
            idf_n[g] = math.log((N + 1.0) / (d + 1.0)) + 1.0
        idf.append(idf_n)

    scores = []
    for i, c in enumerate(cands):
        ctoks = c.lower().split()
        rtoks = refs[i].lower().split()
        s = 0.0
        for n in range(1, 5):
            c_counts = _ngrams(ctoks, n)
            r_counts = _ngrams(rtoks, n)
            v_c = _tfidf_vector(c_counts, idf[n-1])
            v_r = _tfidf_vector(r_counts, idf[n-1])
            s += _cosine(v_c, v_r)
        s = (s / 4.0) * 10.0
        scores.append(s)

    return float(sum(scores) / max(1, len(scores)))


# -----------------------
# BERTScore
# -----------------------
def compute_bertscore_f1(
    cands: List[str],
    refs: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    device: Optional[str] = None,
) -> float:
    """
    Return the average BERTScore F1 value (kept in [0,1], not multiplied by 100).
    Requires bert-score package + an available local transformers model.
    """
    assert len(cands) == len(refs)
    try:
        from bert_score import score  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "BERTScore requires bert-score. Try: pip install bert-score"
        ) from e

    kwargs = {}
    if model_type:
        kwargs["model_type"] = model_type
    else:
        # Default for English is usually roberta-large; if unavailable use distilroberta-base
        kwargs["model_type"] = "roberta-large"

    if device:
        kwargs["device"] = device

    P, R, F1 = score(cands, refs, lang=lang, **kwargs)
    return float(F1.mean().item())


# -----------------------
# Segment alignment evaluation
# -----------------------
def gen_text_for_segment(
    gen_outputs: List[Dict[str, Any]],
    start: float,
    end: float,
) -> str:
    """Select generated sentences with time in [start, end) and concatenate. You may change the last segment to <= end if needed."""
    if not gen_outputs:
        return ""
    picked = []
    for x in gen_outputs:
        t = x.get("time", None)
        if t is None:
            continue
        t = float(t)
        if (t >= start) and (t < end):
            txt = normalize_text(x.get("text", ""))
            if txt:
                picked.append((t, txt))
    picked.sort(key=lambda z: z[0])
    return " ".join([p[1] for p in picked]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_jsonl", type=str, default="eval/narration/youcook2/results/yc2_text_q_85.jsonl")
    parser.add_argument("--gt_json", type=str, default="data/youcook2_ourtest.json") 
    parser.add_argument("--bertscore", dest="bertscore", action="store_true", help="enable BERTScore (default: on)")
    parser.add_argument("--no-bertscore", dest="bertscore", action="store_false", help="disable BERTScore")
    parser.set_defaults(bertscore=True)
    parser.add_argument("--bertscore_model", type=str, default=None, help="HF model name or local path, e.g., roberta-large or /path/to/model")
    parser.add_argument("--bertscore_lang", type=str, default="en")
    parser.add_argument("--bertscore_device", type=str, default=None, help="e.g., cuda:0 or cpu")
    parser.add_argument("--skip_empty_seg", action="store_true", help="segment eval: skip segments where prediction text is empty")
    args = parser.parse_args()

    pred_data = read_jsonl(args.pred_jsonl)
    gt_data = read_json(args.gt_json)

    pred_map = build_pred_map(pred_data)
    gt_map = build_gt_map(gt_data)

    common_ids = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    missing_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))
    missing_gt = sorted(set(pred_map.keys()) - set(gt_map.keys()))

    print(f"[Info] pred items: {len(pred_map)}, gt items: {len(gt_map)}")
    print(f"[Info] matched ids: {len(common_ids)}")
    if missing_pred:
        print(f"[Warn] missing predictions for {len(missing_pred)} ids (show first 5): {missing_pred[:5]}")
    if missing_gt:
        print(f"[Warn] missing groundtruth for {len(missing_gt)} ids (show first 5): {missing_gt[:5]}")

    # ---------- (1) full concat eval ----------
    full_cands, full_refs = [], []
    for _id in common_ids:
        cand = join_generated(pred_map[_id])
        ref = join_gt_all_segments(gt_map[_id])
        full_cands.append(cand)
        full_refs.append(ref)

    bleu_full = compute_bleu(full_cands, full_refs)
    cider_full = compute_cider(full_cands, full_refs)
    print("\n====== (1) Full concat (per-id) ======")
    print(f"BLEU  : {bleu_full:.4f}")
    print(f"CIDEr : {cider_full:.4f}")

    if args.bertscore:
        try:
            bs_full = compute_bertscore_f1(
                full_cands, full_refs,
                lang=args.bertscore_lang,
                model_type=args.bertscore_model,
                device=args.bertscore_device,
            )
            print(f"BERTScore(F1): {bs_full:.6f}")
        except Exception as e:
            print(f"[BERTScore Error] {e}")
            print("Tip: If you are in an offline environment, ensure the model is cached locally, or use --bertscore_model to point to a local directory; you may also set TRANSFORMERS_OFFLINE=1.")

    # ---------- (2) aligned-by-gt-segment eval ----------
    seg_cands, seg_refs = [], []
    for _id in common_ids:
        gen_outputs = pred_map[_id]
        gt_segments = gt_map[_id]  # list of {"segment":[s,e], "text":...}
        for seg in gt_segments:
            seg_range = seg.get("segment", None)
            if not seg_range or len(seg_range) != 2:
                continue
            s, e = float(seg_range[0]), float(seg_range[1])
            ref = normalize_text(seg.get("text", ""))
            cand = gen_text_for_segment(gen_outputs, s, e)

            if args.skip_empty_seg and (not cand):
                continue

            seg_cands.append(cand)
            seg_refs.append(ref)

    bleu_seg = compute_bleu(seg_cands, seg_refs) if seg_cands else float("nan")
    cider_seg = compute_cider(seg_cands, seg_refs) if seg_cands else float("nan")
    print("\n====== (2) Segment-aligned (GT segment-level) ======")
    print(f"[Info] evaluated segments: {len(seg_cands)}")
    print(f"BLEU  : {bleu_seg:.4f}")
    print(f"CIDEr : {cider_seg:.4f}")

    if args.bertscore:
        try:
            bs_seg = compute_bertscore_f1(
                seg_cands, seg_refs,
                lang=args.bertscore_lang,
                model_type=args.bertscore_model,
                device=args.bertscore_device,
            )
            print(f"BERTScore(F1): {bs_seg:.6f}")
        except Exception as e:
            print(f"[BERTScore Error] {e}")
            print("Tip: Same advice for offline environments; alternatively try a smaller model: --bertscore_model distilroberta-base")


if __name__ == "__main__":
    main()