import asyncio
import json
import os
import re
import time
from typing import Dict, List, Tuple

from openai import AsyncAzureOpenAI
from tqdm import tqdm


# ===================== 1. Configuration =====================

PRED_FILE = "eval/narration/youcook2/results/yc2_text_q_85.jsonl"
GT_FILE = "xxx/YouCook2/data/youcook2_ourtest.json"

OUTPUT_FILE = "eval/narration/youcook2/test_q_eval_85.jsonl"

azure_base_url = ""
azure_api_version = ""
azure_ak = ""
azure_model_name = ""

MAX_CONCURRENT_REQUESTS = 50
MAX_RETRIES = 5
RETRY_DELAY = 8
TIMEOUT = 10  # Maximum waiting time (seconds) for each request


# ===================== 2. Text Preprocessing & Structure Construction =====================

def clean_text_for_concat(text: str) -> str:
    """Simple text cleaning: remove <|im_end|> and compress whitespace."""
    if text.startswith(".<|im_end|>"):
        text = text.replace(".<|im_end|>", "")
        print(text)
    text = text.replace("<|im_end|>", " ")
    if text != "":
        text = " ".join(text.strip().split())
    return text


def load_gt(gt_path: str) -> Dict[str, Dict]:
    """
    Load youcook2_ourtest.json

    Returns:
      id2gt[vid] = {
         "segments": [ { "segment": [start, end], "text": ... }, ... ],
         "query_text": "<video> ...",
      }
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2gt = {}
    for item in data:
        vid = item["id"]
        segments = item.get("answer", [])
        # Sort by start time to ensure consistent ordering
        segments = sorted(segments, key=lambda s: s["segment"][0])

        query_list = item.get("query", [])
        if query_list:
            query_text = query_list[0].get("text", "")
        else:
            query_text = ""

        id2gt[vid] = {
            "segments": segments,
            "query_text": query_text,
        }
    return id2gt


def load_preds(pred_path: str) -> Dict[str, Dict]:
    """
    Load prediction results from jsonl.
    Each line is a sample. If the same id appears multiple times, keep the last one.
    """
    preds = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vid = obj["id"]
            preds[vid] = obj
    return preds


def build_overall_gt_text(gt_info: Dict) -> str:
    """
    Concatenate all GT segment texts for one sample into a single overall description.
    """
    segments = gt_info.get("segments", []) or []
    segments = sorted(segments, key=lambda s: s["segment"][0])
    texts = []
    for seg in segments:
        t = (seg.get("text") or "").strip()
        if t:
            texts.append(t)
    return " ".join(texts).strip()


def build_overall_pred_text(pred_item: Dict) -> str:
    """
    Concatenate all generated_outputs texts for one sample into a single description
    in chronological order. No deduplication is applied.
    """
    outs = pred_item.get("generated_outputs", []) or []   ## prediction
    outs = sorted(outs, key=lambda o: o.get("time", 0.0))

    texts = []
    for o in outs:
        raw = o.get("text", "") or ""
        cleaned = clean_text_for_concat(raw)
        if cleaned:
            texts.append(cleaned)

    return " ".join(texts).strip()


def build_items_for_gpt(
    id2gt: Dict[str, Dict],
    preds: Dict[str, Dict],
) -> List[dict]:
    """
    Construct the sample list for GPT evaluation (one sample per video):

      item = {
        "id": vid,
        "video_id": vid,
        "question": query_text,
        "gt_text": overall_gt_text,
        "pred_text": overall_pred_text,
        "gt_empty": bool,
        "pred_empty": bool,
      }
    """
    items = []

    total_videos = 0
    skipped_no_pred_id = 0
    skipped_empty_gt = 0
    empty_pred_videos = 0

    for vid, gt_info in id2gt.items():
        total_videos += 1
        query_text = gt_info.get("query_text", "")

        if vid not in preds:
            skipped_no_pred_id += 1
            continue

        gt_text = build_overall_gt_text(gt_info)
        if not gt_text:
            skipped_empty_gt += 1
            continue

        pred_item = preds[vid]
        pred_text = build_overall_pred_text(pred_item)

        gt_empty = (gt_text == "")
        pred_empty = (pred_text == "")
        if pred_empty:
            empty_pred_videos += 1

        item = {
            "id": str(vid),
            "video_id": vid,
            "question": query_text,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "gt_empty": gt_empty,
            "pred_empty": pred_empty,
        }
        items.append(item)

    print(f"Total videos (in GT file): {total_videos}")
    print(f"  - Videos with missing prediction id: {skipped_no_pred_id}")
    print(f"  - Videos skipped due to empty GT text: {skipped_empty_gt}")
    print(f"  - Videos actually used for evaluation: {len(items)}")
    print(f"  - Videos with empty prediction text: {empty_pred_videos}")

    return items


# ===================== 3. GPT Scoring =====================

EVALUATION_SYSTEM_PROMPT = """
You are an expert evaluator for video narration quality. Your task is to compare
a reference description of a video (ground truth) with a model-generated description
for the same video, and output THREE scores between 0 and 1.

You must consider the model response as a SINGLE long story (it may contain multiple
sentences describing different moments in the video).

IMPORTANT: Higher scores are always better.

Definitions:

1. coherence (story coherence):
   - How internally coherent and well-structured is the model-generated story by itself?
   - Does it read like a reasonable, temporally plausible sequence of actions and states?
   - Penalize contradictions, abrupt jumps, and incoherent, rambling structure.
   - 1.0 = very coherent and well-structured; 0.0 = completely incoherent.

2. alignment (semantic alignment with ground truth):
   - How well does the model-generated story capture the key actions and steps in the ground truth?
   - Consider whether important actions/events are present, correctly described, and roughly in a reasonable order.
   - Hallucinated major steps that clearly do not appear in the ground truth should reduce this score.
   - 1.0 = almost all key content in GT is covered with correct semantics; 0.0 = almost completely unrelated.

3. conciseness (relevant non-redundancy / brevity):
   - This score measures whether the model response is concise GIVEN IT IS RELEVANT to the ground truth.
   - If the model response is largely unrelated to the ground truth (low semantic overlap, wrong topic, ignores the video), conciseness MUST be near 0, even if the response is short.
   - Penalize heavy repetition of similar sentences, long irrelevant digressions, and obvious padding.
   - However, do NOT penalize necessary detail that genuinely helps describe the steps.
   - 1.0 = succinct, minimal redundancy while preserving essential details;
     0.0 = extremely repetitive / rambling / full of irrelevant filler / irrelevant with the groundtruth.
     

Empty or meaningless model responses (or responses that ignore the task) should receive low scores,
typically near 0 for all dimensions.

Output format (VERY IMPORTANT):
- You MUST output valid JSON with exactly the following keys:
  {"coherence": <float>, "alignment": <float>, "conciseness": <float>}
- Each value must be a number between 0 and 1 (inclusive).
- Do NOT output any extra text or explanation.
"""


def construct_evaluation_prompt(question: str, ground_truth: str, prediction: str) -> str:
    q = question.replace("<video>", "").strip() if question else "Describe what happens in this cooking video."
    return f"""
We are evaluating a model that narrates an entire instructional cooking video.

Video query / title (for context):
{q}

Ground truth description of the whole video (concatenation of all key steps):
{ground_truth}

Model-generated narration for the same video (concatenation of all generated sentences over time):
{prediction}

Please read both carefully and then score:
- coherence: how coherent and well-structured the model story is by itself.
- alignment: how well the model story matches the ground truth in terms of key actions and steps.
- conciseness: whether the model story is reasonably concise (low redundancy) given the ground truth.

Remember to output ONLY a JSON object:
{{"coherence": x, "alignment": y, "conciseness": z}}
with each x, y, z in [0, 1].
"""


def parse_scores_from_model_output(text: str) -> Tuple[float, float, float]:
    """
    Parse coherence / alignment / conciseness scores in [0,1] from GPT output.
    Expected format:
      {"coherence": 0.8, "alignment": 0.7, "conciseness": 0.5}
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        c = float(obj["coherence"])
        a = float(obj["alignment"])
        con = float(obj["conciseness"])
        for v in (c, a, con):
            if not (0.0 <= v <= 1.0):
                raise ValueError("Score out of [0,1] range")
        return c, a, con
    except Exception as e:
        raise ValueError(f"Cannot parse scores from model output: {text!r}; error: {e}")


async def process_item(item: dict, semaphore: asyncio.Semaphore, client: AsyncAzureOpenAI, pbar: tqdm):
    """
    Process a single video-level sample:
      - If prediction is empty: assign 0 to all three scores
      - Otherwise call GPT to obtain three scores in [0,1]
    """
    async with semaphore:
        try:
            if item.get("pred_empty", False):
                item["gpt_coherence"] = 0.0
                item["gpt_alignment"] = 0.0
                item["gpt_conciseness"] = 0.0
                pbar.update(1)
                return item

            gt_text = (item.get("gt_text") or "").strip()
            pred_text = (item.get("pred_text") or "").strip()
            question = (item.get("question") or "").strip()

            if not gt_text or not pred_text:
                item["gpt_coherence"] = 0.0
                item["gpt_alignment"] = 0.0
                item["gpt_conciseness"] = 0.0
                pbar.update(1)
                return item

            user_prompt = construct_evaluation_prompt(question, gt_text, pred_text)
            messages = [
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            for attempt in range(MAX_RETRIES):
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=azure_model_name,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=64,
                        ),
                        timeout=TIMEOUT
                    )

                    model_output = response.choices[0].message.content.strip()
                    coh, ali, con = parse_scores_from_model_output(model_output)
                    item["gpt_coherence"] = float(coh)
                    item["gpt_alignment"] = float(ali)
                    item["gpt_conciseness"] = float(con)
                    pbar.update(1)
                    return item

                except asyncio.TimeoutError:
                    tqdm.write(f"⏰ Timeout on item {item.get('id', 'N/A')} (attempt {attempt+1}/{MAX_RETRIES})")
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        tqdm.write(f"⚠️ Error on item {item.get('id', 'N/A')} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        tqdm.write(f"🚨 All retries failed for item {item.get('id', 'N/A')}: {e}")
                        item["gpt_coherence"] = -1.0
                        item["gpt_alignment"] = -1.0
                        item["gpt_conciseness"] = -1.0
                        pbar.update(1)
                        return item

            item["gpt_coherence"] = -1.0
            item["gpt_alignment"] = -1.0
            item["gpt_conciseness"] = -1.0
            pbar.update(1)
            return item

        except Exception as e:
            tqdm.write(f"❌ Error processing item {item.get('id', 'N/A')}: {e}")
            item["gpt_coherence"] = -1.0
            item["gpt_alignment"] = -1.0
            item["gpt_conciseness"] = -1.0
            pbar.update(1)
            return item


# ===================== 4. Main Pipeline =====================

async def main():
    # --- Construct video-level samples ---
    id2gt = load_gt(GT_FILE)
    preds = load_preds(PRED_FILE)
    items = build_items_for_gpt(id2gt, preds)

    total_items = len(items)
    print(f"\n🚀 Number of videos to be evaluated by GPT: {total_items}")

    # --- Initialize GPT client ---
    client = AsyncAzureOpenAI(
        azure_endpoint=azure_base_url,
        api_version=azure_api_version,
        api_key=azure_ak,
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    pbar = tqdm(total=total_items, desc="Evaluating (GPT global three-dimension scoring)")

    try:
        tasks = [process_item(item, semaphore, client, pbar) for item in items]
        results = await asyncio.gather(*tasks)
    finally:
        pbar.close()
        await client.close()

    processed_results = [r for r in results if r is not None]

    # --- Write results ---
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"\n✍️ Writing {len(processed_results)} results to {OUTPUT_FILE} ...")
    try:
        if OUTPUT_FILE.endswith(".jsonl"):
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in processed_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(processed_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"🚨 File writing error: {e}")

    # --- Compute overall averages ---
    coh_scores, ali_scores, con_scores = [], [], []
    for r in processed_results:
        c = r.get("gpt_coherence")
        a = r.get("gpt_alignment")
        con = r.get("gpt_conciseness")
        if isinstance(c, (int, float)) and isinstance(a, (int, float)) and isinstance(con, (int, float)):
            if c >= 0 and a >= 0 and con >= 0:
                coh_scores.append(float(c))
                ali_scores.append(float(a))
                con_scores.append(float(con))

    print("\n--- ✨ GPT Overall Evaluation Statistics ✨ ---")
    print(f"Valid video count: {len(coh_scores)}")
    if coh_scores:
        print(f"Average coherence (story coherence): {sum(coh_scores) / len(coh_scores):.4f}")
        print(f"Average alignment (consistency with GT): {sum(ali_scores) / len(ali_scores):.4f}")
        print(f"Average conciseness (non-redundancy / compactness): {sum(con_scores) / len(con_scores):.4f}")
    else:
        print("No valid GPT scores available, cannot compute averages.")


if __name__ == "__main__":
    asyncio.run(main())