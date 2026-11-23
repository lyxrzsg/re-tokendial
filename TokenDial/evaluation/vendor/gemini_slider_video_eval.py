#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini VLM evaluation for ConceptSlider-style video sliders.

Dataset layout (example):
  {root}/{concept}/{attribute}/{method}/{scale}/{video}.mp4

Where:
  - scale: "1" is the base (before edit), "2".."5" are edited with increasing intensity
  - method: e.g. ours, conceptslider, freeslider, sliderspace, textslider

Evaluation:
  - Sample K aligned timestamps (frame indices) across the 5 videos (use min frame_count across scales)
  - For each timestamp, send 5 frames (Scale 1..5) to Gemini in one request
  - Gemini returns 4 metric scores in [0, 5] with per-metric justifications
  - Output:
      (1) CSV: one row per (concept, attribute, method, filename) with averages + per-frame details
      (2) JSONL: one row per frame evaluation containing raw response + parsed JSON for debugging

Notes:
  - Set GEMINI_API_KEY in env.
  - Requires: google-genai, opencv-python, pillow, numpy
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
import time
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from google import genai

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


SCALES: List[str] = ["1", "2", "3", "4", "5"]


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return n
    finally:
        cap.release()


def read_frame_at(video_path: str, frame_idx: int) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame={frame_idx} from {video_path}")
        return _bgr_to_pil(frame)
    finally:
        cap.release()


def sample_indices(n_frames: int, k: int) -> List[int]:
    """
    Returns up to k indices in [0, n_frames-1], roughly evenly spaced.
    Deduplicated, preserves order.
    """
    if n_frames <= 0:
        return [0]
    if k <= 1:
        return [max(min(n_frames // 2, n_frames - 1), 0)]
    xs = np.linspace(0, max(n_frames - 1, 0), k).round().astype(int).tolist()
    out: List[int] = []
    seen = set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_prompt(concept: str, attribute: str) -> str:
    """
    Minimal wording you can refine later.
    Assumes attribute is interpreted as "make the concept more <attribute>".
    """
    return f"""
You are a meticulous video editing quality evaluator.

Your task is to assess a VIDEO EDIT SLIDER by analyzing 5 images sampled from the SAME timestamp,
corresponding to 5 increasing slider scales.

Scale definition:
- Scale 1: Original video frame (before editing)
- Scale 2-5: Edited frames with progressively stronger edit strength

Concept: "{concept}"
Target attribute direction: "{attribute}"
Goal: Make the concept become more "{attribute}" progressively from Scale 2 to Scale 5.

Instructions:
Analyze the 5-scale sequence as a whole and evaluate how well the editing slider satisfies the editing goal while preserving visual consistency.
- All 5 images depict the SAME moment in time.
- The main subject identity and background should remain the SAME across all scales despite the edit.
- Stronger edits NEVER justify replacing the subject or changing the background.

Score the ENTIRE 5-scale sequence on 4 criteria, each integer in [0,5] where higher is better.
Provide a short justification for each score.

You will evaluate the slider across FOUR criteria.
For each criterion, provide a score from 0 (worst) to 5 (best) and a brief justification.


1. Prompt Following (Score: 0-5)
Question:
Does the slider edit direction correctly and consistently make the SAME "{concept}"
appear more "{attribute}" from Scale 2 to Scale 5?

Scoring Guide:
- 5: The edit perfectly follows the prompt, with Scale 5 being the strongest and most aligned.
- 4: The edit mostly follows the prompt with minor weaknesses.
- 3: The edit partially follows the prompt but is ambiguous or inconsistent.
- 2: The edit weakly reflects the prompt.
- 1: The edit barely relates to the prompt.
- 0: The prompt is ignored or contradicted.

2. Identity Preserving (Score: 0-5)
Question:
Does the main subject remain the SAME identity and category across all scales?

If the subject is replaced, changes category, or becomes a different person/object at ANY scale,
this is considered a severe violation and will be heavily penalized.

Scoring Guide:
- 5: Identity is perfectly preserved across all scales.
- 4: Very minor visual changes but clearly the same subject.
- 3: Noticeable drift, but identity is still mostly recognizable.
- 2: Major identity inconsistency or partial replacement.
- 1: Severe identity change.
- 0: The subject is completely replaced or unrecognizable.

3. Background Consistency (Score: 0-5)
Question:
Have the regions that should NOT be edited (background, scene layout, camera viewpoint)
remained stable across all scales?

Scoring Guide:
- 5: Background is perfectly preserved and stable.
- 4: Minor, subtle background changes.
- 3: Noticeable but non-catastrophic background drift.
- 2: Significant background changes or redraw.
- 1: Severe background inconsistency.
- 0: Background is completely altered.

4. Progressive Intensity (Score: 0-5)
Question:
Does the edit strength increase monotonically and smoothly from Scale 1 to Scale 5?

Scoring Guide:
- 5: Smooth, monotonic increase with clear ordering from weak to strong.
- 4: Mostly monotonic with minor irregularities.
- 3: Inconsistent progression.
- 2: Weak or unclear progression.
- 1: Reversed or chaotic progression.
- 0: No meaningful progression.

Return STRICT JSON ONLY (no markdown) with this schema:
{{
  "prompt_following": {{"score": <int 0-5>, "reason": <string>}},
  "id_preserving": {{"score": <int 0-5>, "reason": <string>}},
  "background_consistency": {{"score": <int 0-5>, "reason": <string>}},
  "progressive_intensity": {{"score": <int 0-5>, "reason": <string>}}
}}
""".strip()


JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_scores_and_reasons(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the strict JSON block from Gemini response.
    Returns dict[k] = {"score": int, "reason": str}
    """
    m = JSON_RE.search(text or "")
    if not m:
        raise ValueError(f"No JSON found in response: {text!r}")
    obj = json.loads(m.group(0))
    keys = ["prompt_following", "id_preserving", "background_consistency", "progressive_intensity"]
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        v = obj.get(k)
        if not isinstance(v, dict):
            raise ValueError(f"Key {k} is not an object: {obj}")
        score = v.get("score")
        reason = v.get("reason")
        if not isinstance(score, int) or score < 0 or score > 5:
            raise ValueError(f"Key {k}.score must be int in [0,5]: {obj}")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            reason = str(reason)
        out[k] = {"score": score, "reason": reason}
    return out


@dataclasses.dataclass(frozen=True)
class Item:
    concept: str
    attribute: str
    method: str
    filename: str
    scale_to_path: Dict[str, str]


def discover_items(root: str, methods: Optional[List[str]] = None) -> List[Item]:
    """
    Discover items that have all 5 scales present for a given method.
    """
    items: List[Item] = []
    for concept in sorted(os.listdir(root)):
        concept_dir = os.path.join(root, concept)
        if not os.path.isdir(concept_dir):
            continue
        for attribute in sorted(os.listdir(concept_dir)):
            attr_dir = os.path.join(concept_dir, attribute)
            if not os.path.isdir(attr_dir):
                continue
            for method in sorted(os.listdir(attr_dir)):
                if methods and method not in methods:
                    continue
                method_dir = os.path.join(attr_dir, method)
                if not os.path.isdir(method_dir):
                    continue
                base_dir = os.path.join(method_dir, "1")
                if not os.path.isdir(base_dir):
                    continue

                for fn in sorted(os.listdir(base_dir)):
                    if not fn.endswith(".mp4"):
                        continue
                    scale_to_path: Dict[str, str] = {}
                    ok = True
                    for s in SCALES:
                        p = os.path.join(method_dir, s, fn)
                        if not os.path.isfile(p):
                            ok = False
                            break
                        scale_to_path[s] = p
                    if ok:
                        items.append(Item(concept, attribute, method, fn, scale_to_path))
    return items


def _avg(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _item_key(concept: str, attribute: str, method: str, filename: str) -> Tuple[str, str, str, str]:
    return (concept, attribute, method, filename)


def load_completed_keys_from_csv(out_csv: str) -> "set[Tuple[str, str, str, str]]":
    """
    Resume mechanism: treat existing rows as completed items.
    Uses (concept, attribute, method, filename) as the unique key.
    """
    keys: "set[Tuple[str, str, str, str]]" = set()
    if not os.path.isfile(out_csv):
        return keys
    try:
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                c = (row.get("concept") or "").strip()
                a = (row.get("attribute") or "").strip()
                m = (row.get("method") or "").strip()
                fn = (row.get("filename") or "").strip()
                if c and a and m and fn:
                    keys.add(_item_key(c, a, m, fn))
    except Exception:
        # If CSV is malformed/empty, just don't resume.
        return set()
    return keys


def eval_one_item(
    client: Any,
    model: str,
    item: Item,
    num_frames: int,
    sleep_s: float,
    max_retries: int,
    out_jsonl_fh,
    skip_bad_json: bool,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns:
      - avg_scores: dict metric -> float
      - per_frame: dict with details useful for CSV (scores/reasons/frame_indices)
    """
    counts = [get_frame_count(item.scale_to_path[s]) for s in SCALES]
    positive_counts = [c for c in counts if c > 0]
    n_min = min(positive_counts) if positive_counts else 0
    frame_idxs = sample_indices(n_min, num_frames)

    per_frame_scores: Dict[str, List[int]] = {
        k: [] for k in ["prompt_following", "id_preserving", "background_consistency", "progressive_intensity"]
    }
    per_frame_reasons: Dict[str, List[str]] = {
        k: [] for k in ["prompt_following", "id_preserving", "background_consistency", "progressive_intensity"]
    }
    raw_texts: List[str] = []
    ok_frame_indices: List[int] = []
    skipped_frames: List[Dict[str, Any]] = []

    for fi in frame_idxs:
        prompt = build_prompt(item.concept, item.attribute)
        imgs = {s: read_frame_at(item.scale_to_path[s], fi) for s in SCALES}

        contents: List[Any] = [
            prompt,
            "--- Scale 1 (base) ---", imgs["1"],
            "--- Scale 2 (edited) ---", imgs["2"],
            "--- Scale 3 (edited) ---", imgs["3"],
            "--- Scale 4 (edited) ---", imgs["4"],
            "--- Scale 5 (edited) ---", imgs["5"],
        ]

        last_err: Optional[Exception] = None
        parsed: Optional[Dict[str, Dict[str, Any]]] = None
        resp_text: str = ""

        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=contents)
                resp_text = getattr(resp, "text", "") or ""
                parsed = parse_scores_and_reasons(resp_text)
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep_s * (2**attempt))

        if parsed is None:
            # Debug: record the raw response and parsing error; optionally skip.
            err_str = repr(last_err) if last_err is not None else "unknown_error"
            out_jsonl_fh.write(
                json.dumps(
                    {
                        "concept": item.concept,
                        "attribute": item.attribute,
                        "method": item.method,
                        "filename": item.filename,
                        "frame_idx": fi,
                        "error": err_str,
                        "raw_response_text": resp_text,
                        "skipped": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_jsonl_fh.flush()
            skipped_frames.append({"frame_idx": fi, "error": err_str})
            if skip_bad_json:
                continue
            raise RuntimeError(f"Gemini failed for {item.method}/{item.filename} frame={fi}: {last_err}")

        raw_texts.append(resp_text)
        ok_frame_indices.append(fi)
        for k, v in parsed.items():
            per_frame_scores[k].append(int(v["score"]))
            per_frame_reasons[k].append(str(v["reason"]))

        out_jsonl_fh.write(
            json.dumps(
                {
                    "concept": item.concept,
                    "attribute": item.attribute,
                    "method": item.method,
                    "filename": item.filename,
                    "frame_idx": fi,
                    "scores": {k: per_frame_scores[k][-1] for k in per_frame_scores},
                    "reasons": {k: per_frame_reasons[k][-1] for k in per_frame_reasons},
                    "raw_response_text": resp_text,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        out_jsonl_fh.flush()

        time.sleep(sleep_s)

    if not ok_frame_indices:
        raise RuntimeError(
            f"All frames were skipped (non-JSON) for {item.concept}/{item.attribute}/{item.method}/{item.filename}. "
            f"Try increasing retries, adjusting the prompt, or lowering concurrency."
        )

    avg_scores = {k: _avg([float(x) for x in per_frame_scores[k]]) for k in per_frame_scores}
    per_frame = {
        "frame_indices": frame_idxs,
        "ok_frame_indices": ok_frame_indices,
        "skipped_frames": skipped_frames,
        "per_frame_scores": per_frame_scores,
        "per_frame_reasons": per_frame_reasons,
        "raw_texts": raw_texts,
        "min_frame_count": n_min,
        "frame_counts": counts,
    }
    return avg_scores, per_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. /.../siggraph_AMT_sample")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--methods", default="", help="comma-separated; empty=all")
    ap.add_argument("--num_frames", type=int, default=3, help="sampled timestamps per video")
    ap.add_argument("--sleep_s", type=float, default=0.5, help="basic rate limit / spacing")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="if >0, only evaluate first N items (debug)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If --out_csv exists, append to it and skip items already present (by concept+attribute+method+filename).",
    )
    ap.add_argument(
        "--skip_bad_json",
        action="store_true",
        help="If Gemini output cannot be parsed as strict JSON, record it to JSONL and skip that frame instead of failing.",
    )
    ap.add_argument(
        "--skip_item_on_all_bad",
        action="store_true",
        help="If all sampled frames are skipped for an item, skip the item instead of stopping the whole run.",
    )
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in your environment before running.")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or None
    client = genai.Client(api_key=api_key)

    items = discover_items(args.root, methods=methods)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    print(f"Discovered {len(items)} items")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    completed_keys = load_completed_keys_from_csv(args.out_csv) if args.resume else set()
    if args.resume and completed_keys:
        print(f"Resume enabled: {len(completed_keys)} items already in CSV; will skip them.")

    csv_exists = os.path.isfile(args.out_csv) and os.path.getsize(args.out_csv) > 0
    jsonl_exists = os.path.isfile(args.out_jsonl) and os.path.getsize(args.out_jsonl) > 0

    csv_mode = "a" if (args.resume and csv_exists) else "w"
    jsonl_mode = "a" if (args.resume and jsonl_exists) else "w"

    with open(args.out_jsonl, jsonl_mode, encoding="utf-8") as jf:
        with open(args.out_csv, csv_mode, newline="", encoding="utf-8") as cf:
            fieldnames = [
                "concept",
                "attribute",
                "method",
                "filename",
                "num_frames",
                "frame_indices",
                "ok_frame_indices",
                "skipped_frames",
                "prompt_following_avg",
                "id_preserving_avg",
                "background_consistency_avg",
                "progressive_intensity_avg",
                "prompt_following_per_frame",
                "id_preserving_per_frame",
                "background_consistency_per_frame",
                "progressive_intensity_per_frame",
                "prompt_following_reasons",
                "id_preserving_reasons",
                "background_consistency_reasons",
                "progressive_intensity_reasons",
                "min_frame_count",
                "frame_counts",
            ]
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            if csv_mode == "w":
                w.writeheader()

            t_all0 = perf_counter()
            times_s: List[float] = []
            it_iter = tqdm(items, desc="Evaluating", unit="item") if tqdm is not None else items

            for idx, it in enumerate(it_iter, start=1):
                key = _item_key(it.concept, it.attribute, it.method, it.filename)
                if args.resume and key in completed_keys:
                    if tqdm is not None:
                        tqdm.write(f"[skip(resume)] {it.concept}/{it.attribute}/{it.method}/{it.filename}")
                    else:
                        print(f"[skip(resume)] {it.concept}/{it.attribute}/{it.method}/{it.filename}")
                    continue

                t0 = perf_counter()
                try:
                    avg_scores, per_frame = eval_one_item(
                        client=client,
                        model=args.model,
                        item=it,
                        num_frames=args.num_frames,
                        sleep_s=args.sleep_s,
                        max_retries=args.max_retries,
                        out_jsonl_fh=jf,
                        skip_bad_json=bool(args.skip_bad_json),
                    )
                except Exception as e:
                    if args.skip_item_on_all_bad and args.skip_bad_json:
                        # Record item-level failure and keep going.
                        jf.write(
                            json.dumps(
                                {
                                    "concept": it.concept,
                                    "attribute": it.attribute,
                                    "method": it.method,
                                    "filename": it.filename,
                                    "error": repr(e),
                                    "item_skipped": True,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        jf.flush()
                        print("Skipped item:", it.concept, it.attribute, it.method, it.filename, "error:", repr(e))
                        continue
                    raise

                row = {
                    "concept": it.concept,
                    "attribute": it.attribute,
                    "method": it.method,
                    "filename": it.filename,
                    "num_frames": args.num_frames,
                    "frame_indices": json.dumps(per_frame["frame_indices"], ensure_ascii=False),
                    "ok_frame_indices": json.dumps(per_frame["ok_frame_indices"], ensure_ascii=False),
                    "skipped_frames": json.dumps(per_frame["skipped_frames"], ensure_ascii=False),
                    "prompt_following_avg": round(avg_scores["prompt_following"], 6),
                    "id_preserving_avg": round(avg_scores["id_preserving"], 6),
                    "background_consistency_avg": round(avg_scores["background_consistency"], 6),
                    "progressive_intensity_avg": round(avg_scores["progressive_intensity"], 6),
                    "prompt_following_per_frame": json.dumps(per_frame["per_frame_scores"]["prompt_following"], ensure_ascii=False),
                    "id_preserving_per_frame": json.dumps(per_frame["per_frame_scores"]["id_preserving"], ensure_ascii=False),
                    "background_consistency_per_frame": json.dumps(per_frame["per_frame_scores"]["background_consistency"], ensure_ascii=False),
                    "progressive_intensity_per_frame": json.dumps(per_frame["per_frame_scores"]["progressive_intensity"], ensure_ascii=False),
                    "prompt_following_reasons": json.dumps(per_frame["per_frame_reasons"]["prompt_following"], ensure_ascii=False),
                    "id_preserving_reasons": json.dumps(per_frame["per_frame_reasons"]["id_preserving"], ensure_ascii=False),
                    "background_consistency_reasons": json.dumps(per_frame["per_frame_reasons"]["background_consistency"], ensure_ascii=False),
                    "progressive_intensity_reasons": json.dumps(per_frame["per_frame_reasons"]["progressive_intensity"], ensure_ascii=False),
                    "min_frame_count": per_frame["min_frame_count"],
                    "frame_counts": json.dumps(per_frame["frame_counts"], ensure_ascii=False),
                }
                w.writerow(row)
                cf.flush()
                if args.resume:
                    completed_keys.add(key)

                dt = perf_counter() - t0
                times_s.append(dt)
                avg_dt = float(sum(times_s) / len(times_s))
                remaining = max(len(items) - idx, 0)
                eta_s = remaining * avg_dt
                total_s = perf_counter() - t_all0
                msg = (
                    f"[{idx}/{len(items)}] {it.concept}/{it.attribute}/{it.method}/{it.filename} "
                    f"dt={dt:.2f}s avg={avg_dt:.2f}s ETA={eta_s/60:.1f}m total={total_s/60:.1f}m "
                    f"scores={avg_scores}"
                )
                if tqdm is not None:
                    tqdm.write(msg)
                else:
                    print(msg)


if __name__ == "__main__":
    main()

