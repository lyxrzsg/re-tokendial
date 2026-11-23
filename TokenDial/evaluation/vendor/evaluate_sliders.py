#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate continuous attribute-control sliders on generated videos.

This release version keeps exactly the four metrics used in the final evaluation:

  - strength_anchor
  - monotonicity_anchor
  - one_sided_span_anchor
  - lpips_preservation

Expected layout (with --method):
  {root}/{concept}/{attribute}/{method}/{scale}/[images/]sample.mp4

Expected layout (without --method):
  {root}/{concept}/{attribute}/{scale}/[images/]sample.mp4

Scale folders must have numeric names. Samples are discovered from the baseline
(the numerically smallest scale) and matched by filename across all scales.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# ============================================================================
# Dataset naming helpers
# ============================================================================

@dataclass(frozen=True)
class ConceptSpec:
    concept_word: str
    dir_concepts: Optional[List[str]] = None
    only_attribute: Optional[str] = None
    exclude_attributes: Optional[List[str]] = None


def default_concept_specs() -> Dict[str, ConceptSpec]:
    """Known naming conventions from the released benchmark.

    Unknown concepts are supported automatically by replacing underscores with
    spaces, e.g. ``water_splash`` -> ``water splash``.
    """
    return {
        "aurora": ConceptSpec(concept_word="aurora"),
        "bubble": ConceptSpec(concept_word="bubble"),
        "campfire": ConceptSpec(concept_word="campfire"),
        "confetti": ConceptSpec(concept_word="confetti"),
        "explosion": ConceptSpec(concept_word="fire explosion"),
        "ink": ConceptSpec(concept_word="ink"),
        "person": ConceptSpec(concept_word="person", exclude_attributes=["walk_to_run"]),
        "person_walk": ConceptSpec(
            concept_word="person",
            dir_concepts=["person_walk", "person"],
            only_attribute="walk_to_run",
        ),
        "smoke": ConceptSpec(concept_word="smoke"),
        "snowflake": ConceptSpec(concept_word="snowflake"),
        "spark": ConceptSpec(concept_word="spark"),
        "speed": ConceptSpec(concept_word="speed"),
        "water_splash": ConceptSpec(concept_word="water splash"),
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate continuous attribute-control sliders with CLIP and LPIPS."
    )
    p.add_argument(
        "--root",
        required=True,
        help="Root directory containing concept/attribute/scale video folders.",
    )
    p.add_argument(
        "--guidance_dir",
        required=True,
        help=(
            "Directory containing guidance JSON files at "
            "{guidance_dir}/{concept}/{attribute}.json."
        ),
    )
    p.add_argument(
        "--method",
        default="",
        help=(
            "Optional method folder name between attribute and scale. "
            "Leave empty if --root already points to results without a method layer."
        ),
    )
    p.add_argument(
        "--out_csv",
        default="slider_metrics.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--scales",
        default="auto",
        help="Comma-separated numeric scale folder names, or 'auto' to discover them.",
    )
    p.add_argument("--concepts", default="", help="Comma-separated concept filter. Empty = all.")
    p.add_argument("--attributes", default="", help="Comma-separated attribute filter. Empty = all.")
    p.add_argument("--stride", type=int, default=4, help="Read every Nth video frame.")
    p.add_argument("--max_frames", type=int, default=16, help="Cap sampled frames (0 = unlimited).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--clip_batch_size", type=int, default=8)
    p.add_argument("--lpips_batch_size", type=int, default=8)
    p.add_argument("--monotonicity_eps", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true", help="Append to an existing CSV and skip completed samples.")
    p.add_argument("--skip_errors", action="store_true", help="Skip a sample on error instead of stopping.")
    return p.parse_args()


# ============================================================================
# Filesystem helpers
# ============================================================================

def list_concepts(root: str) -> List[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def choose_concept_dir(root: str, dir_concepts: List[str]) -> Optional[Tuple[str, str]]:
    for dc in dir_concepts:
        p = os.path.join(root, dc)
        if os.path.isdir(p):
            return dc, p
    return None


def scale_images_dir(scale_path: str) -> Optional[str]:
    images_dir = os.path.join(scale_path, "images")
    if os.path.isdir(images_dir):
        return images_dir
    if os.path.isdir(scale_path):
        return scale_path
    return None


def list_mp4_samples(images_dir: str) -> List[str]:
    return sorted(
        name
        for name in os.listdir(images_dir)
        if name.lower().endswith(".mp4") and os.path.isfile(os.path.join(images_dir, name))
    )


def parse_scale_value(scale_name: str) -> float:
    try:
        return float(scale_name)
    except Exception:
        return float("inf")


def discover_scales(attr_path: str) -> List[str]:
    scales: List[str] = []
    for d in os.listdir(attr_path):
        full = os.path.join(attr_path, d)
        if not os.path.isdir(full):
            continue
        try:
            float(d)
        except Exception:
            continue
        scales.append(d)
    scales.sort(key=lambda s: (parse_scale_value(s), s))
    return scales


def resolve_scales_for_attr(attr_path: str, scales_arg: str) -> List[str]:
    discovered = discover_scales(attr_path)
    if scales_arg.strip().lower() == "auto":
        return discovered

    requested = [s.strip() for s in scales_arg.split(",") if s.strip()]
    float_to_names: Dict[float, List[str]] = {}
    for name in discovered:
        v = parse_scale_value(name)
        if v == float("inf"):
            continue
        float_to_names.setdefault(v, []).append(name)
    for v in float_to_names:
        float_to_names[v].sort(key=lambda n: (len(n), n))

    chosen: List[str] = []
    for s in requested:
        if s in discovered:
            chosen.append(s)
            continue
        v = parse_scale_value(s)
        if v in float_to_names and float_to_names[v]:
            chosen.append(float_to_names[v][0])
            continue
        chosen.append(s)
    chosen.sort(key=lambda n: (parse_scale_value(n), n))
    return chosen


# ============================================================================
# Prompt helpers
# ============================================================================

def attribute_phrase(attr: str) -> str:
    return attr.replace("_", " ")


def build_suffix_by_rank(concept_word: str, attr: str, rank: int, total: int) -> str:
    """Preserves the exact rank-to-text mapping used in the original evaluation."""
    ap = attribute_phrase(attr)
    if total <= 1 or rank <= 0:
        return ""
    levels = ["slightly", "", "very", "extremely"]
    rank_idx = rank - 1
    denom = max(1, total - 2)
    idx = int((rank_idx * (len(levels) - 1)) // denom)
    idx = max(0, min(idx, len(levels) - 1))
    w = levels[idx]
    if w == "":
        return f"{concept_word} {ap}"
    return f"{concept_word} {w} {ap}"


def build_anchor_prompt(concept_word: str, attr: str, highest_rank: int, total_scales: int) -> str:
    suffix = build_suffix_by_rank(concept_word, attr, highest_rank, total_scales).strip()
    if suffix:
        return suffix
    return f"{concept_word} {attribute_phrase(attr)}"


def load_guidance_pos_neg_prompts(
    guidance_dir: str,
    concept_dir_candidates: List[str],
    attr: str,
) -> Tuple[List[str], List[str], str]:
    last_err: Optional[Exception] = None
    searched: List[str] = []

    for concept_dir in concept_dir_candidates:
        p = os.path.join(guidance_dir, concept_dir, f"{attr}.json")
        searched.append(p)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            pos = [str(s).strip() for s in (data.get("direction_prompts") or []) if str(s).strip()]
            neg = [str(s).strip() for s in (data.get("neg_direction_prompts") or []) if str(s).strip()]
            if not pos or not neg:
                raise ValueError(
                    f"Expected non-empty 'direction_prompts' and 'neg_direction_prompts' in {p}"
                )
            return pos, neg, p
        except Exception as e:
            last_err = e

    if last_err is not None:
        raise last_err
    raise FileNotFoundError(
        f"Cannot find guidance JSON for attribute '{attr}'. Searched: {searched}"
    )


# ============================================================================
# Video / model helpers
# ============================================================================

def read_video_frames(mp4_path: str, stride: int, max_frames: int) -> List:
    import decord
    from PIL import Image

    vr = decord.VideoReader(mp4_path, num_threads=1)
    idxs = list(range(0, len(vr), max(1, stride)))
    if max_frames and max_frames > 0:
        idxs = idxs[:max_frames]
    return [Image.fromarray(vr[i].asnumpy()) for i in idxs]


def resolve_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_mean_score_for_frames(frames, prompt: str, model, processor, device, batch_size: int) -> float:
    """Exact CLIP frame-averaging logic from the original evaluation."""
    import torch

    if not frames:
        return float("nan")
    vals: List[float] = []
    with torch.no_grad():
        for i in range(0, len(frames), max(1, batch_size)):
            chunk = frames[i : i + max(1, batch_size)]
            inputs = processor(text=[prompt] * len(chunk), images=chunk, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            # Each text entry is identical. Take one text column so every frame
            # contributes exactly once, including the final partial batch.
            logits = out.logits_per_image[:, 0].detach().float().cpu().numpy().reshape(-1)
            vals.extend([float(x) for x in logits])
    return float(np.mean(vals)) if vals else float("nan")


def clip_mean_score_for_frames_multi_prompt(
    frames, prompts: List[str], model, processor, device, batch_size: int
) -> float:
    """Exact multi-prompt averaging logic from the original evaluation."""
    if not frames:
        return float("nan")
    vals: List[float] = []
    for pr in prompts:
        pr = (pr or "").strip()
        if not pr:
            continue
        v = clip_mean_score_for_frames(frames, pr, model, processor, device, batch_size)
        if np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def _pil_to_lpips_tensor(img, size: int = 256):
    """Exact LPIPS preprocessing logic from the original evaluation."""
    import numpy as _np
    import torch as _torch
    from PIL import Image

    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got: {type(img)}")
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Empty image")
    if w < h:
        new_w = size
        new_h = int(round(h * (size / w)))
    else:
        new_h = size
        new_w = int(round(w * (size / h)))
    img_r = img.resize((new_w, new_h), resample=Image.BICUBIC)
    left = max(0, (new_w - size) // 2)
    top = max(0, (new_h - size) // 2)
    img_c = img_r.crop((left, top, left + size, top + size))
    arr = _np.asarray(img_c).astype(_np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image array [H,W,3], got shape={arr.shape}")
    t = _torch.from_numpy(arr).permute(2, 0, 1)
    t = t * 2.0 - 1.0
    return t


def lpips_mean_for_frame_pairs(base_frames, edit_frames, lpips_model, device, batch_size: int) -> float:
    """Exact frame-aligned LPIPS averaging logic from the original evaluation."""
    import torch

    n = min(len(base_frames), len(edit_frames))
    if n <= 0:
        return float("nan")
    dists: List[float] = []
    with torch.no_grad():
        for i in range(0, n, max(1, batch_size)):
            j = min(n, i + max(1, batch_size))
            b = torch.stack([_pil_to_lpips_tensor(im) for im in base_frames[i:j]], dim=0).to(device)
            e = torch.stack([_pil_to_lpips_tensor(im) for im in edit_frames[i:j]], dim=0).to(device)
            out = lpips_model(b, e)
            out = out.view(-1).detach().float().cpu().numpy().tolist()
            dists.extend([float(x) for x in out])
    return float(np.mean(dists)) if dists else float("nan")


# ============================================================================
# Metric definitions -- intentionally unchanged from the original evaluation
# ============================================================================

def _ordered_clip_sequence(scale_to_clip: Dict[str, float], ordered_scales: List[str]) -> List[float]:
    return [scale_to_clip[s] for s in ordered_scales if s in scale_to_clip and np.isfinite(scale_to_clip[s])]


def monotonicity_from_scale_scores(
    scale_to_score: Dict[str, float], ordered_scales: List[str], eps: float
) -> float:
    seq = _ordered_clip_sequence(scale_to_score, ordered_scales)
    if len(seq) < 2:
        return float("nan")
    diffs = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
    if not diffs:
        return float("nan")
    return float(np.mean([1.0 if d > eps else 0.0 for d in diffs]))


def one_sided_span_from_scale_scores(
    scale_to_score: Dict[str, float], ordered_scales: List[str]
) -> float:
    seq = _ordered_clip_sequence(scale_to_score, ordered_scales)
    if len(seq) < 2:
        return float("nan")
    diffs = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
    if not diffs:
        return float("nan")
    return float(np.std(np.array(diffs, dtype=np.float32)))


def compute_strength_anchor(
    scale_to_clip_anchor_pos: Dict[str, float],
    scale_to_clip_anchor_neg: Dict[str, float],
    ordered_scales: List[str],
) -> float:
    seq_pos = _ordered_clip_sequence(scale_to_clip_anchor_pos, ordered_scales)
    seq_neg = _ordered_clip_sequence(scale_to_clip_anchor_neg, ordered_scales)
    if len(seq_pos) >= 2 and len(seq_neg) >= 2:
        strength_pos = float(seq_pos[-1] - seq_pos[0])
        strength_neg = float(seq_neg[0] - seq_neg[-1])
        return float(0.5 * (strength_pos + strength_neg))
    return float("nan")


def compute_lpips_preservation(scale_to_lpips: Dict[str, float], baseline_scale: str) -> float:
    vals = [v for s, v in scale_to_lpips.items() if s != baseline_scale and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ============================================================================
# Resume helpers
# ============================================================================

def _row_key(method: str, concept: str, attribute: str, sample: str) -> str:
    return f"{method}|{concept}|{attribute}|{sample}"


def load_seen_keys(out_csv: str) -> Set[str]:
    seen: Set[str] = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) <= 0:
        return seen
    try:
        with open(out_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                return seen
            for row in r:
                method = (row.get("method") or "").strip()
                concept = (row.get("concept") or "").strip()
                attribute = (row.get("attribute") or "").strip()
                sample = (row.get("sample") or "").strip()
                if concept and attribute and sample:
                    seen.add(_row_key(method, concept, attribute, sample))
    except Exception:
        return set()
    return seen


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root)
    guidance_dir = os.path.abspath(args.guidance_dir)
    method = args.method.strip()

    if not os.path.isdir(root):
        raise NotADirectoryError(f"--root does not exist or is not a directory: {root}")
    if not os.path.isdir(guidance_dir):
        raise NotADirectoryError(
            f"--guidance_dir does not exist or is not a directory: {guidance_dir}"
        )

    concept_filter = {c.strip() for c in args.concepts.split(",") if c.strip()}
    attr_filter = {a.strip() for a in args.attributes.split(",") if a.strip()}

    device = resolve_device(args.device)

    import lpips as _lpips
    from transformers import CLIPModel, CLIPProcessor

    lpips_model = _lpips.LPIPS(net="alex").to(device).eval()
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=False
    ).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    specs = default_concept_specs()
    physical = list_concepts(root)
    concepts = sorted(set(physical) | set(specs.keys()))
    if concept_filter:
        concepts = [c for c in concepts if c in concept_filter]

    out_fields = [
        "method",
        "concept",
        "attribute",
        "sample",
        "baseline_scale",
        "last_scale",
        "total_scales",
        "scales_present",
        "anchor_prompt",
        "strength_anchor",
        "monotonicity_anchor",
        "one_sided_span_anchor",
        "lpips_preservation",
    ]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    seen_keys = load_seen_keys(args.out_csv) if args.resume else set()
    append_mode = bool(args.resume and os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0)
    file_mode = "a" if append_mode else "w"
    write_header = not append_mode

    if append_mode:
        with open(args.out_csv, "r", newline="") as rf:
            r = csv.reader(rf)
            existing_header = next(r, None)
        if existing_header and [h.strip() for h in existing_header] != out_fields:
            raise ValueError(
                "Cannot --resume into an existing CSV with a different schema.\n"
                f"Existing header: {existing_header}\n"
                f"Expected header: {out_fields}\n"
                "Use a new --out_csv or remove the old file."
            )

    with open(args.out_csv, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        if write_header:
            writer.writeheader()

        pbar = tqdm(unit="sample") if tqdm is not None else None

        for concept in concepts:
            if concept not in specs:
                specs[concept] = ConceptSpec(concept_word=concept.replace("_", " "))
            spec = specs[concept]

            dir_concepts = spec.dir_concepts or [concept]
            chosen = choose_concept_dir(root, dir_concepts)
            if chosen is None:
                continue
            _, concept_path = chosen

            attributes = [
                d
                for d in sorted(os.listdir(concept_path))
                if os.path.isdir(os.path.join(concept_path, d))
            ]
            if spec.only_attribute is not None:
                attributes = [a for a in attributes if a == spec.only_attribute]
            if spec.exclude_attributes:
                exclude = set(spec.exclude_attributes)
                attributes = [a for a in attributes if a not in exclude]
            if attr_filter:
                attributes = [a for a in attributes if a in attr_filter]

            for attr in attributes:
                attr_path = os.path.join(concept_path, attr)
                result_path = os.path.join(attr_path, method) if method else attr_path
                if not os.path.isdir(result_path):
                    continue

                scales_for_attr = resolve_scales_for_attr(result_path, args.scales)
                total_scales = len(scales_for_attr)
                if total_scales < 2:
                    continue

                baseline_scale = scales_for_attr[0]
                last_scale = scales_for_attr[-1]
                highest_rank = max(0, total_scales - 1)

                baseline_images_dir = scale_images_dir(os.path.join(result_path, baseline_scale))
                if baseline_images_dir is None:
                    continue

                samples = list_mp4_samples(baseline_images_dir)
                if not samples:
                    continue

                # strength_anchor uses positive + negative prompt sets from this JSON.
                concept_dir_candidates = [concept]
                for dc in dir_concepts:
                    if dc not in concept_dir_candidates:
                        concept_dir_candidates.append(dc)
                guidance_pos_prompts, guidance_neg_prompts, _ = load_guidance_pos_neg_prompts(
                    guidance_dir=guidance_dir,
                    concept_dir_candidates=concept_dir_candidates,
                    attr=attr,
                )

                # monotonicity_anchor / one_sided_span_anchor use this single anchor prompt.
                anchor_prompt = build_anchor_prompt(
                    spec.concept_word, attr, highest_rank, total_scales
                )

                for sample in samples:
                    sample_id = os.path.splitext(sample)[0]
                    key = _row_key(method, concept, attr, sample_id)
                    if args.resume and key in seen_keys:
                        continue

                    if pbar is not None:
                        pbar.set_description(f"{concept}/{attr}/{sample_id}")

                    try:
                        baseline_mp4 = os.path.join(baseline_images_dir, sample)
                        base_frames = read_video_frames(
                            baseline_mp4, args.stride, args.max_frames
                        )
                        if not base_frames:
                            raise ValueError(f"No frames decoded from baseline video: {baseline_mp4}")

                        scale_to_clip_anchor: Dict[str, float] = {}
                        scale_to_clip_guidance_pos: Dict[str, float] = {}
                        scale_to_clip_guidance_neg: Dict[str, float] = {}
                        scale_to_lpips: Dict[str, float] = {}
                        scales_present: List[str] = []

                        for scale in scales_for_attr:
                            images_dir = scale_images_dir(os.path.join(result_path, scale))
                            if images_dir is None:
                                raise FileNotFoundError(
                                    f"Missing scale directory for {concept}/{attr}/{sample_id}: {scale}"
                                )
                            mp4 = os.path.join(images_dir, sample)
                            if not os.path.exists(mp4):
                                raise FileNotFoundError(
                                    f"Missing video for {concept}/{attr}/{sample_id} at scale {scale}: {mp4}"
                                )

                            frames = read_video_frames(mp4, args.stride, args.max_frames)
                            if not frames:
                                raise ValueError(
                                    f"No frames decoded for {concept}/{attr}/{sample_id} at scale {scale}: {mp4}"
                                )

                            scales_present.append(scale)

                            # Single anchor-prompt CLIP score used only for
                            # monotonicity_anchor and one_sided_span_anchor.
                            scale_to_clip_anchor[scale] = clip_mean_score_for_frames(
                                frames=frames,
                                prompt=anchor_prompt,
                                model=clip_model,
                                processor=clip_processor,
                                device=device,
                                batch_size=args.clip_batch_size,
                            )

                            # Positive/negative guidance CLIP scores used only for
                            # strength_anchor.
                            scale_to_clip_guidance_pos[scale] = clip_mean_score_for_frames_multi_prompt(
                                frames=frames,
                                prompts=guidance_pos_prompts,
                                model=clip_model,
                                processor=clip_processor,
                                device=device,
                                batch_size=args.clip_batch_size,
                            )
                            scale_to_clip_guidance_neg[scale] = clip_mean_score_for_frames_multi_prompt(
                                frames=frames,
                                prompts=guidance_neg_prompts,
                                model=clip_model,
                                processor=clip_processor,
                                device=device,
                                batch_size=args.clip_batch_size,
                            )

                            # Frame-aligned LPIPS to baseline.
                            scale_to_lpips[scale] = lpips_mean_for_frame_pairs(
                                base_frames=base_frames,
                                edit_frames=frames,
                                lpips_model=lpips_model,
                                device=device,
                                batch_size=args.lpips_batch_size,
                            )

                        if len(scales_present) < 2 or baseline_scale not in scales_present:
                            raise ValueError(
                                f"Need baseline plus at least one additional scale for {concept}/{attr}/{sample_id}. "
                                f"Found: {scales_present}"
                            )

                        m_strength_anchor = compute_strength_anchor(
                            scale_to_clip_guidance_pos,
                            scale_to_clip_guidance_neg,
                            scales_for_attr,
                        )
                        m_monotonicity_anchor = monotonicity_from_scale_scores(
                            scale_to_clip_anchor,
                            scales_for_attr,
                            eps=float(args.monotonicity_eps),
                        )
                        m_one_sided_span_anchor = one_sided_span_from_scale_scores(
                            scale_to_clip_anchor,
                            scales_for_attr,
                        )
                        m_lpips_preservation = compute_lpips_preservation(
                            scale_to_lpips,
                            baseline_scale,
                        )

                        writer.writerow(
                            {
                                "method": method,
                                "concept": concept,
                                "attribute": attr,
                                "sample": sample_id,
                                "baseline_scale": baseline_scale,
                                "last_scale": last_scale,
                                "total_scales": total_scales,
                                "scales_present": ",".join(scales_present),
                                "anchor_prompt": anchor_prompt,
                                "strength_anchor": m_strength_anchor,
                                "monotonicity_anchor": m_monotonicity_anchor,
                                "one_sided_span_anchor": m_one_sided_span_anchor,
                                "lpips_preservation": m_lpips_preservation,
                            }
                        )
                        f.flush()

                        if args.resume:
                            seen_keys.add(key)
                        if pbar is not None:
                            pbar.update(1)

                    except Exception as e:
                        if args.skip_errors:
                            print(f"Warning: skipping {concept}/{attr}/{sample_id}: {e}")
                            continue
                        raise

        if pbar is not None:
            pbar.close()


if __name__ == "__main__":
    main()
