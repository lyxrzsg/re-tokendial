# VLM Metrics (Gemini) — Video Slider Evaluation

This folder contains a Gemini-based VLM evaluator for **ConceptSlider-style video sliders**.

## What it does

Given a sample with **5 videos**:
- Scale **1**: base (before edit)
- Scale **2–5**: edited results with **increasing edit intensity**

The evaluator:
- samples **K aligned timestamps** (default **3**) using the same `frame_idx` across all 5 videos
- sends **5 frames** (Scale 1..5) to Gemini in one request
- asks Gemini to output **4 metrics**, each scored **0–5** (higher is better) **with reasons**
- writes both summarized and per-frame debug outputs

## Dataset layout expected

```
{root}/{concept}/{attribute}/{method}/{scale}/{video}.mp4
```

Example:

```
/sensei-fs-3/users/zhixuanl/colligo_train/siggraph_AMT_sample/aurora/brighter/ours/1/row_8_col_1.mp4
/sensei-fs-3/users/zhixuanl/colligo_train/siggraph_AMT_sample/aurora/brighter/ours/2/row_8_col_1.mp4
...
/sensei-fs-3/users/zhixuanl/colligo_train/siggraph_AMT_sample/aurora/brighter/ours/5/row_8_col_1.mp4
```

## Metrics (0–5)

- **prompt_following**: edit direction matches “make `{concept}` more `{attribute}`”, and Scale 5 is strongest/most aligned
- **id_preserving**: main subject identity/category stays the same (no replacement)
- **background_consistency**: non-edited/background regions remain stable across scales
- **progressive_intensity**: edit strength increases monotonically from Scale 1 → 5

## Setup

### 1) Install dependencies

Use whatever environment you already have; you’ll need at least:
- `google-genai`
- `opencv-python`
- `Pillow`
- `numpy`

### 2) Set API key

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

## Run

### Debug (run 1 item)

```bash
python vlm_metrics/gemini_slider_video_eval.py \
  --root /sensei-fs-3/users/zhixuanl/colligo_train/siggraph_AMT_sample \
  --out_csv /sensei-fs-3/users/zhixuanl/colligo_train/metrics/gemini_slider_scores_debug.csv \
  --out_jsonl /sensei-fs-3/users/zhixuanl/colligo_train/metrics/gemini_slider_scores_debug.jsonl \
  --methods ours \
  --num_frames 3 \
  --limit 1
```

### Full run (multiple methods)

```bash
python vlm_metrics/gemini_slider_video_eval.py \
  --root /sensei-fs-3/users/zhixuanl/colligo_train/siggraph_AMT_sample \
  --out_csv /sensei-fs-3/users/zhixuanl/colligo_train/metrics/gemini_slider_scores.csv \
  --out_jsonl /sensei-fs-3/users/zhixuanl/colligo_train/metrics/gemini_slider_scores.jsonl \
  --methods ours,conceptslider,freeslider,sliderspace,textslider \
  --num_frames 3
```

### Resume

If the CSV already exists, you can resume and skip items already present:

```bash
python vlm_metrics/gemini_slider_video_eval.py \
  --root /sensei-fs-3/users/zhixuanl/colligo_train/siggraph_AMT_sample \
  --out_csv /sensei-fs-3/users/zhixuanl/colligo_train/metrics/gemini_slider_scores.csv \
  --out_jsonl /sensei-fs-3/users/zhixuanl/colligo_train/metrics/gemini_slider_scores.jsonl \
  --methods ours,conceptslider,freeslider,sliderspace,textslider \
  --num_frames 3 \
  --resume
```

## Outputs

### CSV (`--out_csv`)
One row per `(concept, attribute, method, filename)` with:
- `*_avg`: averages over sampled frames
- `*_per_frame`: list of per-frame scores (length = `num_frames`)
- `*_reasons`: list of per-frame explanations (length = `num_frames`)
- `frame_indices`: the sampled indices used for all 5 videos

### JSONL (`--out_jsonl`)
One row per evaluated frame with:
- identifiers (`concept/attribute/method/filename/frame_idx`)
- parsed `scores` + `reasons`
- `raw_response_text` (full Gemini output for debugging)

## Notes / Tips

- **Frame alignment**: the script uses `min(frame_count)` across the 5 videos, then samples indices in `[0, n_min-1]`.
- **Prompt wording**: edit prompt is generated from `{concept}` and `{attribute}` in `build_prompt()`; feel free to refine it for your task.

