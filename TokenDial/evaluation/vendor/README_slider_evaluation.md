# Slider Evaluation

This script evaluates continuous attribute-control sliders using the four metrics used in the final evaluation. The implementation keeps the original code-level metric names, while the paper reports them with the following names:

| Code metric | Paper name |
| --- | --- |
| `strength_anchor` | **Range (CR)** |
| `monotonicity_anchor` | **Monotonicity** |
| `one_sided_span_anchor` | **Smoothness (CSM)** |
| `lpips_preservation` | **LPIPS Preservation** |

The metric definitions and frame/model scoring logic are kept unchanged from the original evaluation code. The release version only removes unused metrics, venue-specific naming, hardcoded local paths, and the fixed `row_X_col_Y` sample-name requirement.

## 1. Installation

Python 3.9+ is recommended.

```bash
pip install numpy torch transformers lpips decord pillow tqdm
```

The script uses:

- CLIP: `openai/clip-vit-base-patch32`
- LPIPS: AlexNet backbone

## 2. Video directory format

Scale folders must have numeric names. The script sorts them numerically, uses the smallest scale as the baseline, and uses the largest scale as the last scale.

### With a method folder

```text
results/
└── aurora/
    └── brighter/
        └── my_method/
            ├── 1.0/
            │   ├── sample_001.mp4
            │   └── sample_002.mp4
            ├── 1.5/
            │   ├── sample_001.mp4
            │   └── sample_002.mp4
            ├── 2.0/
            ├── 2.5/
            └── 3.0/
```

Run with `--method my_method`.

### Without a method folder

```text
results/
└── aurora/
    └── brighter/
        ├── 1.0/
        ├── 1.5/
        ├── 2.0/
        ├── 2.5/
        └── 3.0/
```

In this case, omit `--method`.

An optional `images/` folder under each scale is also supported:

```text
3.0/images/sample_001.mp4
```

### Sample naming

Sample filenames can be arbitrary, for example:

```text
sample_001.mp4
row_1_col_0.mp4
cat.mp4
```

The baseline scale is used to discover the sample list. A sample is matched across scales by exact filename.

## 3. Guidance JSON format

`strength_anchor` requires one guidance JSON file per `(concept, attribute)`:

```text
guidance/
└── aurora/
    └── brighter.json
```

Example `brighter.json`:

```json
{
  "direction_prompts": [
    "a brighter aurora",
    "an aurora with increased brightness"
  ],
  "neg_direction_prompts": [
    "a dimmer aurora",
    "an aurora with reduced brightness"
  ]
}
```

Both lists must be non-empty.

The expected path is:

```text
{guidance_dir}/{concept}/{attribute}.json
```

## 4. Usage

### Example with a method folder

```bash
python evaluate_sliders.py \
    --root /path/to/results \
    --method my_method \
    --guidance_dir /path/to/guidance \
    --out_csv metrics.csv
```

### Example without a method folder

```bash
python evaluate_sliders.py \
    --root /path/to/results \
    --guidance_dir /path/to/guidance \
    --out_csv metrics.csv
```

By default, numeric scale folders are discovered automatically. To evaluate only specific scales:

```bash
python evaluate_sliders.py \
    --root /path/to/results \
    --guidance_dir /path/to/guidance \
    --scales 1.0,1.5,2.0,2.5,3.0 \
    --out_csv metrics.csv
```

Useful options:

```text
--stride 4                 sample every 4th frame
--max_frames 16            use at most 16 sampled frames per video
--device auto              auto / cpu / cuda
--clip_batch_size 8
--lpips_batch_size 8
--monotonicity_eps 1e-4
--concepts aurora,smoke     optional concept filter
--attributes brighter      optional attribute filter
--resume                    append and skip completed samples
--skip_errors               skip bad/missing samples instead of stopping
```

## 5. Metric definitions

For each scale video, CLIP scores are averaged over sampled frames. By default, frames `0, 4, 8, ...` are used, up to 16 frames.

### Range (CR) — `strength_anchor`

This metric uses the positive and negative prompt sets from the guidance JSON.

For each scale `i`, let:

```text
P_i = mean CLIP score over direction_prompts
N_i = mean CLIP score over neg_direction_prompts
```

Only the baseline and last scale are used:

```text
pos_strength = P_last - P_base
neg_strength = N_base - N_last

strength_anchor = 0.5 * (pos_strength + neg_strength)
```

Higher is better.

### Monotonicity — `monotonicity_anchor`

A single attribute-focused anchor prompt is used for every scale, e.g.:

```text
aurora extremely brighter
```

Let the per-scale CLIP scores be:

```text
A_0, A_1, ..., A_K
```

Compute adjacent differences:

```text
d_i = A_i - A_{i-1}
```

Then:

```text
monotonicity_anchor = fraction of d_i > 1e-4
```

Higher is better. A value of `1.0` means every adjacent slider step increases the anchor CLIP score.

### Smoothness (CSM) — `one_sided_span_anchor`

This uses the same anchor CLIP sequence as `monotonicity_anchor`:

```text
one_sided_span_anchor = std(d_1, d_2, ..., d_K)
```

Lower means the semantic change is more evenly distributed across adjacent slider steps.

### LPIPS Preservation — `lpips_preservation`

Each non-baseline scale is compared with the baseline using frame-aligned LPIPS.

For each non-baseline scale:

```text
LPIPS_scale = mean LPIPS(baseline_frame_t, edited_frame_t)
```

Then:

```text
lpips_preservation = mean LPIPS_scale over all non-baseline scales
```

Lower indicates better visual preservation relative to the baseline.

## 6. Output CSV

The output contains one row per `(concept, attribute, sample)`:

```text
method
concept
attribute
sample
baseline_scale
last_scale
total_scales
scales_present
anchor_prompt
strength_anchor
monotonicity_anchor
one_sided_span_anchor
lpips_preservation
```

## 7. Important notes

- Numeric scale values are used to determine ordering. The scale spacing itself is **not** used in the metric formulas.
- `strength_anchor` uses the guidance JSON positive/negative prompt sets.
- `monotonicity_anchor` and `one_sided_span_anchor` use a single attribute-focused anchor prompt.
- `lpips_preservation` does not use any text prompt.
- Guidance JSON is required; the release script does not silently fall back to another `strength_anchor` definition if guidance is missing.
