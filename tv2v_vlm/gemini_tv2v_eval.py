"""Score a five-level TV2V slider against its source using Gemini."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import time


MODEL = "gemini-3.8-flash"
SCALES = (-4, -2, 0, 2, 4)
METRICS = (
    "prompt_following",
    "id_preserving",
    "background_consistency",
    "progressive_intensity",
    "source_preservation",
)
SCORE = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 5},
                   "reason": {"type": "string"}},
    "required": ["score", "reason"],
}
SCHEMA = {"type": "object", "properties": dict.fromkeys(METRICS, SCORE),
          "required": list(METRICS), "additionalProperties": False}


@dataclass(frozen=True)
class Sample:
    key: tuple[str, str, str, str]
    source: Path
    outputs: tuple[Path, ...]
    prompts: tuple[str, ...]
    zero_is_source_copy: bool


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_samples(manifest):
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1 or not isinstance(data.get("samples"), list) or not data["samples"]:
        raise ValueError("Expected schema_version=1 and a nonempty samples list")

    def video_path(value):
        path = (manifest.parent / value).resolve()
        if not path.is_file():
            raise ValueError(f"Missing video: {path}")
        return path

    samples = []
    seen = set()
    for row in data["samples"]:
        key = tuple(row[field] for field in ("concept", "attribute", "method", "sample"))
        if any(not isinstance(value, str) or not value.strip() for value in key) or key in seen:
            raise ValueError(f"Missing or duplicate sample identifiers: {key}")
        seen.add(key)
        clips = row["clips"]
        levels = [clip["intensity"] for clip in clips]
        if any(type(level) is not int for level in levels) or levels != list(SCALES):
            raise ValueError(f"{key}: expected ordered intensities {SCALES}")
        prompts = tuple(clip.get("prompt", "") for clip in clips)
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError(f"{key}: clip prompts must be strings")
        source = video_path(row["source_video"])
        outputs = tuple(video_path(clip["video"]) for clip in clips)
        copied = row.get("zero_is_source_copy", False)
        if type(copied) is not bool:
            raise ValueError(f"{key}: zero_is_source_copy must be boolean")
        zero = outputs[SCALES.index(0)]
        copied = copied or source.samefile(zero) or (
            source.stat().st_size == zero.stat().st_size and sha256(source) == sha256(zero)
        )
        samples.append(Sample(key, source, outputs, prompts, copied))
    return samples


@contextmanager
def open_clip(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = Fraction(str(capture.get(cv2.CAP_PROP_FPS))).limit_denominator(100_000)
        if frames < 1 or fps <= 0:
            raise ValueError(f"Invalid frame count or FPS: {path}")
        yield capture, frames, fps
    finally:
        capture.release()


def aligned_samples(clips, count):
    # Equal FPS reproduces the author's evenly spaced indices over the shortest clip.
    fps = min(fps for _, _, fps in clips)
    end = min(Fraction(frames - 1, rate) for _, frames, rate in clips)
    last = int(end * fps)
    if count == 1:
        indices = [last // 2]
    else:
        indices = [round(Fraction(last * i, count - 1)) for i in range(count)]
    for index in dict.fromkeys(indices):
        timestamp = Fraction(index, fps)
        yield float(timestamp), [round(timestamp * rate) for _, _, rate in clips]


def read_image(capture, frame_index, path):
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot decode frame {frame_index} from {path}")
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError(f"Cannot encode frame {frame_index} from {path}")
    return encoded.tobytes()


def make_prompt(sample):
    concept, attribute, _, _ = sample.key
    scale_prompts = "\n".join(
        f"  Scale {scale:+d}: {prompt}" for scale, prompt in zip(SCALES, sample.prompts) if prompt
    )
    source_levels = "-4, -2, +2 and +4" if sample.zero_is_source_copy else "-4, -2, 0, +2 and +4"
    return f"""
You are a meticulous video editing quality evaluator. You will see one frame from an external
source video, followed by five frames from edited outputs at the same playback time.
The source is a reference outside the scale sequence. The edited outputs are ordered
-4, -2, 0, +2, +4. Scale 0 is the intended neutral output.

Concept: {json.dumps(concept, ensure_ascii=False)}
Target attribute direction: {json.dumps(attribute, ensure_ascii=False)}
Increasing the scale should increase the target attribute in this direction. Negative scales
should move in the opposite direction from neutral. Assess the visible strength of the target
attribute itself along the ordered scales. Distance from the source may grow in both directions.
Any scale-specific instructions below describe the expected edits.
{scale_prompts}

Score the entire five-level sequence on each criterion. Give each an integer from 0 to 5
and a short reason based on visual evidence. When a frame does not establish actual speed,
frequency or timing, state that evidence limit in the relevant reason.

1. Prompt Following
Check that the same concept is edited in the requested direction and that the negative
and positive sides agree with the expected edits.
5 perfectly follows, 4 mostly follows, 3 partial or ambiguous, 2 weak,
1 barely related, 0 ignored or contradicted.

2. Identity Preserving
Check identity and category across the five outputs. Treat replacing the subject at any
scale as a severe violation. Allow visual changes explicitly requested by the target edit.
5 perfect, 4 minor variation, 3 recognizable with drift, 2 major drift or partial replacement,
1 severe identity change, 0 replaced or unrecognizable.

3. Background Consistency
Check unedited background regions, layout and camera viewpoint across the five outputs.
5 stable, 4 subtle changes, 3 noticeable drift, 2 significant redraw, 1 severe drift,
0 completely altered.

4. Progressive Intensity
Check for a smooth monotonic increase of the target attribute from -4 through 0 to +4.
5 clear and smooth, 4 mostly monotonic, 3 inconsistent, 2 weak or unclear,
1 reversed or chaotic, 0 no meaningful progression.

5. Source Preservation
Compare scales {source_levels} individually with the external source reference and
give one score for their preservation of content outside the intended edit. Look at
unrelated objects, scene geometry, camera framing, untargeted subject details and
visible action state. Allow appearance, motion, lighting or style changes needed for the
requested edit. Base the score on the source comparisons, including drift shared by
multiple outputs. An exact source copy at scale 0 is excluded when indicated above.
5 almost all source content preserved, 4 minor unrelated drift, 3 noticeable drift
but the source scene is clearly recognizable, 2 major collateral changes, 1 severe
source drift, 0 the source scene or event is effectively replaced.

Return JSON with a score and reason for each criterion.
""".strip()


def parse_scores(raw):
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != set(METRICS):
        raise ValueError("Expected all five rubric fields")
    for metric, entry in result.items():
        if (not isinstance(entry, dict) or type(entry.get("score")) is not int
                or not 0 <= entry["score"] <= 5 or not isinstance(entry.get("reason"), str)
                or not entry["reason"].strip()):
            raise ValueError(f"Invalid score or reason for {metric}")
    return result


def evaluate(call, prompt, images, max_retries):
    attempts = []
    for attempt in range(max_retries):
        raw = ""
        try:
            raw, model_version = call(prompt, images)
            return {"scores": parse_scores(raw), "model_version": model_version,
                    "raw_response": raw, "previous_attempts": attempts}
        except Exception as error:
            # SDK exception messages may contain request information.
            attempts.append({"error_type": type(error).__name__,
                             "api_code": getattr(error, "code", None), "raw_response": raw})
            if attempt + 1 < max_retries:
                time.sleep(0.5 * 2**attempt)
    return {"scores": None, "previous_attempts": attempts}


def request_frames(sample, count):
    paths = (sample.source, *sample.outputs)
    with ExitStack() as stack:
        clips = [stack.enter_context(open_clip(path)) for path in paths]
        for timestamp, indices in aligned_samples(clips, count):
            images = [read_image(capture, index, path)
                      for (capture, _, _), index, path in zip(clips, indices, paths)]
            yield timestamp, indices, images


def average(sample, rows):
    values = {metric: mean(row["scores"][metric]["score"] for row in rows) for metric in METRICS}
    return dict(zip(("concept", "attribute", "method", "sample"), sample.key)) | {
        "timestamps_scored": len(rows),
        "zero_is_source_copy": sample.zero_is_source_copy,
        **{f"{metric}_avg": round(value, 6) for metric, value in values.items()},
        "overall_4": round(mean(values[metric] for metric in METRICS[:4]), 6),
    }


def result_key(row):
    return tuple(row["sample_key"]), tuple(row["frame_indices"])


def run(samples, output, protocol, call, max_retries=3, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output / "protocol.json"
    log_path = output / "scores.jsonl"
    cached = {}
    if protocol_file.exists() and resume:
        if json.loads(protocol_file.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Protocol or input videos changed; choose a new output directory")
        if log_path.exists():
            with log_path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["scores"] is not None:
                        cached[result_key(row)] = row
    elif any(output.iterdir()):
        raise ValueError("Output directory is nonempty; choose a new directory or use --resume")
    else:
        protocol_file.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = []
    with log_path.open("a", encoding="utf-8") as log:
        for sample in samples:
            prompt = make_prompt(sample)
            rows = []
            for timestamp, indices, images in request_frames(sample, protocol["num_frames"]):
                row = {"sample_key": sample.key, "timestamp_seconds": timestamp, "frame_indices": indices,
                       "frame_order": ["source", *SCALES], "model": protocol["model"],
                       "zero_is_source_copy": sample.zero_is_source_copy}
                existing = cached.get(result_key(row))
                if existing is not None:
                    rows.append(existing)
                    continue
                row.update(evaluate(call, prompt, images, max_retries))
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                if row["scores"] is None:
                    raise RuntimeError(f"Scoring failed for {sample.key} at {timestamp:.3f}s; see {log_path}")
                rows.append(row)
            summary.append(average(sample, rows))
    temp = output / "scores.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    temp.replace(output / "scores.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Decode and validate locally without calling Gemini")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_frames < 1 or args.max_retries < 1:
        parser.error("--num-frames and --max-retries must be positive")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not args.dry_run and (not args.out_dir or not key):
        parser.error("Online scoring requires --out-dir and GEMINI_API_KEY. Use --dry-run for offline validation.")
    manifest = args.manifest.resolve()
    samples = load_samples(manifest)
    if args.dry_run:
        count = sum(1 for sample in samples for _ in request_frames(sample, args.num_frames))
        print(f"Offline validation passed for {len(samples)} samples and {count} aligned timestamps.")
        return

    from google import genai
    from google.genai import types

    paths = sorted({path for sample in samples for path in (sample.source, *sample.outputs)})
    protocol = {"model": args.model, "num_frames": args.num_frames,
                "script_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(manifest),
                "video_sha256": {str(path): sha256(path) for path in paths},
                "scales": list(SCALES), "input": "one external source frame and five output frames per timestamp"}
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120_000)) as client:
        def call(prompt, images):
            labels = ["External source reference", *(f"Edited output at scale {scale:+d}" for scale in SCALES)]
            contents = [prompt]
            for label, image in zip(labels, images):
                contents.extend([label, types.Part.from_bytes(data=image, mime_type="image/png")])
            response = client.models.generate_content(
                model=args.model, contents=contents,
                config={"response_mime_type": "application/json", "response_json_schema": SCHEMA},
            )
            return response.text or "", getattr(response, "model_version", None)

        run(samples, args.out_dir, protocol, call, args.max_retries, args.resume)
    print(f"Wrote {len(samples)} sample scores to {args.out_dir / 'scores.csv'}")


if __name__ == "__main__":
    main()
