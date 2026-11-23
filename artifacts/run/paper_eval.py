"""Evaluate the locally trained Table 7 offsets with the released Wan and scoring code."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from paper_train import save_atomic, sha256


def write_json(path, data):
    save_atomic(lambda d, f: f.write_text(json.dumps(d, indent=2)+"\n", encoding="utf-8"), data, path)


def valid_video(path, cfg):
    import cv2
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - cfg["inference_fps"]) > 0.001:
            raise ValueError(f"Invalid frame rate or broken video: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (cfg["inference_height"], cfg["inference_width"]):
                raise ValueError(f"Invalid frame size: {path}")
            count += 1
    finally:
        cap.release()
    if count != cfg["inference_frames"]:
        raise ValueError(f"Decoded {count} frames: {path}")


def target_indices(original, pipe, prompt, words):
    try:
        return original(pipe, prompt, words)
    except ValueError:
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        tokens = pipe.tokenizer.tokenizer.convert_ids_to_tokens(ids[0][:int(mask[0].sum())].tolist())
        def boundary(token):
            return token.startswith(("▁", "Ġ", "<")) or not token[:1].isalnum()
        def norm(token):
            return token.lower().removeprefix("▁").removeprefix("Ġ").removeprefix("##")
        targets = set(words)
        matched = set()
        for start, token in enumerate(tokens):
            if not boundary(token):
                continue
            for end in range(start + 1, len(tokens) + 1):
                if end == len(tokens) or boundary(tokens[end]):
                    text = "".join(map(norm, tokens[start:end]))
                    if text in targets or text.rstrip("s") in targets:
                        matched.update(range(start, end))
                    break
        if not matched:
            raise
        return sorted(matched)


def mask_target_words(job, prompt):
    words = job["target_words"]
    if job["id"] == "ink_redder":
        # Two released captions describe ink without a standalone 'ink' token.
        if prompt.startswith("Inky plume blooming in clear water"):
            return ["inky", "plume"]
        if prompt.startswith("Bold sumi-e brushstroke on textured rice paper"):
            return ["brushstroke"]
    if job["id"] != "person_older":
        return words
    subjects = (*words, "traveler", "child", "musician", "writer", "chef", "couple", "scientist")
    matches = set(re.findall(r"[a-z]+", prompt.lower()))
    targets = [word for word in subjects if word in matches]
    if not targets:
        raise ValueError(f"No person subject configured for masked evaluation: {prompt}")
    return targets


def evaluate_videos(manifest, output, repo):
    common = ["--manifest", str(manifest), "--guidance-dir", str(repo / "evaluation/guidance"),
              "--device", "cpu", "--stride", "4", "--max-frames", "16", "--monotonicity-eps", "0.0001"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for command in [
        ["evaluate", *common, "--out-dir", str(output / "metrics")],
        ["align", *common, "--metrics-csv", str(output / "metrics/metrics.csv"), "--out-dir", str(output / "alignment")],
    ]:
        subprocess.run([sys.executable, "-m", "evaluation", *command], env=env, cwd=repo, check=True)
    return json.loads((output / "alignment/alignment.json").read_text())["passed"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parent.parent / "TokenDial"
    plan = json.loads((root / "plan.json").read_text())
    job = next(j for j in plan["jobs"] if j["id"] == args.job)
    settings = plan["evaluation"]
    motion = job["concept"] == "speed"
    out = args.out.resolve()
    manifest = out / "manifest.json"
    if args.score_only:
        if not evaluate_videos(manifest, out, repo):
            raise RuntimeError("Author-score alignment failed")
        return

    from tokendial import inference
    prompts_file = root / "held_out_prompts.json"
    frozen = json.loads(prompts_file.read_text())[args.job]
    prompts = frozen["released"] + frozen["local_supplement"]
    if sha256(repo / frozen["public_file"]) != frozen["public_file_sha256"]:
        raise ValueError("Released prompt file changed")
    if not args.pilot and len(prompts) != settings["prompts_per_task"]:
        raise ValueError("Wrong prompt count")
    ckpt = args.checkpoint.resolve()
    checkpoint_hash = sha256(ckpt)
    protocol_spec = json.loads((out.parent / "trained.json").read_text())
    assert not args.pilot
    assert protocol_spec["weights"][out.name]["sha256"] == checkpoint_hash
    out.mkdir(parents=True, exist_ok=True)
    status = dict(job=args.job, checkpoint=str(ckpt), checkpoint_sha256=checkpoint_hash,
                  prompts_sha256=sha256(prompts_file), plan_sha256=sha256(root / "plan.json"), pilot=args.pilot)
    protocol = out / "invocation.json"
    if protocol.is_file() and json.loads(protocol.read_text()) != status:
        raise ValueError("Output directory already contains results for a different invocation")
    write_json(protocol, status)
    pipe = inference.load_pipeline(str(ckpt))
    pipe.dit.rgb_skip_first_latent = motion
    original_find = inference.find_target_token_indices
    inference.find_target_token_indices = lambda p, t, w: target_indices(original_find, p, t, w)

    def call_pipe(**kwargs):
        return pipe(cfg_scale=settings["text_cfg"], num_inference_steps=settings["inference_steps"], **kwargs)

    class ConfiguredPipeline:
        def __getattr__(self, key):
            return getattr(pipe, key)
        def __call__(self, **kwargs):
            return call_pipe(**kwargs)

    scales = settings["motion_wan_rgb_cfg_scales"] if motion else settings["strengths"]
    modes = ["nomask", "mask"]
    seeds = settings["seeds"][:1] if args.pilot else settings["seeds"]
    samples = []
    for index, prompt in enumerate(prompts):
        for seed in seeds:
            for mode in modes:
                sample_id = f"prompt{index:02}_seed{seed}"
                folder = out / mode / sample_id
                folder.mkdir(parents=True, exist_ok=True)
                generation = dict(prompt=prompt, seed=seed, negative_prompt=inference.DEFAULT_NEGATIVE_PROMPT,
                                  height=settings["inference_height"], width=settings["inference_width"],
                                  num_frames=settings["inference_frames"])
                mask = None
                if out.name == "repository" and any(not (folder / f"level{level}.mp4").is_file() for level in range(1, len(scales))):
                    mask = inference.build_mask(ConfiguredPipeline(), SimpleNamespace(
                        **generation, target_words=mask_target_words(job, prompt), attn_timesteps=None,
                        attn_layers="13,15,17,21", mask_quantile=0.75, mask_gamma=0.6,
                        save_mask_vis=False, output_dir=str(folder),
                    ), scales)
                clips = []
                for level, scale in enumerate(scales):
                    path = folder / f"level{level}.mp4"
                    if not path.is_file():
                        if mode == "mask" and (level == 0 or motion):
                            path = out / "nomask" / sample_id / f"level{level}.mp4"
                        else:
                            pipe.dit.use_rgb_token = level != 0
                            pipe.dit.rgb_token_weight = 1.0 if level else 0.0
                            if mode == "mask":
                                from timing_hook import generate
                                indices = inference.find_target_token_indices(pipe, prompt, mask_target_words(job,prompt))
                                video = generate(pipe, call_pipe, generation, scale, indices, folder, level)
                            else:
                                video = call_pipe(**generation,tiled=True,rgb_cfg_scale=scale,rgb_attention_mask=mask)
                            temp = path.with_suffix(".tmp.mp4")
                            inference.save_video(video, str(temp), fps=settings["inference_fps"], quality=5)
                            valid_video(temp, settings)
                            temp.replace(path)
                    valid_video(path, settings)
                    clips.append(dict(intensity=level, model_scale=scale, is_baseline=level == 0,
                                      video=str(path), sha256=sha256(path)))
                samples.append(dict(method=mode, concept=job["concept"], attribute="faster" if motion else job["attribute"],
                                    sample=sample_id, prompt=prompt, seed=seed, checkpoint=str(ckpt),
                                    baseline_kind="zero_strength_t2v", clips=clips))
    write_json(manifest, {"schema_version": 1, "samples": samples})


if __name__ == "__main__":
    main()


