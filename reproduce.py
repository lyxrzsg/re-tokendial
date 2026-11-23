"""Portable single-job entry points using the November Wan training and evaluation code."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "TokenDial"
RUN = ROOT / "artifacts/run"
PLAN = json.loads((RUN / "plan.json").read_text(encoding="utf-8"))
JOBS = {job["id"]: job for job in PLAN["jobs"]}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "train", "generate", "score"])
    parser.add_argument("job", choices=JOBS)
    parser.add_argument("--clips", type=Path, help="Directory containing the training MP4 files")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--work", type=Path, default=ROOT / "runs")
    parser.add_argument("--arm", choices=["native", "author"], default="native")
    args = parser.parse_args()
    job = JOBS[args.job]
    out = args.work.resolve() / args.job / args.arm
    ckpt = args.checkpoint.resolve() if args.checkpoint else out / "weights/epoch-1.safetensors"
    meta = ROOT / "datasets" / args.job / "metadata.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "training/overlay"), str(REPO), env.get("PYTHONPATH", "")])
    if args.stage in {"verify", "train"}:
        if not args.clips:
            parser.error("verify and train require --clips")
        clips = args.clips.resolve()
        with meta.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if digest(clips / row["video"]) != row["sha256"]:
                raise ValueError("Training clip differs from the recorded input: " + row["video"])
        print(f"Verified {len(rows)} input clips", flush=True)
        if args.stage == "verify":
            return
        if args.arm != "native":
            parser.error("Self-training uses the native arm")
        motion = job["concept"] == "speed"
        entry = RUN / "motion_entry.py" if motion else ROOT / "training/appearance_entry.py"
        config = "configs/motion.yaml" if motion else "configs/appearance.yaml"
        cmd = [sys.executable, str(entry), "--config", config, "--dataset_base_path", str(clips),
               "--dataset_metadata_path", str(meta), "--output_path", str(out / "weights")]
        if motion:
            env["WAN_DINO_LOCAL"] = str(Path(env["WAN_DINO_LOCAL"]).resolve())
        else:
            env["BASELINE_BERT"] = str(Path(env["BASELINE_BERT"]).resolve())
            cmd += ["--direction", job["training_guidance"], "--internvideo_pretrained_path",
                    str(Path(env["INTERNVIDEO_CKPT"]).resolve(strict=True))]
    else:
        # Each arm gets its own record so a released checkpoint and a self-trained one can coexist.
        dest = out / args.arm
        record = out / "trained.json"
        if args.stage == "generate":
            data = {"weights": {args.arm: {"path": str(ckpt), "sha256": digest(ckpt)}}}
            out.mkdir(parents=True, exist_ok=True)
            if record.exists() and json.loads(record.read_text()) != data:
                raise ValueError("Checkpoint changed. Choose a fresh --work directory.")
            record.write_text(json.dumps(data, indent=2))
        cmd = [sys.executable, str(RUN / "paper_eval.py"), "--job", args.job, "--checkpoint", str(ckpt),
               "--out", str(dest)]
        if args.stage == "score":
            cmd.append("--score-only")
            env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()

