"""Run the historical 12-direction or full Table 7 Wan reproduction."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from reproduce import JOBS, PLAN, REPO, ROOT, RUN, digest

sys.path.insert(0, str(REPO))
from evaluation.manifest import load_guidance, load_manifest
from evaluation.table7 import ATTRIBUTES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(path, obj):
    if path.exists():
        if read(path) != obj:
            raise ValueError(f"Inputs changed; select a fresh work directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def tasks(args):
    historical = {job["concept"]: job for job in JOBS.values()}
    frozen = read(RUN / "held_out_prompts.json")
    result = []
    for concept, attributes in ATTRIBUTES.items():
        base = historical[concept]
        speed = concept == "speed"
        old_attr = "faster" if speed else base["attribute"]
        for attr in attributes.split() if args.suite == "57" else [old_attr]:
            group = "person_walk" if attr == "walk_to_run" else concept
            name = base["id"] if attr == old_attr else f"{concept}_{attr}"
            prompt_name = "fire_explosion" if concept == "explosion" else group
            prompt_file = REPO / "evaluation/prompts" / (prompt_name + ".txt")
            prompts = [s.strip() for s in prompt_file.read_text(encoding="utf-8").splitlines() if s.strip()]
            if args.suite == "12":
                source = frozen[name]
                if digest(prompt_file) != source["public_file_sha256"] or prompts != source["released"]:
                    raise ValueError(f"Historical prompt mismatch: {name}")
                prompts += source["local_supplement"]
            seeds = PLAN["evaluation"]["seeds"] if args.suite == "12" else args.seeds
            scales = PLAN["evaluation"]["motion_wan_rgb_cfg_scales" if speed else "strengths"]
            if attr == "slower":
                scales = [-v for v in scales]
            if len(prompts) != (16 if args.suite == "12" else 8) or len(set(prompts)) != len(prompts):
                raise ValueError(f"Invalid prompt set: {name}")
            result.append(dict(id=name, concept=concept, manifest_concept=group, attribute=attr,
                               targets=base["target_words"], dataset=base["id"],
                               train_id=base["id"] if speed else name, prompts=prompts,
                               seeds=seeds, scales=scales))
    if args.scales_file:
        overrides = read(args.scales_file)
        if not isinstance(overrides, dict) or set(overrides) - {job["id"] for job in result}:
            raise ValueError("Scale overrides must be keyed by a job ID in this suite")
        for job in result:
            if job["id"] in overrides:
                values = overrides[job["id"]]
                if (not isinstance(values, list) or len(values) != 5
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                        or values[0] != 0
                        or not (all(a < b for a, b in zip(values, values[1:]))
                                or all(a > b for a, b in zip(values, values[1:])))):
                    raise ValueError(f"Expected five ordered finite scales from zero: {job['id']}")
                job["scales"] = values
    return result


def inputs(args, jobs):
    recipes = {}
    weights = {}
    seen = set()
    for job in jobs:
        key = job["dataset"]
        clips = args.data_root / key
        meta = clips / "metadata.csv"
        if not meta.exists():
            meta = ROOT / "datasets" / key / "metadata.csv"
        if key not in seen:
            with meta.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError(f"Empty training dataset: {meta}")
            for row in rows:
                if not row["prompt"].strip() or digest(clips / row["video"]) != row["sha256"]:
                    raise ValueError(f"Invalid caption or video hash in {meta}: {row['video']}")
            seen.add(key)
        _, guidance = load_guidance(args.guidance_dir, dict(concept=job["manifest_concept"], attribute=job["attribute"]))
        job["guidance_sha256"] = digest(guidance)
        train_id = job["train_id"]
        if train_id in recipes:
            continue
        motion = job["concept"] == "speed"
        config = REPO / "configs" / ("motion.yaml" if motion else "appearance.yaml")
        entry = RUN / "motion_entry.py" if motion else ROOT / "training/appearance_entry.py"
        dest = args.work / "training" / train_id
        weights[train_id] = dest / "epoch-1.safetensors"
        cmd = [sys.executable, str(entry), "--config", str(config), "--dataset_base_path", str(clips),
               "--dataset_metadata_path", str(meta), "--output_path", str(dest)]
        recipe = dict(command=cmd, metadata_sha256=digest(meta), config_sha256=digest(config))
        if not motion:
            direction = REPO / "configs/directions/person_older.json" if train_id == "person_older" else guidance
            cmd.extend(["--direction", str(direction), "--internvideo_pretrained_path", os.environ["INTERNVIDEO_CKPT"]])
            recipe["direction_sha256"] = digest(direction)
        recipes[train_id] = recipe
    return recipes, weights


def run(args):
    jobs = tasks(args)
    if args.stage == "plan":
        print(json.dumps(jobs, indent=2))
        return
    for env_name in ("INTERNVIDEO_CKPT", "BASELINE_BERT", "WAN_DINO_LOCAL"):
        path = Path(os.environ[env_name]).resolve(strict=True)
        if path.is_file() != (env_name == "INTERNVIDEO_CKPT"):
            raise ValueError(f"Invalid {env_name}: {path}")
        os.environ[env_name] = str(path)
    recipes, weights = inputs(args, jobs)
    if args.stage == "check":
        print(f"Verified {len(jobs)} directions and inputs for {len(recipes)} training jobs")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "training/overlay"), str(REPO), env.get("PYTHONPATH", "")])
    freeze(args.work / "protocol.json", dict(jobs=jobs, training=recipes, evaluation=PLAN["evaluation"],
                                             guidance_dir=str(args.guidance_dir),
                                             bert=env["BASELINE_BERT"], dino=env["WAN_DINO_LOCAL"]))
    samples = []
    for job in jobs:
        train_id = job["train_id"]
        ckpt = weights[train_id]
        marker = ckpt.parent / "completed.json"
        if not marker.exists():
            if ckpt.parent.exists() and any(ckpt.parent.iterdir()):
                raise ValueError(f"Incomplete training: move {ckpt.parent} aside before retrying")
            subprocess.run(recipes[train_id]["command"], cwd=REPO, env=env, check=True)
            freeze(marker, {"sha256": digest(ckpt)})
        if digest(ckpt) != read(marker)["sha256"]:
            raise ValueError(f"Checkpoint hash changed: {ckpt}")
        dest = args.work / "generated" / job["id"]
        protocol = dest / "task.json"
        freeze(protocol, job)
        subprocess.run([sys.executable, str(RUN / "paper_eval.py"), "--job", job["id"],
                        "--suite-task", str(protocol), "--checkpoint", str(ckpt), "--out", str(dest)],
                       cwd=REPO, env=env, check=True)
        rows = load_manifest(dest / "manifest.json")["samples"]
        expected = {(m, p, s) for m in ("nomask", "mask") for p in job["prompts"] for s in job["seeds"]}
        if len(rows) != len(expected) or {(r["method"], r["prompt"], r["seed"]) for r in rows} != expected:
            raise ValueError(f"Incomplete generation: {job['id']}")
        for r in rows:
            if (r["concept"] != job["manifest_concept"] or r["attribute"] != job["attribute"]
                    or [c["model_scale"] for c in r["clips"]] != job["scales"]
                    or [c["intensity"] for c in r["clips"]] != list(range(5))):
                raise ValueError(f"Incorrect attribute or levels: {job['id']}")
        samples.extend(rows)
    manifest = args.work / "manifest.json"
    freeze(manifest, dict(schema_version=1, samples=samples))
    from tempfile import mkdtemp
    # Keep incomplete or earlier score attempts separate.
    attempt = Path(mkdtemp(prefix="scores-", dir=args.work))
    common = ["--manifest", str(manifest), "--guidance-dir", str(args.guidance_dir), "--device", "cpu"]
    for command in (["evaluate", "--out-dir", str(attempt / "metrics")],
                    ["align", "--metrics-csv", str(attempt / "metrics/metrics.csv"),
                     "--out-dir", str(attempt / "alignment")]):
        subprocess.run([sys.executable, "-m", "evaluation", *command, *common], cwd=REPO, env=env, check=True)
    if not read(attempt / "alignment/alignment.json")["passed"]:
        raise RuntimeError("Author-score alignment failed")
    from collections import defaultdict
    from evaluation.metrics import METRICS, summarize
    grouped = defaultdict(list)
    with (attempt / "metrics/metrics.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["concept"], row["attribute"]].append(row)
    with (attempt / "by_attribute.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["concept", "attribute", "method", "sample_count", *METRICS,
                  "overall_mean_of_sample_scores", "overall_from_aggregate_metrics"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (group, attribute), rows in sorted(grouped.items()):
            for method, stats in summarize(rows)["methods"].items():
                concept = "person" if group == "person_walk" else group
                writer.writerow(dict(concept=concept, attribute=attribute, method=method, **stats))
    print(f"Scores and alignment: {attempt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "check", "run"])
    parser.add_argument("--suite", choices=["12", "57"], required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--guidance-dir", type=Path, default=REPO / "evaluation/guidance")
    parser.add_argument("--seeds", type=int, nargs=2, help="57 only; reconstruction default: 2 3")
    parser.add_argument("--scales-file", type=Path, help="Optional JSON: job ID to five actual Wan scales")
    args = parser.parse_args()
    if args.seeds and (args.suite == "12" or len(set(args.seeds)) != 2):
        parser.error("--seeds requires suite 57 and two distinct integers")
    args.seeds = args.seeds or [2, 3]
    args.work = (args.work or ROOT / "runs" / ("suite" + args.suite)).resolve()
    args.data_root = args.data_root.resolve()
    args.guidance_dir = args.guidance_dir.resolve()
    if args.stage == "run":
        # Training runs on Linux. Kernel locks are released automatically after a crash.
        import fcntl
        args.work.mkdir(parents=True, exist_ok=True)
        with (args.work / ".suite.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    else:
        run(args)
