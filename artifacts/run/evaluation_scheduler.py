"""Start generation under the training scheduler's idle-GPU policy, then score on CPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import training_scheduler as guard
from paper_train import save_atomic

p = argparse.ArgumentParser()
p.add_argument("--job", required=True)
p.add_argument("--checkpoint", required=True, type=Path)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--pilot", action="store_true")
p.add_argument("--gpu-free", type=int, default=78000)
p.add_argument("--ram", type=int, default=20480)
args = p.parse_args()
root, base, repo = guard.ROOT, guard.BASE, guard.REPO
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)


def status(state, **extra):
    save_atomic(lambda d, f: f.write_text(json.dumps(d, indent=2), encoding="utf-8"),
                dict(status=state, utc=time.strftime("%FT%TZ", time.gmtime()), **extra),
                out / "evaluation_status.json")


def invoke(gpu, command):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(repo),
               DIFFSYNTH_MODEL_BASE_PATH=str(base / "models"), DIFFSYNTH_SKIP_DOWNLOAD="true",
               HF_HUB_OFFLINE="1", HF_HOME=str(base / "cache/huggingface"),
               TORCH_HOME=str(base / "cache/torch"), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if not gpu:
        with (out / "score.log").open("a") as stream:
            subprocess.run([*command, "--score-only"], env=env, cwd=repo, stdout=stream,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        return True

    cards, apps = guard.snapshot()
    free, used, util = cards[gpu]
    if apps[gpu] or not (free > args.gpu_free and used < 512 and util < 5 and guard.host_ram() >= args.ram):
        return False
    with (out / "generation.log").open("a") as stream:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        try:
            while guard.running:
                try:
                    code = child.wait(timeout=10)
                    if code:
                        raise RuntimeError(f"Generation exited {code}")
                    return True
                except subprocess.TimeoutExpired:
                    try:
                        _, apps = guard.snapshot()
                        foreign = apps[gpu] - {child.pid}
                    except Exception:
                        foreign = {"gpu_query_failed"}
                    if foreign:
                        return False
            return False
        finally:
            guard.kill_ours(child)


def main():
    signal.signal(signal.SIGINT, guard.signal_stop)
    signal.signal(signal.SIGTERM, guard.signal_stop)
    command = [sys.executable, str(root / "paper_eval.py"), "--job", args.job,
               "--checkpoint", str(args.checkpoint.resolve()), "--out", str(out)]
    if args.pilot:
        command.append("--pilot")
    idle = {}
    deadline = time.monotonic() + 48 * 3600
    with (out / "evaluation.lock").open("a") as instance:
        fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while guard.running and time.monotonic() < deadline:
            if (out / "manifest.json").is_file():
                status("scoring_cpu")
                invoke("", command)
                status("evaluated", pilot=args.pilot)
                return
            try:
                cards, apps = guard.snapshot()
            except (subprocess.SubprocessError, OSError, ValueError, KeyError):
                idle.clear()
                status("waiting_for_gpu_query")
                time.sleep(10)
                continue
            available = {gpu for gpu, (free, used, util) in cards.items()
                         if free > args.gpu_free and used < 512 and util < 5 and not apps[gpu]}
            if guard.host_ram() < args.ram:
                available.clear()
            now = time.monotonic()
            idle = {gpu: idle.get(gpu, now) for gpu in available}
            stable = sorted(gpu for gpu, since in idle.items() if now - since >= 120)
            if stable:
                gpu = stable[0]
                with (base / "artifacts" / f"t2v_gpu_{gpu}.lock").open("a") as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        idle.pop(gpu)
                        continue
                    status("generating", gpu=gpu, pilot=args.pilot)
                    if invoke(gpu, command):
                        continue
                idle.clear()
            status("waiting_for_unused_gpu", idle_candidates=len(available), pilot=args.pilot)
            time.sleep(10)
        status("stopped" if not guard.running else "expired")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=str(error))
        raise

