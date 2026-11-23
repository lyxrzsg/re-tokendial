"""Run Table 7 pilot and formal jobs only on a continuously idle GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
REPO = BASE / "TokenDial"
DINO = ROOT / "dependencies/dinov2"
MOTION = "speed_higher_motion_magnitude"
APPEARANCE_PILOT = "person_older"
running = True


def signal_stop(signum, frame):
    global running
    running = False


def status(state, **detail):
    temp = ROOT / "scheduler_status.tmp"
    temp.write_text(json.dumps(dict(status=state, updated_utc=time.strftime("%FT%TZ", time.gmtime()), **detail)))
    temp.replace(ROOT / "scheduler_status.json")


def gpu_query(query, field):
    return subprocess.check_output(
        ["nvidia-smi", f"--query-{query}={field}", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    ).splitlines()


def snapshot():
    cards = {}
    for row in gpu_query("gpu", "uuid,memory.free,memory.used,utilization.gpu"):
        uuid, free, used, util = (item.strip() for item in row.split(","))
        cards[uuid] = (int(free), int(used), int(util))
    apps = {uuid: set() for uuid in cards}
    for row in gpu_query("compute-apps", "gpu_uuid,pid"):
        uuid, pid = (item.strip() for item in row.split(","))
        apps[uuid].add(int(pid))
    return cards, apps


def host_ram():
    line = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return int(line.split()[1]) // 1024


def kill_ours(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run(gpu, dataset, output, job, pilot, weight):
    cards, apps = snapshot()
    free, used, util = cards[gpu]
    if apps[gpu] or not (free > args.gpu_free and used < 512 and util < 5 and host_ram() >= args.ram):
        return "resource_yield"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               PYTHONPATH=str(REPO), DIFFSYNTH_MODEL_BASE_PATH=str(BASE / "models"),
               DIFFSYNTH_SKIP_DOWNLOAD="true", HF_HUB_OFFLINE="1",
               HF_HOME=str(BASE / "cache/huggingface"), TORCH_HOME=str(BASE / "cache/torch"),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    entrypoint = "paper_train.py"
    if job != MOTION:
        deps = ROOT / "dependencies"
        env["PYTHONPATH"] = os.pathsep.join(map(str, (deps / "appearance_overlay", deps / "python", REPO)))
        entrypoint = "appearance_train.py"
    command = [sys.executable, str(ROOT / entrypoint), "--plan", str(ROOT / "plan.json"),
               "--job", job, "--dataset", str(dataset), "--out", str(output), "--dino-repo", str(DINO),
               "--internvideo", str(weight)]
    if pilot:
        command += ["--pilot", "--steps", "2"]
    status("running_pilot" if pilot else "training", gpu=gpu, job=job, log=str(output / "train.log"))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "train.log").open("a") as log:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        try:
            while running:
                try:
                    code = proc.wait(timeout=10)
                    return "completed" if code == 0 else f"failed_exit_{code}"
                except subprocess.TimeoutExpired:
                    try:
                        _, apps = snapshot()
                        foreign = apps[gpu] - {proc.pid}
                    except Exception:
                        foreign = {"query_failed"}
                    if foreign:
                        return "resource_yield"
            return "stopped"
        finally:
            kill_ours(proc)


def main():
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    with (ROOT / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT / "scheduler_pid.json").write_text(json.dumps({"pid": os.getpid()}))
        deadline = time.monotonic() + 48 * 3600
        idle = {}
        while running and time.monotonic() < deadline:
            try:
                cards, apps = snapshot()
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                idle.clear()
                status("waiting_for_gpu_query", error=type(error).__name__)
                time.sleep(10)
                continue
            ready = []
            output = ROOT / "pilot/motion"
            dataset = ROOT / "pilot_data/speed/metadata.csv"
            dino_weights = BASE / "cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
            weight = ROOT / "dependencies/InternVideo2-stage2_1b-224p-f4.pt"
            has_dino = (DINO / "hubconf.py").is_file() and dino_weights.is_file()
            pilot_ok = (output / "complete.json").is_file()
            if dataset.is_file() and has_dino and not pilot_ok:
                ready.append((MOTION, dataset, output, True))
            appearance_data = ROOT / "pilot_data/person_older/metadata.csv"
            appearance_out = ROOT / "pilot/person_older"
            appearance_ok = (appearance_out / "complete.json").is_file()
            preflight = ROOT / "dependencies/internvideo2_cpu_preflight.log"
            appearance_ready = weight.is_file() and preflight.is_file() and (
                "Verified Stage2 appearance checkpoint and nonzero person_older guidance"
                in preflight.read_text(errors="replace")
            )
            if appearance_data.is_file() and appearance_ready and not appearance_ok:
                ready.append((APPEARANCE_PILOT, appearance_data, appearance_out, True))
            if pilot_ok or appearance_ok:
                plan = json.loads((ROOT / "plan.json").read_text())
                for job in plan["jobs"]:
                    name = job["id"]
                    data = ROOT / "data" / name
                    formal = ROOT / "checkpoints" / name
                    deps = pilot_ok and has_dino if job["concept"] == "speed" else appearance_ok and appearance_ready
                    if deps and (data / "ready.json").is_file() and (data / "metadata.csv").is_file() and not (formal / "complete.json").is_file():
                        ready.append((name, data / "metadata.csv", formal, False))
            if not ready:
                status("waiting_for_formal_training_data" if pilot_ok else "waiting_for_pilot_inputs",
                       motion_pilot_complete=pilot_ok, appearance_pilot_complete=appearance_ok,
                       dino_available=has_dino, internvideo_available=weight.is_file())
                idle.clear()
                time.sleep(30)
                continue
            memory = host_ram()
            free_cards = {gpu for gpu, (free, used, util) in cards.items()
                          if free > args.gpu_free and used < 512 and util < 5 and not apps[gpu]}
            eligible = free_cards if memory >= args.ram else set()
            now = time.monotonic()
            idle = {gpu: idle.get(gpu, now) for gpu in eligible}
            stable = sorted(gpu for gpu, start in idle.items() if now - start >= 120)
            if not stable:
                state = "waiting_for_host_ram" if free_cards and memory < args.ram else "waiting_for_unused_gpu"
                status(state, candidates=len(free_cards), mem_available_mib=memory, next_job=ready[0][0])
                time.sleep(10)
                continue
            gpu = stable[0]
            with (BASE / "artifacts" / f"t2v_gpu_{gpu}.lock").open("a") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    idle.pop(gpu)
                    continue
                state = run(gpu, ready[0][1], ready[0][2], ready[0][0], ready[0][3], weight)
            idle.clear()
            if state.startswith("failed"):
                status(state, job=ready[0][0], log=str(ready[0][2] / "train.log"))
                return
        status("stopped" if not running else "expired")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-free", type=int, default=120000)
    p.add_argument("--ram", type=int, default=20480)
    args = p.parse_args()
    main()

