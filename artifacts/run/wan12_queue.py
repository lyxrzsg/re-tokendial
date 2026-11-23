"""Queue the remaining eleven Wan attributes, using the running person protocol."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

root = Path(__file__).resolve().parent
base = root.parent.parent
art = base / "artifacts"
old = art / "table7_training_h20_20261116"
person = art / "person_formal_table2_v1"
repo = base / "TokenDial"
deps = old / "dependencies"
motion_id = "speed_higher_motion_magnitude"
python = sys.executable
sys.path.insert(0, str(old))
import training_scheduler as guard


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def freeze(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen protocol changed: {path}")
    else:
        path.write_bytes(data)


def prepare():
    plan = read(old / "plan.json")
    settings = read(person / "plan.json")["evaluation"]
    plan["evaluation"] = settings
    freeze(root / "plan.json", json.dumps(plan, indent=2).encode())
    freeze(root / "held_out_prompts.json", (old / "held_out_prompts.json").read_bytes())
    for name in ("training_scheduler.py", "evaluation_scheduler.py", "paper_train.py", "timing_hook.py"):
        freeze(root / name, (person / name).read_bytes())
    src = (person / "paper_eval.py").read_text()
    for before, after in [
        ('root / "protocol.json"', 'out.parent / "trained.json"'),
        ('if mode == "mask" and level == 0:', 'if mode == "mask" and (level == 0 or motion):'),
        ('path = out / "nomask" / sample_id / "level0.mp4"', 'path = out / "nomask" / sample_id / f"level{level}.mp4"'),
    ]:
        assert src.count(before) == 1, before
        src = src.replace(before, after)
    compile(src, "paper_eval.py", "exec")
    freeze(root / "paper_eval.py", src.encode())
    prompts = read(root / "held_out_prompts.json")
    spec = []
    report = read(root / "pack_hashes.json")
    for job in plan["jobs"]:
        name = job["id"]
        if name == "person_older":
            continue
        source = old / "data" / name
        if not (source / "metadata.csv").exists():
            source = root / "data" / name
            source.mkdir(exist_ok=True, parents=True)
            pack = art / f"wan12_20261121_{name}.tar.gz"
            assert sha(pack) == report[name]["sha256"]
            with tarfile.open(pack) as archive:
                for entry in archive:
                    if not entry.isfile() or Path(entry.name).name != entry.name:
                        raise ValueError(entry.name)
                    path = source / entry.name
                    if not path.exists():
                        tmp = path.with_suffix(".partial")
                        tmp.write_bytes(archive.extractfile(entry).read())
                        tmp.replace(path)
        meta = source / "metadata.csv"
        ready = read(source / "ready.json")
        assert ready["job"] == name and ready["metadata_sha256"] == sha(meta)
        assert sha(source / "source_provenance.json") == ready["provenance_sha256"]
        rows = list(csv.DictReader(meta.open()))
        assert len(rows) >= job["minimum_training_clips"]
        assert all(sha(source / row["video"]) == row["sha256"] for row in rows)
        f = prompts[name]
        texts = f["released"] + f["local_supplement"]
        assert len(set(texts)) == len(texts) == settings["prompts_per_task"] == 16
        assert sha(repo / f["public_file"]) == f["public_file_sha256"]
        caps = {row["prompt"].strip().casefold() for row in rows}
        assert not caps.intersection(t.strip().casefold() for t in texts), name
        motion = name == motion_id
        config = repo / "configs" / ("motion.yaml" if motion else "appearance.yaml")
        direction = None if motion else repo / job["training_guidance"]
        if direction:
            guidance = read(direction)
            assert all(guidance[k] for k in ("direction_prompts","neg_direction_prompts","base_prompts"))
        spec.append(dict(job=name, data=str(source), count=len(rows), metadata_sha256=sha(meta),
                         config=str(config), config_sha256=sha(config),
                         direction=str(direction) if direction else None,
                         direction_sha256=sha(direction) if direction else None,
                         released=len(f["released"]), supplemental=len(f["local_supplement"]),
                         physical_clips=80 if motion else 144))
    spec.sort(key=lambda item: item["job"] == motion_id)
    freeze(root / "jobs.json", json.dumps(spec, indent=2).encode())
    return spec


def train(item, out):
    name = item["job"]
    motion = name == motion_id
    dino = art / "table7_training_20261108/dependencies/dinov2"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,[deps/"appearance_overlay", deps/"python", repo])),
               BASELINE_BERT=str(deps/"bert-large-uncased"),
               DIFFSYNTH_MODEL_BASE_PATH=str(base/"models"), DIFFSYNTH_SKIP_DOWNLOAD="true",
               HF_HUB_OFFLINE="1", HF_HOME=str(base/"cache/huggingface"), TORCH_HOME=str(base/"cache/torch"),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", WAN_DINO_LOCAL=str(dino))
    if motion:
        assert (dino / "hubconf.py").is_file() and (base/"cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth").is_file()
    idle = {}
    deadline = time.monotonic() + 48*3600
    while time.monotonic() < deadline:
        try:
            cards, apps = guard.snapshot()
            now = time.monotonic()
            idle = {g:idle.get(g,now) for g,(f,u,t) in cards.items() if f>120000 and u<512 and t<5 and not apps[g] and guard.host_ram()>=20480}
        except (OSError,ValueError,KeyError,subprocess.SubprocessError):
            idle.clear()
        for g,since in sorted(idle.items()):
            if time.monotonic()-since < 120:
                continue
            with (art / f"t2v_gpu_{g}.lock").open("a") as lease:
                try:
                    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                cards,apps=guard.snapshot()
                f,u,t=cards[g]
                if apps[g] or f<=120000 or u>=512 or t>=5 or guard.host_ram()<20480:
                    continue
                work=out/f"train_{time.time_ns()}"
                work.mkdir()
                cmd=[python,str(root/"motion_entry.py" if motion else art/"public_person_baseline_20261120/train_entry.py"),
                     "--config",item["config"],"--dataset_base_path",item["data"],
                     "--dataset_metadata_path",str(Path(item["data"])/"metadata.csv"),"--output_path",str(work)]
                if not motion:
                    cmd += ["--direction",item["direction"],"--internvideo_pretrained_path",str(deps/"InternVideo2-stage2_1b-224p-f4.pt")]
                save(work/"invocation.json",dict(argv=cmd,**item))
                with (work/"train.log").open("a") as log:
                    child=subprocess.Popen(cmd,cwd=repo,env=dict(env,CUDA_VISIBLE_DEVICES=g),stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                    try:
                        while child.poll() is None:
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                if guard.snapshot()[1][g]-{child.pid}:
                                    break
                        code=child.poll()
                    finally:
                        guard.kill_ours(child)
                if code is None:
                    idle.clear()
                    break
                ck=work/"epoch-1.safetensors"
                if code or not ck.is_file():
                    raise RuntimeError(f"Native training exit {code}, see {work}")
                save(out/"trained.json",{"weights":{"native":{"path":str(ck),"sha256":sha(ck)}}})
                return
        time.sleep(10)
    raise TimeoutError("48 hours waiting for exclusive GPU")


def main():
    with (root/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=prepare()
        states={j["job"]:"queued_for_training" for j in jobs}
        save(root/"queue_status.json",states)
        for item in jobs:
            name=item["job"]
            out=root/name
            out.mkdir(exist_ok=True)
            def state(s):
                states[name]=s
                save(root/"queue_status.json",states)
            try:
                if not (out/"trained.json").exists():
                    state("waiting_or_training_exclusive_gpu")
                    train(item,out)
                ck=read(out/"trained.json")["weights"]["native"]
                assert sha(Path(ck["path"])) == ck["sha256"]
                result=out/"native"
                status=result/"evaluation_status.json"
                if not status.exists() or read(status)["status"]!="evaluated":
                    state("queued_for_generation")
                    with (out/"generation_scheduler.log").open("a") as log:
                        subprocess.run([python,str(root/"evaluation_scheduler.py"),"--job",name,"--checkpoint",ck["path"],
                                        "--out",str(result),"--gpu-free","120000"],cwd=root,stdin=subprocess.DEVNULL,
                                        stdout=log,stderr=subprocess.STDOUT,check=True)
                if read(status)["status"]!="evaluated" or not read(result/"alignment/alignment.json")["passed"]:
                    raise RuntimeError(f"Evaluation status {read(status)}")
                state("evaluated")
            except Exception as error:
                state("needs_attention: "+str(error))
                print(name,repr(error),flush=True)

if __name__=="__main__":
    main()

