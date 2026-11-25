"""Run the matched 24-update person_older training with resumable checkpoints."""
import csv
import json
import math
import os
from pathlib import Path
import runpy
import statistics

import torch
from accelerate.utils import set_seed
from tqdm import tqdm

import diffsynth.diffusion as diffusion
from feature_losses.internvideo2.configs.model import TextEncoders

SEED = 20261123
MATCHED_EPOCHS = {"single": 12, "multi": 3}

REL_TOL = 0.005
ABS_TOL = 0.002


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def train(accelerator, dataset, model, model_logger, args=None, **kwargs):
    arm = os.environ["TOKENDIAL_PERSON_ARM"]
    epochs = MATCHED_EPOCHS[arm]
    expected = {"single": (1, 2), "multi": (8, 1)}[arm]
    if accelerator.num_processes != 1 or args.learning_rate != 5e-4:
        raise ValueError("The matched study uses one process and learning rate 5e-4")
    out = Path(model_logger.output_path)
    out.mkdir(exist_ok=True, parents=True)
    snapshots = out / "checkpoints"
    snapshots.mkdir(exist_ok=True)
    model_logger.output_path = str(snapshots)
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda sample: sample[0],
                                        num_workers=args.dataset_num_workers)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    token = accelerator.unwrap_model(model).pipe.dit.rgb_token
    state_file = out / "resume.pt"
    history = []
    best = math.inf
    stale = 0
    if state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        with torch.no_grad():
            token.copy_(state["token"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["gpu_rng"])
        history, best, stale = state["history"], state["best"], state["stale"]
        if state["stop"]:
            save_json(out/"loss_curve.json", history)
            save_json(out/"completed.json", dict(epochs=len(history),best_loss=best,updates=sum(row["updates"] for row in history), arm=arm))
            return
    step_state_file = out / "step_resume.pt"
    step_state = None
    if step_state_file.exists():
        candidate = torch.load(step_state_file, map_location="cpu", weights_only=True)
        if candidate["epoch"] == len(history):
            step_state = candidate
            with torch.no_grad():
                token.copy_(step_state["token"])
            optimizer.load_state_dict(step_state["optimizer"])
            scheduler.load_state_dict(step_state["scheduler"])
    assert accelerator.gradient_accumulation_steps == 1
    # Probe the identical clip/noise pairs at each epoch. These are training clips.
    unique = len(dataset) // args.dataset_repeat
    if (unique, args.dataset_repeat) != expected or len(loader) * epochs != 24:
        raise ValueError("Unexpected dataset size or repetition count")
    indices = [0]
    def fixed_probe():
        probes = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for i, index in enumerate(indices):
                    for repeat in range(2):
                        torch.manual_seed(SEED+1000+2*i+repeat)
                        data = dataset[index]
                        value = float(model(data).detach().float())
                        if not math.isfinite(value):
                            raise RuntimeError(f"Non-finite fixed loss probe at index {index}")
                        probes.append(value)
        finally:
            model.train(was_training)
        return probes

    if not history and not (out/"initial_probe.json").exists():
        save_json(out/"initial_probe.json", fixed_probe())
    progress = out / "epoch_progress.json"
    with (out / "step_loss.csv").open("a", newline="", buffering=1) as log:
        writer = csv.writer(log)
        if log.tell() == 0:
            writer.writerow(["attempt", "epoch", "step_in_epoch", "loss", "learning_rate", "rgb_token_grad_norm"])
        while True:
            epoch = len(history)
            losses = []
            if step_state is None:
                sampler_cpu_rng = torch.get_rng_state()
                sampler_gpu_rng = torch.cuda.get_rng_state()
                iterator = iter(loader)
            else:
                losses = step_state["losses"]
                sampler_cpu_rng, sampler_gpu_rng = step_state["sampler_cpu_rng"], step_state["sampler_gpu_rng"]
                # Recreate the same shuffled iterator/worker seed, then replay only data loading.
                torch.set_rng_state(sampler_cpu_rng)
                torch.cuda.set_rng_state(sampler_gpu_rng)
                iterator = iter(loader)
                for _ in losses:
                    next(iterator)
                torch.set_rng_state(step_state["cpu_rng"])
                torch.cuda.set_rng_state(step_state["gpu_rng"])
                step_state = None
            start = len(losses)
            for step, data in enumerate(tqdm(iterator, total=len(loader), initial=start, desc=f"Epoch {epoch+1}"), start+1):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                    value = float(loss.detach().float())
                    if not math.isfinite(value):
                        raise RuntimeError(f"Non-finite training loss at epoch {epoch+1}, step {step}")
                    lr = optimizer.param_groups[0]["lr"]
                    accelerator.backward(loss)
                    grad_norm = float(token.grad.detach().float().norm())
                    if not math.isfinite(grad_norm):
                        raise RuntimeError("Non-finite rgb_token gradient")
                    optimizer.step()
                    scheduler.step()
                losses.append(value)
                writer.writerow([os.environ.get("PLATEAU_ATTEMPT", "manual"),epoch+1,step,value,lr,grad_norm])
                if step % 1 == 0:
                    save_json(progress, dict(epoch=epoch+1,step=step,total=len(loader),window_loss=statistics.mean(losses[-50:])))
                    tmp = step_state_file.with_suffix(".tmp")
                    torch.save(dict(epoch=epoch, losses=losses, token=token.detach().cpu(),
                                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                    cpu_rng=torch.get_rng_state(), gpu_rng=torch.cuda.get_rng_state(),
                                    sampler_cpu_rng=sampler_cpu_rng, sampler_gpu_rng=sampler_gpu_rng), tmp)
                    tmp.replace(step_state_file)
            probes = fixed_probe()
            probe = statistics.mean(probes)
            if not math.isfinite(best) or best - probe >= max(ABS_TOL, REL_TOL*abs(best)):
                best, stale = probe, 0
            else:
                stale += 1
            model_logger.on_epoch_end(accelerator,model,epoch)
            history.append(dict(epoch=epoch+1,updates=len(losses),train_loss_mean=statistics.mean(losses),
                                fixed_probe_mean=probe,fixed_probe_values=probes,stale_epochs=stale,
                                token_rms=float(token.detach().float().square().mean().sqrt())))
            stop = len(history) >= epochs
            tmp = state_file.with_suffix(".tmp")
            torch.save(dict(token=token.detach().cpu(),optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),cpu_rng=torch.get_rng_state(),
                            gpu_rng=torch.cuda.get_rng_state(),history=history,best=best,stale=stale,stop=stop),tmp)
            tmp.replace(state_file)
            save_json(out/"loss_curve.json",history)
            print(json.dumps(history[-1]),flush=True)
            if stop:
                save_json(out/"completed.json",dict(epochs=len(history),best_loss=best,updates=sum(row["updates"] for row in history), arm=arm))
                return


TextEncoders["bert_large"]["pretrained"] = os.environ["BASELINE_BERT"]
set_seed(SEED)
diffusion.launch_training_task = train
runpy.run_module("tokendial.train",run_name="__main__")


