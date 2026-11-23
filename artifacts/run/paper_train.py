"""Paper-parameter training adapter for the pinned public TokenDial Wan implementation."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def read_dataset(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Training dataset is empty")
    paths = []
    for row in rows:
        file = (path.parent / row["video"]).resolve()
        if not file.is_file() or not row["prompt"].strip():
            raise ValueError(f"Invalid training row: {row}")
        if row.get("sha256") and sha256(file) != row["sha256"]:
            raise ValueError(f"Training clip hash mismatch: {file}")
        paths.append(file)
    if len(set(paths)) != len(paths):
        raise ValueError("Training paths must be unique")
    return rows


def update_indices(size, steps, seed):
    rng = random.Random(seed)
    indices = []
    while len(indices) < steps:
        epoch = list(range(size))
        rng.shuffle(epoch)
        indices.extend(epoch)
    return indices[:steps]


def paper_loss(pipe, inputs, settings, motion, perceptual):
    import torch
    from torch.utils.checkpoint import checkpoint

    base = inputs["input_latents"]
    index = int(torch.randint(len(pipe.scheduler.sigmas), ()).item())
    sigma = float(pipe.scheduler.sigmas[index])
    noise = torch.randn_like(base)
    noisy = ((1 - sigma) * base.float() + sigma * noise.float()).to(base.dtype)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    model_inputs = {key: value for key, value in inputs.items() if key != "latents"}
    dit = pipe.dit

    def predict(x, level, grad):
        timestep = torch.tensor([level * 1000], device=x.device, dtype=pipe.torch_dtype)
        return pipe.model_fn(
            **models, **dict(model_inputs, use_gradient_checkpointing=grad),
            latents=x.to(base.dtype), timestep=timestep, rgb_skip_first_latent=motion,
        ).float()

    def branch(enabled):
        dit.use_rgb_token = enabled
        with torch.set_grad_enabled(enabled):
            first_velocity = predict(noisy, sigma, enabled)
            initial = noisy.float() - sigma * first_velocity
        count = settings["posterior_additional_steps"]
        if sigma < settings["posterior_sigma_threshold"] or count == 0:
            return initial
        with torch.no_grad():
            value = noisy.float()
            velocity = first_velocity.detach()
            current = sigma
            for i in range(1, count + 1):
                following = sigma * (count + 1 - i) / (count + 1)
                value = value + (following - current) * velocity
                current = following
                velocity = predict(value, current, False)
            refined = value - current * velocity
        return refined + (initial - initial.detach()) if enabled else refined

    original_flag = dit.use_rgb_token
    try:
        target = branch(False)
        edited = branch(True)
    finally:
        dit.use_rgb_token = original_flag

    vae = pipe.vae
    def decode(latents):
        pixels = vae.model.decode(latents.to(pipe.torch_dtype), vae.scale)
        return pixels[:, :, :settings["supervision_frames"]].clamp(-1, 1)

    with torch.no_grad():
        target_pixels = decode(target)
    edited_pixels = checkpoint(decode, edited, use_reentrant=False)
    if edited_pixels.shape[2] != settings["supervision_frames"]:
        raise RuntimeError(f"Unexpected decoded training length: {edited_pixels.shape}")
    if motion:
        dino = pipe.dino_speed_loss_fn
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attribute = dino.calculate_fast_motion_loss_scale(
                edited_pixels, stride=1, num_keyframes=settings["supervision_frames"],
                fast_ratio=settings["motion_gamma"],
            ).mean()
            preservation = dino(edited_pixels, target_pixels).mean()
        weight = settings["motion_firstframe_weight"]
    else:
        attribute = pipe.internvideo_loss_fn(
            pred_pixel=edited_pixels, target_pixel=target_pixels,
        ).mean()

        def frames(pixels):
            return pixels.permute(0, 2, 1, 3, 4).flatten(0, 1).float()

        edit_frames, target_frames = frames(edited_pixels), frames(target_pixels)
        preservation = torch.cat([
            checkpoint(perceptual, part, ref, use_reentrant=False).flatten()
            for part, ref in zip(edit_frames.split(4), target_frames.split(4))
        ]).mean()
        weight = settings["appearance_lpips_weight"]
    total = attribute.float() + weight * preservation.float()
    return total, float(attribute.detach()), float(preservation.detach()), sigma


def save_atomic(save, value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    save(value, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dino-repo", type=Path)
    p.add_argument("--internvideo", type=Path)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--steps", type=int)
    args = p.parse_args()

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from diffsynth.core import UnifiedDataset
    from tokendial.train import WanTrainingModule

    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    job = next(j for j in plan["jobs"] if j["id"] == args.job)
    settings = plan["training"]
    steps = args.steps if args.steps is not None else settings["optimizer_updates"]
    if not 1 <= steps <= settings["optimizer_updates"] or (not args.pilot and steps != settings["optimizer_updates"]):
        p.error("Only an explicitly labelled pilot may override optimizer updates")
    rows = read_dataset(args.dataset)
    if not args.pilot and len(rows) < job["minimum_training_clips"]:
        p.error("Insufficient independent training clips for the formal run")

    motion = job["concept"] == "speed"
    if motion:
        if not args.dino_repo or not (args.dino_repo / "hubconf.py").is_file():
            p.error("A pinned local DINOv2 repository is required")
        original_hub_load = torch.hub.load
        def local_dino(repo, name, *pos, **kw):
            if repo == "facebookresearch/dinov2":
                return original_hub_load(str(args.dino_repo), name, *pos, source="local", **kw)
            return original_hub_load(repo, name, *pos, **kw)
        torch.hub.load = local_dino
    elif not args.internvideo or not args.internvideo.is_file():
        p.error("The official InternVideo2 weight file is required")

    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    paths = "Wan-AI/Wan2.1-T2V-1.3B:"
    model = WanTrainingModule(
        model_id_with_origin_paths=",".join(paths + name for name in (
            "diffusion_pytorch_model*.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth")),
        device="cuda", task="tokendial:motion" if motion else "tokendial:appearance",
        internvideo_pretrained_path=str(args.internvideo or ""),
        internvideo_guidance_concept=job["concept"],
        internvideo_prompts_path=job.get("training_guidance"),
        dino_speed_fast_ratio=settings["motion_gamma"],
        dino_speed_num_keyframes=settings["supervision_frames"],
    )
    pipe = model.pipe
    model.requires_grad_(False)
    pipe.dit.rgb_token = torch.nn.Parameter(pipe.dit.rgb_token.detach().float())
    token = pipe.dit.rgb_token
    pipe.dit.use_rgb_token = True
    pipe.dit.rgb_token_weight = 1.0
    if motion:
        perceptual = None
    else:
        import lpips
        perceptual = lpips.LPIPS(net="alex").eval().requires_grad_(False).to("cuda")
    active = [name for name, param in model.named_parameters() if param.requires_grad]
    if active != ["pipe.dit.rgb_token"]:
        raise RuntimeError(f"Unexpected trainable parameters: {active}")
    operator = UnifiedDataset.default_video_operator(
        base_path=str(args.dataset.parent), height=settings["height"], width=settings["width"],
        max_pixels=settings["height"] * settings["width"], height_division_factor=16,
        width_division_factor=16, num_frames=settings["wan_input_frames"],
        time_division_factor=4, time_division_remainder=1,
    )
    optimizer = torch.optim.AdamW([token], lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    args.out.mkdir(parents=True, exist_ok=True)
    resume = args.out / "resume.pt"
    identity = {
        "plan": sha256(args.plan), "dataset": sha256(args.dataset), "code": sha256(Path(__file__)),
        "steps": steps, "job": args.job, "pilot": args.pilot,
    }
    start = 0
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume inputs or training version differ")
        token.data.copy_(state["rgb_token"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start = state["step"]
    choices = update_indices(len(rows), steps, seed)
    for step in range(start, steps):
        row = rows[choices[step]]
        frames = operator(row["video"])
        if len(frames) != settings["wan_input_frames"]:
            raise ValueError(f"Unexpected input length: {row['video']}")
        with torch.no_grad():
            shared, positive, negative = model.transfer_data_to_device(
                model.get_pipeline_inputs({"video": frames, "prompt": row["prompt"]}),
                pipe.device, pipe.torch_dtype,
            )
            for unit in pipe.units:
                shared, positive, negative = pipe.unit_runner(unit, pipe, shared, positive, negative)
        optimizer.zero_grad(set_to_none=True)
        total, attribute, preservation, sigma = paper_loss(
            pipe, dict(shared, **positive), settings, motion, perceptual,
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite loss at step {step + 1}")
        total.backward()
        grad = token.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"Invalid offset gradient at step {step + 1}")
        grad_norm = float(grad.norm())
        optimizer.step()
        print(json.dumps(dict(step=step + 1, loss=float(total.detach()), attribute=attribute,
                              preservation=preservation, sigma=sigma, offset_grad_norm=grad_norm,
                              offset_norm=float(token.detach().norm()), source=row["video"])), flush=True)
        save_atomic(torch.save, dict(
            identity=identity, step=step + 1, rgb_token=token.detach().cpu(),
            optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
        ), resume)

    if not torch.isfinite(token).all() or not torch.count_nonzero(token):
        raise RuntimeError("Trained offset is zero or nonfinite")
    checkpoint_path = args.out / f"step-{steps}.safetensors"
    save_atomic(save_file, {"rgb_token": token.detach().cpu().contiguous()}, checkpoint_path)
    save_atomic(lambda data, path: path.write_text(json.dumps(data, indent=2), encoding="utf-8"),
                dict(identity, checkpoint_sha256=sha256(checkpoint_path)), args.out / "complete.json")


if __name__ == "__main__":
    main()

