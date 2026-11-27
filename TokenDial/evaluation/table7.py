"""Table 7 task inventory and paired five-level T2V evaluation coverage."""
from pathlib import Path

from .manifest import load_guidance, load_manifest

ATTRIBUTES = {
    "aurora": "brighter dimmer larger purple smaller",
    "bubble": "burst denser larger smaller sparser",
    "campfire": "bluer brighter larger smaller warmer",
    "confetti": "chaotic color_saturation denser sparser stable",
    "explosion": "brighter dimmer larger smaller smoky",
    "ink": "dilution greener redder spread thick",
    "person": "curly_hair happy heavier older walk_to_run",
    "smoke": "calm darker thicker thinner turbulence",
    "snowflake": "brighter denser larger_particle small_particle sparser",
    "spark": "brighter chaotic denser sparser stable",
    "water_splash": "chaotic less_droplet more_droplet stable viscosity",
    "speed": "faster slower",
}


def inspect(args):
    tasks = []
    expected = set()
    for concept, attributes in ATTRIBUTES.items():
        for attribute in attributes.split():
            group = "person_walk" if (concept, attribute) == ("person", "walk_to_run") else concept
            prompt_group = "fire_explosion" if concept == "explosion" else group
            path = Path(args.prompts_dir) / (prompt_group + ".txt")
            prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(set(prompts)) != 8 or len(prompts) != 8:
                raise ValueError(f"{path}: expected 8 distinct prompts")
            try:
                _, guidance = load_guidance(args.guidance_dir, {"concept": group, "attribute": attribute})
            except FileNotFoundError:
                guidance = None
            tasks.append(dict(concept=concept, attribute=attribute, manifest_concept=group,
                              prompt_file=path.name, guidance=str(guidance) if guidance else None))
            for method in ("nomask", "mask"):
                for prompt in prompts:
                    for seed in args.seeds:
                        expected.add((method, group, attribute, prompt, seed))
    missing_guidance = [f"{t['concept']}/{t['attribute']}" for t in tasks if not t["guidance"]]
    report = dict(tasks=tasks, attributes=len(tasks), seeds=args.seeds,
                  missing_guidance=missing_guidance, expected_trajectories=len(expected),
                  expected_level_entries=5 * len(expected))
    if args.manifest:
        if missing_guidance:
            raise ValueError(f"Missing scoring guidance: {missing_guidance}")
        observed = set()
        scales_by_task = {}
        for sample in load_manifest(args.manifest)["samples"]:
            key = tuple(sample.get(k) for k in ("method", "concept", "attribute", "prompt", "seed"))
            if type(key[-1]) is not int or key not in expected or key in observed:
                raise ValueError(f"Unexpected or duplicate method/task/prompt/seed: {key}")
            observed.add(key)
            clips = sample["clips"]
            scales = tuple(c.get("model_scale") for c in clips)
            if ([c["intensity"] for c in clips] != list(range(5)) or scales[0] != 0
                    or None in scales or len(set(scales)) != 5):
                raise ValueError(f"{key}: expected logical levels 0..4 and five distinct actual scales starting at 0")
            if scales_by_task.setdefault(key[1:3], scales) != scales:
                raise ValueError(f"{key}: mismatched actual scale schedule within attribute")
        if expected != observed:
            missing = expected - observed
            raise ValueError(f"{len(missing)} missing trajectories; first: {min(missing)}")
        report["validated_trajectories"] = len(observed)
    return report

