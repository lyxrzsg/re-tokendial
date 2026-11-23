"""Map TokenDial filenames to a portable, explicit evaluation manifest."""
import json
from pathlib import Path
import subprocess

from .manifest import load_manifest, sha256


def adapt(args):
    scales = [float(x) for x in args.model_scales.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    if len(scales) < 2 or scales[0] != args.baseline_scale or len(set(scales)) != len(scales):
        raise ValueError('Provide unique ordered model scales with the explicit baseline first')
    output = Path(args.manifest).resolve()
    if output.exists():
        raise ValueError(f'Manifest already exists: {output}')
    root = Path(args.root).resolve()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    data = {'schema_version': 1, 'generation': {'repository_commit': revision,
            'kind': args.kind, 'notes': args.notes, 'model_scales': scales}, 'samples': []}
    if args.checkpoint:
        data['generation']['checkpoint'] = str(Path(args.checkpoint).resolve())
        data['generation']['checkpoint_sha256'] = sha256(args.checkpoint)
    for seed in seeds:
        folder = root / f'seed_{seed}'
        clips = []
        for rank, scale in enumerate(scales):
            filename = f'flowedit_rgb{scale}.mp4' if args.kind == 'v2v' else f'cfg_{scale}.mp4'
            video = folder / filename
            if not video.is_file():
                raise ValueError(f'Missing expected TokenDial output: {video}')
            # Relative paths preserve portability when the whole project is copied.
            import os
            clips.append({'intensity': rank, 'model_scale': scale, 'is_baseline': rank == 0,
                          'video': os.path.relpath(video, output.parent)})
        sample = {'method': args.method, 'concept': args.concept, 'attribute': args.attribute,
                  'sample': f'seed_{seed}', 'seed': seed, 'prompt': args.prompt,
                  'baseline_kind': 'zero_strength_reconstruction' if args.kind == 'v2v' else 'zero_strength_generation',
                  'clips': clips}
        if args.baseline_scale != 0:
            sample['baseline_kind'] = 'explicit_nonzero_reference'
        if args.source_video:
            import os
            sample['source_video'] = os.path.relpath(Path(args.source_video).resolve(), output.parent)
        data['samples'].append(sample)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    try:
        load_manifest(output, probe=True)
    except Exception:
        output.unlink()
        raise
    print(f'Created manifest: {output}')
