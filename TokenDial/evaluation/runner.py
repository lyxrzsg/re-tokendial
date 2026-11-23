import csv
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from .author import load_author
from .manifest import load_manifest, load_guidance, sha256
from .metrics import METRICS, calculate, summarize


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(args):
    import torch
    import lpips
    from transformers import CLIPModel, CLIPProcessor
    data = load_manifest(args.manifest, probe=True)
    # Fail on missing guidance before loading models or writing partial metrics.
    guidance = [load_guidance(args.guidance_dir, s) for s in data['samples']]
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}; use a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    author = load_author()
    device = author.resolve_device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=False).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {}
    for name in ['torch', 'torchvision', 'transformers', 'numpy', 'lpips', 'decord', 'opencv-python', 'Pillow']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata = {'status': 'running', 'manifest_sha256': sha256(args.manifest), 'git_commit': commit,
                'author_source_sha256': sha256(Path(__file__).parent / 'vendor/evaluate_sliders.py'),
                'arguments': {k: v for k, v in vars(args).items() if k != 'func'},
                'packages': packages, 'device': str(device),
                'clip_weight_commit': getattr(model.config, '_commit_hash', None),
                'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                'baseline_note': 'LPIPS and CR compare against explicitly designated first clip, not automatically against raw source video',
                'guidance_sha256': {str(p): sha256(p) for _, p in guidance}}
    write_json(out / 'run.json', metadata)
    rows = []
    fields = ['method', 'concept', 'attribute', 'sample', 'baseline_scale', 'last_scale', 'total_scales',
              'scales_present', 'anchor_prompt', *METRICS, 'overall_per_sample']
    try:
        with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f, (out / 'features.jsonl').open('w', encoding='utf-8') as ff:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s, (g, _) in zip(data['samples'], guidance):
                start = time.monotonic()
                scales = [str(i + 1) for i in range(len(s['clips']))]
                specs = author.default_concept_specs()
                word = specs[s['concept']].concept_word if s['concept'] in specs else s['concept'].replace('_', ' ')
                anchor_prompt = author.build_anchor_prompt(word, s['attribute'], len(scales) - 1, len(scales))
                base = author.read_video_frames(s['clips'][0]['video'], args.stride, args.max_frames)
                if not base:
                    raise ValueError('Baseline video decoded zero frames')
                anchors, positives, negatives, distances = [], [], [], []
                for clip in s['clips']:
                    frames = author.read_video_frames(clip['video'], args.stride, args.max_frames)
                    if len(frames) != len(base):
                        raise ValueError(f'{s["sample"]}: unaligned sampled frame counts')
                    anchors.append(author.clip_mean_score_for_frames(frames, anchor_prompt, model, processor, device, args.clip_batch_size))
                    positives.append(author.clip_mean_score_for_frames_multi_prompt(frames, g['direction_prompts'], model, processor, device, args.clip_batch_size))
                    negatives.append(author.clip_mean_score_for_frames_multi_prompt(frames, g['neg_direction_prompts'], model, processor, device, args.clip_batch_size))
                    distances.append(author.lpips_mean_for_frame_pairs(base, frames, lpips_model, device, args.lpips_batch_size))
                row = {k: s[k] for k in ['method', 'concept', 'attribute', 'sample']}
                row.update(baseline_scale=scales[0], last_scale=scales[-1], total_scales=len(scales),
                           scales_present=','.join(scales), anchor_prompt=anchor_prompt)
                row.update(calculate(anchors, positives, negatives, distances, args.monotonicity_eps))
                writer.writerow(row)
                f.flush()
                ff.write(json.dumps({'sample': s, 'anchor_prompt': anchor_prompt, 'anchor': anchors,
                                     'positive': positives, 'negative': negatives, 'lpips': distances,
                                     'video_sha256': [sha256(c['video']) for c in s['clips']]}, ensure_ascii=False, allow_nan=False) + '\n')
                ff.flush()
                rows.append(row)
                print(f'Evaluated {s["sample"]}: {time.monotonic() - start:.1f}s', flush=True)
        write_json(out / 'summary.json', summarize(rows))
        metadata.update(status='complete', evaluated_samples=len(rows))
    except Exception as exc:
        metadata.update(status='failed', evaluated_samples=len(rows), error=str(exc))
        raise
    finally:
        write_json(out / 'run.json', metadata)
