"""Run the untouched author CLI on a symlink view and compare every sample."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from .manifest import load_manifest, sample_key, sha256, load_guidance
from .metrics import METRICS
from .runner import write_json


def materialize(data, root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Reference view already exists and is not empty: {root}')
    signatures = {}
    for s in data['samples']:
        group = tuple(s[k] for k in ['method', 'concept', 'attribute'])
        levels = tuple(c['intensity'] for c in s['clips'])
        if group in signatures and signatures[group] != levels:
            raise ValueError('Author directory layout requires identical levels within each task/method')
        signatures[group] = levels
        for rank, c in enumerate(s['clips'], 1):
            dest = root / s['concept'] / s['attribute'] / s['method'] / str(rank) / (s['sample'] + '.mp4')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(Path(c['video']).resolve())
    return root


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        key = sample_key(row)
        if key in result:
            raise ValueError(f'Duplicate output row: {key}')
        result[key] = row
    return result


def compare(candidate, reference_rows, expected_keys, atol=1e-5, rtol=1e-5):
    expected_keys = set(expected_keys)
    missing = {'candidate_missing': sorted(expected_keys - set(candidate)),
               'reference_missing': sorted(expected_keys - set(reference_rows)),
               'candidate_extra': sorted(set(candidate) - expected_keys),
               'reference_extra': sorted(set(reference_rows) - expected_keys)}
    passed = not any(missing.values())
    differences = []
    maxima = {k: 0.0 for k in METRICS}
    for key in sorted(set(candidate) & set(reference_rows) & expected_keys):
        a, b = candidate[key], reference_rows[key]
        for field in ['anchor_prompt', 'baseline_scale', 'last_scale', 'total_scales', 'scales_present']:
            if str(a[field]) != str(b[field]):
                passed = False
                differences.append({'sample': key, 'field': field, 'candidate': a[field], 'reference': b[field]})
        for metric in METRICS:
            av, bv = float(a[metric]), float(b[metric])
            finite = math.isfinite(av) and math.isfinite(bv)
            delta = abs(av - bv) if finite else None
            if finite:
                maxima[metric] = max(maxima[metric], delta)
            if not finite or not math.isclose(av, bv, abs_tol=atol, rel_tol=rtol):
                passed = False
                differences.append({'sample': key, 'metric': metric, 'candidate': av if math.isfinite(av) else None,
                                    'reference': bv if math.isfinite(bv) else None, 'absolute_error': delta})
    return {'passed': passed, 'expected_samples': len(expected_keys), 'candidate_samples': len(candidate),
            'reference_samples': len(reference_rows), 'atol': atol, 'rtol': rtol,
            'max_absolute_error': maxima, 'coverage': missing, 'differences': differences,
            'scope': 'Shared unmodified low-level author frame/model helpers; independent manifest traversal and metric reduction; not an independent reimplementation of encoders'}


def align(args):
    data = load_manifest(args.manifest, probe=True)
    run_path = Path(args.metrics_csv).parent / 'run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    if run.get('status') != 'complete' or run.get('manifest_sha256') != sha256(args.manifest):
        raise ValueError('Candidate run must be complete and use the identical manifest')
    for name in ['stride', 'max_frames', 'clip_batch_size', 'lpips_batch_size', 'monotonicity_eps', 'clip_model']:
        if run['arguments'].get(name) != getattr(args, name):
            raise ValueError(f'Candidate/reference configuration mismatch: {name}')
    vendor = Path(__file__).parent / 'vendor/evaluate_sliders.py'
    if run.get('author_source_sha256') != sha256(vendor):
        raise ValueError('Author helper source changed since candidate run')
    for sample in data['samples']:
        _, path = load_guidance(args.guidance_dir, sample)
        if run['guidance_sha256'].get(str(path)) != sha256(path):
            raise ValueError(f'Guidance changed since candidate run: {path}')
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Alignment directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    view = materialize(data, out / 'author_view')
    reference_rows = {}
    for i, method in enumerate(sorted({s['method'] for s in data['samples']})):
        csv_path = out / f'author_{i}.csv'
        command = [sys.executable, str(Path(__file__).parent / 'vendor/evaluate_sliders.py'),
                   '--root', str(view), '--method', method, '--guidance_dir', args.guidance_dir,
                   '--out_csv', str(csv_path), '--stride', str(args.stride), '--max_frames', str(args.max_frames),
                   '--device', args.device, '--clip_batch_size', str(args.clip_batch_size),
                   '--lpips_batch_size', str(args.lpips_batch_size), '--monotonicity_eps', str(args.monotonicity_eps)]
        with (out / f'author_{i}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        rows = read_rows(csv_path)
        if set(rows) & set(reference_rows):
            raise ValueError('Repeated reference keys across methods')
        reference_rows.update(rows)
    report = compare(read_rows(args.metrics_csv), reference_rows, [sample_key(s) for s in data['samples']], args.atol, args.rtol)
    report['configuration_verified'] = True
    write_json(out / 'alignment.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)
