"""Optional online scorer using the unchanged supplied Gemini evaluator."""
import os
from pathlib import Path
import subprocess
import sys

from .manifest import load_manifest
from .reference import materialize
from .runner import write_json


def run_gemini(args):
    data = load_manifest(args.manifest, probe=True)
    if any(len(s['clips']) != 5 for s in data['samples']):
        raise ValueError('The supplied VLM rubric requires exactly five levels per sample')
    if not os.environ.get('GEMINI_API_KEY'):
        raise ValueError('Set GEMINI_API_KEY in the environment; never put keys in a manifest')
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'VLM output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    view = materialize(data, out / 'author_view')
    write_json(out / 'protocol.json', {'model': args.model, 'num_frames': args.num_frames,
               'rubric': 'same-time cross-strength progression, not temporal continuity',
               'online_validated': False, 'status': 'running'})
    subprocess.run([sys.executable, str(Path(__file__).parent / 'vendor/gemini_slider_video_eval.py'),
                    '--root', str(view), '--out_csv', str(out / 'scores.csv'),
                    '--out_jsonl', str(out / 'scores.jsonl'), '--model', args.model,
                    '--num_frames', str(args.num_frames)], check=True)
    # Verify the expected sample coverage, rather than treating exit=0 as success.
    import csv
    with (out / 'scores.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    actual = [(r['method'], r['concept'], r['attribute'], Path(r['filename']).stem) for r in rows]
    expected = [(s['method'], s['concept'], s['attribute'], s['sample']) for s in data['samples']]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('VLM output coverage differs from manifest')
    write_json(out / 'protocol.json', {'model': args.model, 'num_frames': args.num_frames,
               'rubric': 'same-time cross-strength progression, not temporal continuity',
               'online_validated': True, 'status': 'complete', 'sample_count': len(rows)})
