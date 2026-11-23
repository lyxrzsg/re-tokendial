import argparse
import json
from .manifest import load_manifest, load_guidance


def common(p):
    p.set_defaults(clip_model='openai/clip-vit-base-patch32')
    p.add_argument('--manifest', required=True)
    p.add_argument('--guidance-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=16)
    p.add_argument('--clip-batch-size', type=int, default=8)
    p.add_argument('--lpips-batch-size', type=int, default=8)
    p.add_argument('--monotonicity-eps', type=float, default=1e-4)


def main():
    p = argparse.ArgumentParser(description='TokenDial evaluation and author-code alignment')
    sub = p.add_subparsers(dest='command', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--manifest', required=True)
    v.add_argument('--guidance-dir', required=True)
    v.add_argument('--probe', action='store_true')
    t = sub.add_parser('adapt', help='Import seed_N directories produced by TokenDial')
    t.add_argument('--root', required=True)
    t.add_argument('--manifest', required=True)
    t.add_argument('--seeds', default='0,1')
    t.add_argument('--model-scales', default='0,1,2,3,4', help='In desired attribute progression order, baseline first')
    t.add_argument('--baseline-scale', type=float, default=0)
    t.add_argument('--kind', choices=['v2v', 't2v'], default='v2v')
    t.add_argument('--method', default='tokendial')
    t.add_argument('--concept', required=True)
    t.add_argument('--attribute', required=True)
    t.add_argument('--prompt', required=True)
    t.add_argument('--checkpoint')
    t.add_argument('--source-video')
    t.add_argument('--notes', default='')
    common(sub.add_parser('evaluate'))
    a = sub.add_parser('align')
    common(a)
    a.add_argument('--metrics-csv', required=True)
    a.add_argument('--atol', type=float, default=1e-5)
    a.add_argument('--rtol', type=float, default=1e-5)
    g = sub.add_parser('gemini')
    g.add_argument('--manifest', required=True)
    g.add_argument('--out-dir', required=True)
    g.add_argument('--model', required=True, help='Explicit model ID; no unverified historical default')
    g.add_argument('--num-frames', type=int, default=3)
    args = p.parse_args()
    for k in ['stride', 'clip_batch_size', 'lpips_batch_size', 'num_frames']:
        if hasattr(args, k) and getattr(args, k) <= 0:
            p.error(f'{k} must be positive')
    if hasattr(args, 'max_frames') and args.max_frames < 0:
        p.error('max_frames must be >=0')
    if args.command == 'adapt':
        from .adapt import adapt
        adapt(args)
    elif args.command == 'validate':
        data = load_manifest(args.manifest, args.probe)
        for s in data['samples']:
            load_guidance(args.guidance_dir, s)
        print(json.dumps({'valid': True, 'samples': len(data['samples']), 'videos': sum(len(s['clips']) for s in data['samples'])}))
    elif args.command == 'evaluate':
        from .runner import run
        run(args)
    elif args.command == 'align':
        from .reference import align
        align(args)
    else:
        from .gemini import run_gemini
        run_gemini(args)
