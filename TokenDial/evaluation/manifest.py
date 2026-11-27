"""Strict input validation; logical intensities and generator scales are distinct."""
import hashlib
import json
import math
from pathlib import Path
import re

SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path, probe=False):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Expected schema_version=1')
    if not isinstance(data.get('samples'), list) or not data['samples']:
        raise ValueError('Manifest must contain a nonempty samples list')
    seen = set()
    for s in data['samples']:
        for k in ['method', 'concept', 'attribute', 'sample']:
            if not isinstance(s.get(k), str) or not SAFE.fullmatch(s[k]):
                raise ValueError(f'Unsafe or missing {k}: {s.get(k)!r}')
        key = sample_key(s)
        if key in seen:
            raise ValueError(f'Duplicate sample key: {key}')
        seen.add(key)
        clips = s.get('clips', [])
        if len(clips) < 2:
            raise ValueError(f'{key}: slider evaluation requires at least two clips')
        levels = [c.get('intensity') for c in clips]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in levels):
            raise ValueError(f'{key}: intensities must be finite numbers')
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError(f'{key}: intensities must be unique and strictly increasing')
        if any('is_baseline' not in c or not isinstance(c['is_baseline'], bool) for c in clips):
            raise ValueError(f'{key}: every clip must explicitly declare is_baseline')
        if [c['is_baseline'] for c in clips] != [True] + [False] * (len(clips) - 1):
            raise ValueError(f'{key}: exactly the first logical intensity must be baseline')
        if not isinstance(s.get('baseline_kind'), str) or not s['baseline_kind'].strip():
            raise ValueError(f'{key}: declare baseline_kind, e.g. zero_strength_reconstruction')
        video_paths = []
        for c in clips:
            scale = c.get('model_scale')
            if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale)):
                raise ValueError(f'{key}: model_scale must be finite when supplied')
            if not isinstance(c.get('video'), str) or not c['video']:
                raise ValueError(f'{key}: missing video path')
            video = Path(c['video'])
            if not video.is_absolute():
                video = path.parent / video
            video = video.resolve()
            if not video.is_file() or video.suffix.lower() != '.mp4':
                raise ValueError(f'{key}: missing MP4: {video}')
            c['video'] = str(video)
            video_paths.append(str(video))
        if len(set(video_paths)) != len(video_paths):
            raise ValueError(f'{key}: multiple levels point to the same video')
        if s['baseline_kind'].startswith('zero_strength') and clips[0].get('model_scale', 0) != 0:
            raise ValueError(f'{key}: zero-strength baseline has nonzero model_scale')
        if s.get('source_video'):
            source = Path(s['source_video'])
            if not source.is_absolute():
                source = path.parent / source
            if not source.is_file():
                raise ValueError(f'{key}: source video does not exist: {source}')
            s['source_video'] = str(source.resolve())
        if probe:
            details = [probe_video(c['video']) for c in clips]
            for field in ['frames', 'width', 'height']:
                if len({d[field] for d in details}) != 1:
                    raise ValueError(f'{key}: mismatched video {field}: {details}')
            if max(d['fps'] for d in details) - min(d['fps'] for d in details) > 1e-3:
                raise ValueError(f'{key}: mismatched video frame rates')
            s['video_probe'] = details
    return data


def probe_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        d = {'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
             'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             'fps': float(cap.get(cv2.CAP_PROP_FPS))}
        if any(d[k] <= 0 for k in d):
            raise ValueError(f'Invalid video metadata: {path}: {d}')
        return d
    finally:
        cap.release()


def sample_key(s):
    return tuple(s[k] for k in ['method', 'concept', 'attribute', 'sample'])


def load_guidance(root, s):
    path = Path(root) / s['concept'] / (s['attribute'] + '.json')
    if s['concept'] == 'person_walk' and s['attribute'] == 'walk_to_run' and not path.exists():
        path = Path(root) / 'person' / 'walk_to_run.json'
    obj = json.loads(path.read_text(encoding='utf-8'))
    for k in ['direction_prompts', 'neg_direction_prompts']:
        values = obj.get(k)
        if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError(f'{path}: {k} must be a nonempty list of nonempty strings')
    return obj, path
