import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.adapt import adapt
from evaluation.manifest import load_manifest


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV needed for actual video adapter tests')
class VideoAdapterTests(unittest.TestCase):
    def test_negative_generator_scales_preserve_zero_baseline(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root/'videos/seed_0'
            folder.mkdir(parents=True)
            for scale in [0.0, -1.0, -2.0, -3.0, -4.0]:
                w = cv2.VideoWriter(str(folder/f'flowedit_rgb{scale}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 15, (64,64))
                for frame in range(9):
                    w.write(np.full((64,64,3), int(80-scale*8+frame), dtype=np.uint8))
                w.release()
            args = argparse.Namespace(root=str(root/'videos'), seeds='0', model_scales='0,-1,-2,-3,-4',
                baseline_scale=0, manifest=str(root/'manifest.json'), kind='v2v', notes='unit test synthetic video',
                method='test', concept='speed', attribute='faster', checkpoint=None, source_video=None, prompt='motion')
            with contextlib.redirect_stdout(io.StringIO()):adapt(args)
            result = load_manifest(args.manifest, probe=True)
            c = result['samples'][0]['clips']
            self.assertEqual([x['model_scale'] for x in c],[0,-1,-2,-3,-4])
            self.assertEqual([x['intensity'] for x in c],[0,1,2,3,4])
            self.assertTrue(c[0]['is_baseline'])

    def test_frame_count_mismatch_is_rejected(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i, count in enumerate([5,9]):
                w=cv2.VideoWriter(str(root/f'{i}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64))
                for _ in range(count):w.write(np.zeros((64,64,3),dtype=np.uint8))
                w.release()
            data={'schema_version':1,'samples':[{'method':'test','concept':'person','attribute':'older','sample':'test',
                'baseline_kind':'zero_strength_reconstruction','clips':[{'intensity':i,'is_baseline':i==0,'video':f'{i}.mp4'} for i in range(2)]}]}
            p=root/'manifest.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'mismatched video frames'):load_manifest(p,probe=True)
