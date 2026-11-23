import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from evaluation.author import load_author
from evaluation.manifest import load_manifest, load_guidance
from evaluation.metrics import calculate, summarize
from evaluation.reference import compare


class MetricTests(unittest.TestCase):
    def test_author_equivalence_on_varied_trajectories(self):
        author = load_author()
        rng = np.random.default_rng(1381)
        for n in [2, 3, 5, 9]:
            for _ in range(20):
                anchor, pos, neg = [rng.normal(size=n).tolist() for _ in range(3)]
                lp = rng.uniform(size=n).tolist()
                keys = list(map(str, range(n)))
                a, p, m, l = [dict(zip(keys, x)) for x in [anchor, pos, neg, lp]]
                result = calculate(anchor, pos, neg, lp)
                self.assertEqual(result['strength_anchor'], author.compute_strength_anchor(p, m, keys))
                self.assertEqual(result['monotonicity_anchor'], author.monotonicity_from_scale_scores(a, keys, 1e-4))
                self.assertEqual(result['one_sided_span_anchor'], author.one_sided_span_from_scale_scores(a, keys))
                self.assertEqual(result['lpips_preservation'], author.compute_lpips_preservation(l, keys[0]))

    def test_eps_and_flat_sequence(self):
        result = calculate([0, 1e-4, 1e-4, 0], [1]*4, [1]*4, [0]*4)
        self.assertEqual(result['monotonicity_anchor'], 0)
        self.assertEqual(result['strength_anchor'], 0)
        self.assertEqual(calculate([1]*5, [1]*5, [1]*5, [0]*5)['one_sided_span_anchor'], 0)

    def test_nonfinite_is_error(self):
        with self.assertRaises(ValueError):
            calculate([0, float('nan')], [0, 1], [0, 1], [0, 1])

    def test_overall_aggregation_is_explicitly_distinct(self):
        rows = []
        for p, lp in [([0, 2], [0, 0]), ([0, 4], [0, 3])]:
            rows.append(dict(method='test', **calculate([0, 1], p, [0, 0], lp)))
        s = summarize(rows)['methods']['test']
        self.assertNotEqual(s['overall_mean_of_sample_scores'], s['overall_from_aggregate_metrics'])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for i in range(5):
            (self.root / f'{i}.mp4').write_bytes(b'placeholder-not-a-real-video')
        self.data = {'schema_version': 1, 'samples': [{'method': 'tokendial', 'concept': 'person',
            'attribute': 'older', 'sample': 'seed_0', 'baseline_kind': 'zero_strength_reconstruction',
            'clips': [{'intensity': i, 'model_scale': i, 'is_baseline': i==0, 'video': f'{i}.mp4'} for i in range(5)]}]}

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        p = self.root / 'manifest.json'
        p.write_text(json.dumps(self.data), encoding='utf-8')
        return load_manifest(p)

    def test_relative_paths_resolve(self):
        self.assertEqual(self.load()['samples'][0]['clips'][0]['video'], str((self.root/'0.mp4').resolve()))

    def test_missing_level_fails(self):
        (self.root/'2.mp4').unlink()
        with self.assertRaisesRegex(ValueError, 'missing MP4'):
            self.load()

    def test_duplicate_key_fails(self):
        self.data['samples'].append(copy.deepcopy(self.data['samples'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.load()

    def test_wrong_baseline_fails(self):
        c = self.data['samples'][0]['clips']
        c[0]['is_baseline'], c[1]['is_baseline'] = False, True
        with self.assertRaisesRegex(ValueError, 'first logical intensity'):
            self.load()

    def test_duplicate_intensity_fails(self):
        self.data['samples'][0]['clips'][1]['intensity'] = 0
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            self.load()

    def test_missing_guidance_does_not_fallback(self):
        with self.assertRaises(FileNotFoundError):
            load_guidance(self.root, self.data['samples'][0])


class AlignmentTests(unittest.TestCase):
    def row(self):
        return dict(method='ours', concept='person', attribute='older', sample='seed0',
                    anchor_prompt='person extremely older', baseline_scale='1',last_scale='5',
                    total_scales='5',scales_present='1,2,3,4,5',
                    strength_anchor='1',monotonicity_anchor='1',one_sided_span_anchor='0',lpips_preservation='.2')

    def test_missing_sample_is_not_success(self):
        key=('ours','person','older','seed0')
        self.assertFalse(compare({key:self.row()}, {}, [key])['passed'])

    def test_changed_anchor_and_metric_are_detected(self):
        key=('ours','person','older','seed0')
        changed=self.row();changed['anchor_prompt']='wrong';changed['strength_anchor']='2'
        report=compare({key:self.row()}, {key:changed}, [key])
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['differences']), 2)

    def test_exact_match_passes(self):
        key=('ours','person','older','seed0')
        self.assertTrue(compare({key:self.row()}, {key:self.row()}, [key])['passed'])


if __name__ == '__main__':
    unittest.main()
