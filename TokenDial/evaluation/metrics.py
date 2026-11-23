"""The four supplied author definitions, with explicitly labelled extra aggregation."""
import math
import numpy as np

METRICS = ['strength_anchor', 'monotonicity_anchor', 'one_sided_span_anchor', 'lpips_preservation']


def calculate(anchor, positive, negative, lpips_values, eps=1e-4):
    n = len(anchor)
    if n < 2 or any(len(v) != n for v in [positive, negative, lpips_values]):
        raise ValueError('Expected aligned score sequences of length >=2')
    if not all(math.isfinite(float(x)) for seq in [anchor, positive, negative, lpips_values] for x in seq):
        raise ValueError('Non-finite scores are not silently omitted')
    diffs = [anchor[i] - anchor[i - 1] for i in range(1, n)]
    cr = float(0.5 * ((positive[-1] - positive[0]) + (negative[0] - negative[-1])))
    csm = float(np.std(np.asarray(diffs, dtype=np.float32)))
    sp = float(np.mean(lpips_values[1:]))
    return {'strength_anchor': cr,
            'monotonicity_anchor': float(np.mean([1.0 if d > eps else 0.0 for d in diffs])),
            'one_sided_span_anchor': csm, 'lpips_preservation': sp,
            'overall_per_sample': cr / (1.0 + sp) + (1.0 - csm)}


def summarize(rows):
    if not rows:
        raise ValueError('No evaluated samples')
    output = {'aggregation': 'micro_mean_over_sample_trajectories',
              'overall_note': 'Added paper-formula aggregation; not validated against an author aggregation script',
              'methods': {}}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == method]
        avg = {k: float(np.mean([float(r[k]) for r in group])) for k in METRICS}
        output['methods'][method] = {'sample_count': len(group), **avg,
            'overall_mean_of_sample_scores': float(np.mean([float(r['overall_per_sample']) for r in group])),
            'overall_from_aggregate_metrics': avg['strength_anchor'] / (1 + avg['lpips_preservation']) + 1 - avg['one_sided_span_anchor']}
    return output
