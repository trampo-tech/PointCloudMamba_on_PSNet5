"""CPU-only checks for mixed sphere center schedules."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import unittest
import tempfile
from sklearn.neighbors import KDTree

spec = importlib.util.spec_from_file_location(
    'sampling_under_test', Path(__file__).resolve().parents[1] /
    'openpoints/dataset/psnet5/sampling.py')
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def clouds():
    result = []
    for ci in range(2):
        points = np.random.default_rng(ci).uniform(-1, 1, (100, 3)).astype(np.float32)
        # All eligible centers satisfy the training buffer; ineligible points do not.
        points[:80, 0] -= 5
        labels = np.arange(100) % 5
        result.append(dict(sub_points=points, sub_labels=labels,
                           eligible=np.arange(100) < 80, tree=KDTree(points),
                           axis=0, threshold=0., buffer_radius=2., role='train'))
    return result


def test_mixed_sampling_reproducible_and_eligible(tmp_path):
    c = clouds()
    args = (c, 2, 100, 2., 50, 17)
    a = sampling.make_schedule(*args, tmp_path/'a.npz', class_sampling_ratio=.25)
    b = sampling.make_schedule(*args, tmp_path/'b.npz', class_sampling_ratio=.25)
    cached = sampling.make_schedule(*args, tmp_path/'a.npz', class_sampling_ratio=.25)
    for x, y, z in zip(a, b, cached):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, z)
    for ci, pi, noise in zip(*a):
        assert c[ci]['eligible'][pi]
        assert (c[ci]['sub_points'][pi] + noise)[0] < -2
    d = json.loads((tmp_path/'a.sampling.json').read_text())
    assert np.sum(d['center_counts_per_epoch']) == 200
    assert 25 < np.sum(d['class_selected_center_counts_per_epoch']) < 80
    assert np.sum(d['valid_point_counts_per_epoch']) > 200
    with unittest.TestCase().assertRaisesRegex(ValueError, 'different'):
        sampling.make_schedule(*args, tmp_path/'a.npz', class_sampling_ratio=.5)


def test_validation_and_invalid_ratios_rejected(tmp_path):
    c = clouds()
    for ratio in [-.1, 1.1, float('nan')]:
        with unittest.TestCase().assertRaises(ValueError):
            sampling.make_schedule(c, 1, 10, 2., 50, 17, tmp_path/'x.npz', ratio)
    c[0]['role'] = 'val'
    with unittest.TestCase().assertRaisesRegex(ValueError, 'only allowed for training'):
        sampling.make_schedule(c, 1, 10, 2., 50, 17, tmp_path/'x.npz', .25)


def test_global_class_selection_handles_missing_classes(tmp_path):
    c = clouds()
    c[0]['sub_labels'][:] = 0
    c[1]['sub_labels'][:] = 4
    a = sampling.make_schedule(c, 1, 100, 2., 50, 17, tmp_path/'x.npz', 1.)
    d = json.loads((tmp_path/'x.sampling.json').read_text())
    assert d['available_center_classes'] == [0, 4]
    counts = d['center_counts_per_epoch'][0]
    assert counts[0] > 25 and counts[4] > 25
    assert sum(counts[1:4]) == 0


if __name__ == '__main__':
    for test in (test_mixed_sampling_reproducible_and_eligible,
                 test_validation_and_invalid_ratios_rejected,
                 test_global_class_selection_handles_missing_classes):
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
        print(test.__name__, 'PASSED')
    # Compare against the pre-change source when supplied by the verification job.
    original = Path('/tmp/psnet5_sampling_original.py')
    if original.exists():
        spec = importlib.util.spec_from_file_location('original_sampling', original)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            args = (clouds(), 2, 100, 2., 50, 17)
            a = module.make_schedule(*args, Path(directory)/'original.npz')
            b = sampling.make_schedule(*args, Path(directory)/'new.npz')
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        print('zero-ratio original schedule parity PASSED')
