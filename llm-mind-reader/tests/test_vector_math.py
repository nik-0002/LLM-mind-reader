"""Run with `pytest tests/` or simply `python tests/test_vector_math.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vector_math as vm


def synthetic(n=60, dim=64, signal=3.0, noise=1.0, seed=0, base_scale=5.0):
    rng = np.random.default_rng(seed)
    true = rng.standard_normal(dim)
    true /= np.linalg.norm(true)
    base = rng.standard_normal((n, dim)) * base_scale  # shared "content" of each prompt
    pos = base + signal * true / 2 + rng.standard_normal((n, dim)) * noise
    neg = base - signal * true / 2 + rng.standard_normal((n, dim)) * noise
    return pos, neg, true


def test_methods_recover_direction():
    # NB: 'pca_raw' (centered PCA on un-flipped diffs) is deliberately excluded - see last test.
    pos, neg, true = synthetic()
    for m in ("pca", "mean_diff"):
        d, _ = vm.compute_direction(pos - neg, m)
        assert vm.cosine(d, true) > 0.8, (m, vm.cosine(d, true))
        assert abs(np.linalg.norm(d) - 1) < 1e-5


def test_sign_is_positive_side():
    pos, neg, true = synthetic()
    d, info = vm.compute_direction(pos - neg, "pca", seed=3)
    assert np.dot(d, true) > 0 and info["cos_with_mean_diff"] > 0


def test_scale_and_calibration():
    pos, neg, true = synthetic(signal=4.0)
    d, _ = vm.compute_direction(pos - neg, "mean_diff")
    s = vm.steer_scale(pos - neg, d)
    assert 3.0 < s < 5.0
    cal = vm.calibration(pos @ d, neg @ d)
    assert abs(cal["gap"] - s) < 1e-4
    assert vm.normalized_score(float((neg @ d).mean()), cal) == 0.0
    assert abs(vm.normalized_score(float((pos @ d).mean()), cal) - 1.0) < 1e-9


def test_heldout_metrics_and_null():
    pos, neg, _ = synthetic(n=80, signal=6.0, base_scale=1.0)
    tr, te = vm.grouped_split(80, 0.25, seed=1)
    assert set(tr).isdisjoint(te) and len(te) == 20
    d, _ = vm.compute_direction((pos - neg)[tr], "pca")
    m = vm.pair_metrics(pos[te], neg[te], d)
    assert m["pair_accuracy"] > 0.9 and m["auroc"] > 0.7
    acts = np.vstack([pos[te], neg[te]])
    y = np.r_[np.ones(20), np.zeros(20)]
    null = vm.null_auroc(acts, y, 100)
    assert abs(null.mean() - 0.5) < 0.1 and m["auroc"] > np.percentile(null, 99)


def test_explained_variance_always_present():
    pos, neg, _ = synthetic()
    _, info = vm.compute_direction(pos - neg, "mean_diff")
    assert len(info["explained_variance_ratio"]) == 1


def test_naive_pca_is_worse_when_diffs_are_consistent():
    """Documents WHY we flip signs: raw centered PCA throws away the mean (= concept) direction."""
    rng = np.random.default_rng(5)
    dim, n = 64, 60
    true = rng.standard_normal(dim); true /= np.linalg.norm(true)
    nuisance = rng.standard_normal(dim); nuisance /= np.linalg.norm(nuisance)
    # diffs: constant concept shift + large *varying* nuisance direction
    diffs = 2.0 * true + rng.standard_normal((n, 1)) * 1.0 * nuisance + rng.standard_normal((n, dim)) * 0.1
    d_raw, _ = vm.compute_direction(diffs, "pca_raw")
    d_flip, _ = vm.compute_direction(diffs, "pca")
    assert vm.cosine(d_flip, true) > 0.9
    d_md, _ = vm.compute_direction(diffs, "mean_diff")
    assert vm.cosine(d_md, true) > 0.9
    assert abs(vm.cosine(d_raw, true)) < abs(vm.cosine(d_md, true))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
