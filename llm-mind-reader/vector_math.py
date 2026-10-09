"""
vector_math.py — Pure-numpy maths for the RepE pipeline (no torch needed).

Kept separate from repe_engine.py so it can be unit-tested without a GPU/model.

Why this module exists
----------------------
Naively running (centered) PCA on paired difference vectors (pos - neg) is
subtle: sklearn's PCA subtracts the mean, and for difference vectors the *mean*
IS the concept direction. Zou et al. handle this by randomising the order
within each pair (so diffs point in +/- directions and the mean is ~0); the
dominant axis of variance is then the concept axis. We implement that, plus a
difference-of-means baseline, so the two can be compared honestly.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

METHODS = ("pca", "mean_diff", "pca_raw")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def compute_direction(
    diffs: np.ndarray, method: str = "pca", n_components: int = 5, seed: int = 0
) -> Tuple[np.ndarray, Dict]:
    """
    Turn difference vectors (n_pairs, dim) into one unit-norm concept direction.

    method:
      "pca"       - RepE-style: randomly flip half the diffs, fit centered PCA, take PC1.
      "mean_diff" - normalised mean of the diffs (difference-of-means).
      "pca_raw"   - centered PCA on the raw diffs (the naive version; kept for comparison).

    The sign is always aligned so the direction points the same way as the mean
    diff (i.e. towards the *positive* side of the concept).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")
    X = np.asarray(diffs, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("Need a (n_pairs>=2, dim) array of difference vectors.")
    mean_diff = X.mean(axis=0)
    info: Dict = {"method": method}

    if method == "mean_diff":
        d = _unit(mean_diff)
    else:
        Z = X
        if method == "pca":
            rng = np.random.default_rng(seed)
            signs = np.ones(len(X))
            signs[: len(X) // 2] = -1.0
            rng.shuffle(signs)
            Z = X * signs[:, None]
        k = max(1, min(n_components, Z.shape[0], Z.shape[1]))
        pca = PCA(n_components=k, random_state=seed).fit(Z)
        d = pca.components_[0]
        info["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        info["pca"] = pca

    if np.dot(d, mean_diff) < 0:
        d = -d
    d = _unit(d)

    if "explained_variance_ratio" not in info:
        # fraction of the total energy of the diffs that lies along d
        info["explained_variance_ratio"] = [float(np.sum((X @ d) ** 2) / np.sum(X ** 2))]
    info["cos_with_mean_diff"] = float(np.dot(d, _unit(mean_diff)))
    return d.astype(np.float32), info


def steer_scale(diffs: np.ndarray, direction: np.ndarray) -> float:
    """
    Typical size of the concept shift along `direction`: mean(diff . d).

    A steering coefficient of 1.0 then means "move the activation by one typical
    (positive - negative) gap". This is what makes coefficients interpretable -
    a bare unit vector is orders of magnitude smaller than a mid-layer residual
    stream, so coefficients like 3.0 on a unit vector would do almost nothing.
    """
    return float(np.mean(np.asarray(diffs) @ direction))


def calibration(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Dict[str, float]:
    p, n = float(np.mean(pos_scores)), float(np.mean(neg_scores))
    return {"pos_mean": p, "neg_mean": n, "threshold": (p + n) / 2.0, "gap": p - n}


def normalized_score(projection: float, cal: Optional[Dict[str, float]]) -> Optional[float]:
    """0.0 = typical negative example, 1.0 = typical positive example."""
    if not cal or abs(cal["gap"]) < 1e-12:
        return None
    return (projection - cal["neg_mean"]) / cal["gap"]


def pair_metrics(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> Dict[str, float]:
    """Held-out style metrics for a direction on paired activations."""
    ps, ns = pos @ direction, neg @ direction
    n = len(ps)
    y = np.r_[np.ones(n), np.zeros(n)]
    s = np.r_[ps, ns]
    pooled = np.sqrt((ps.var() + ns.var()) / 2.0) + 1e-12
    return {
        "pair_accuracy": float(np.mean(ps > ns)),
        "auroc": float(roc_auc_score(y, s)) if n > 0 else float("nan"),
        "d_prime": float((ps.mean() - ns.mean()) / pooled),
    }


def grouped_split(n_pairs: int, test_size: float = 0.2, seed: int = 0):
    """Split by *pair* so a pair's positive and negative never straddle train/test."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pairs)
    n_test = min(max(1, int(round(n_pairs * test_size))), n_pairs - 1)
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def random_unit_directions(dim: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, dim))
    return (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float32)


def null_auroc(acts: np.ndarray, labels: np.ndarray, n_random: int = 200, seed: int = 0) -> np.ndarray:
    """AUROC of random directions on the same data - the chance-level distribution."""
    R = random_unit_directions(acts.shape[1], n_random, seed)
    return np.array([roc_auc_score(labels, acts @ r) for r in R])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(np.asarray(a, dtype=np.float64)), _unit(np.asarray(b, dtype=np.float64))))
