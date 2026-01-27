from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .linalg import _as_1d, _as_2d


def run_toy_mc(
    truth: np.ndarray,
    R: np.ndarray,
    *,
    n_toys: int,
    unfold: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Run toy-MC validation.

    Parameters
    - truth: truth counts (n_truth,)
    - R: response matrix (n_reco, n_truth)
    - unfold: function y -> (x_hat, cov_x)

    Returns a dict with:
    - x_mean, x_std, bias, pull_mean, pull_std
    """

    truth = _as_1d(truth, "truth")
    R = _as_2d(R, "R")
    n_reco, n_truth = R.shape
    if truth.shape[0] != n_truth:
        raise ValueError("truth incompatible with R")

    rng = np.random.default_rng() if rng is None else rng

    xs = np.zeros((n_toys, n_truth), dtype=float)
    pulls = np.zeros((n_toys, n_truth), dtype=float)

    mu = R @ truth

    for i in range(n_toys):
        y = rng.poisson(mu).astype(float)
        x_hat, cov_x = unfold(y)
        xs[i] = x_hat

        var = np.diag(cov_x)
        sigma = np.sqrt(np.maximum(var, 1e-12))
        pulls[i] = (x_hat - truth) / sigma

    x_mean = xs.mean(axis=0)
    x_std = xs.std(axis=0, ddof=1)
    bias = x_mean - truth

    pull_mean = pulls.mean(axis=0)
    pull_std = pulls.std(axis=0, ddof=1)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "bias": bias,
        "pull_mean": pull_mean,
        "pull_std": pull_std,
    }
