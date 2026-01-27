from __future__ import annotations

import numpy as np

from .linalg import _as_1d, _as_2d


def build_response_matrix(
    n_truth: int,
    n_reco: int | None = None,
    *,
    sigma: float = 1.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Build a simple Gaussian-smearing response matrix.

    Convention: y (reco counts) = R @ x (truth counts)
    - R has shape (n_reco, n_truth)
    - Columns of R sum to 1 (probability of reco bin given truth bin)

    This is a toy model, good enough for demonstrating unfolding + regularization.
    """

    n_reco = n_truth if n_reco is None else int(n_reco)
    rng = np.random.default_rng(seed)

    truth_centers = np.linspace(0.0, float(n_truth - 1), n_truth)
    reco_centers = np.linspace(0.0, float(n_reco - 1), n_reco)

    # Slightly randomize bin mapping to avoid being too perfect.
    truth_centers = truth_centers + 0.05 * rng.normal(size=n_truth)

    R = np.empty((n_reco, n_truth), dtype=float)
    for j, mu in enumerate(truth_centers):
        w = np.exp(-0.5 * ((reco_centers - mu) / sigma) ** 2)
        s = float(np.sum(w))
        R[:, j] = w / s if s > 0 else 0.0

    return R


def smear_truth(
    truth: np.ndarray,
    response: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    poisson: bool = True,
) -> np.ndarray:
    """Generate reco observation from truth counts and response matrix."""

    truth = _as_1d(truth, "truth")
    R = _as_2d(response, "response")
    if R.shape[1] != truth.shape[0]:
        raise ValueError(f"response shape {R.shape} incompatible with truth {truth.shape}")

    mu = R @ truth
    if not poisson:
        return mu

    rng = np.random.default_rng() if rng is None else rng
    return rng.poisson(mu).astype(float)
