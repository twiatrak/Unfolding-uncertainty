from __future__ import annotations

import numpy as np

from .linalg import _as_1d, _as_2d, symmetric_psd_pinv


def _diff_matrix(n: int, order: int = 1) -> np.ndarray:
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1, or 2")
    if order == 0:
        return np.eye(n)

    if order == 1:
        # Shape: (n-1, n)
        return np.eye(n - 1, n, k=1) - np.eye(n - 1, n, k=0)

    # order == 2
    # Shape: (n-2, n)
    return np.eye(n - 2, n, k=0) - 2.0 * np.eye(n - 2, n, k=1) + np.eye(n - 2, n, k=2)


def tikhonov_unfold(
    y: np.ndarray,
    R: np.ndarray,
    *,
    reg: float,
    cov_y: np.ndarray | None = None,
    L: np.ndarray | None = None,
    diff_order: int = 2,
    nonneg: bool = False,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Tikhonov-regularized unfolding with linear covariance propagation.

    Solves (R^T W R + reg^2 L^T L) x = R^T W y

    - If cov_y is None, uses Poisson approximation cov_y = diag(max(y, 1)).
    - Returns (x_hat, cov_x).

    Notes:
    - The returned cov_x is the linearized covariance from the estimator.
    - nonneg=True clips negative bins to 0 (nonlinear; covariance is still linearized).
    """

    y = _as_1d(y, "y")
    R = _as_2d(R, "R")

    n_reco, n_truth = R.shape
    if y.shape[0] != n_reco:
        raise ValueError(f"y shape {y.shape} incompatible with R {R.shape}")

    if cov_y is None:
        cov_y = np.diag(np.maximum(y, 1.0))
    cov_y = _as_2d(cov_y, "cov_y")
    if cov_y.shape != (n_reco, n_reco):
        raise ValueError(f"cov_y must be {(n_reco, n_reco)}, got {cov_y.shape}")

    if L is None:
        L = _diff_matrix(n_truth, order=diff_order)
    L = _as_2d(L, "L")
    if L.shape[1] != n_truth:
        raise ValueError(f"L must have {n_truth} columns, got {L.shape}")

    if reg < 0:
        raise ValueError("reg must be non-negative")

    # Weight matrix W = cov_y^{-1}
    W = symmetric_psd_pinv(cov_y, rcond=rcond)

    A = R.T @ W @ R + (reg**2) * (L.T @ L)
    b = R.T @ W @ y

    x = np.linalg.solve(A, b)

    # Linear covariance propagation: x = G y, where G = A^{-1} R^T W
    G = np.linalg.solve(A, R.T @ W)
    cov_x = G @ cov_y @ G.T

    if nonneg:
        x = np.maximum(x, 0.0)

    return x, cov_x
