from __future__ import annotations

import numpy as np

from .linalg import _as_1d, _as_2d


def tsvd_unfold(
    y: np.ndarray,
    R: np.ndarray,
    *,
    k: int,
    cov_y: np.ndarray | None = None,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated-SVD unfolding with linear covariance propagation.

    Uses SVD R = U S V^T, and truncated inverse R_k^+ = V_k S_k^{-1} U_k^T.

    Returns (x_hat, cov_x).
    """

    y = _as_1d(y, "y")
    R = _as_2d(R, "R")
    n_reco, n_truth = R.shape
    if y.shape[0] != n_reco:
        raise ValueError(f"y shape {y.shape} incompatible with R {R.shape}")

    if cov_y is None:
        cov_y = np.diag(np.maximum(y, 1.0))
    cov_y = _as_2d(cov_y, "cov_y")

    U, s, Vt = np.linalg.svd(R, full_matrices=False)

    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, s.size)

    # truncate by rcond as well
    s_max = float(np.max(s)) if s.size else 0.0
    keep = (s[:k] > rcond * s_max)
    kk = int(np.sum(keep))
    if kk == 0:
        raise ValueError("all singular values truncated; increase k or decrease rcond")

    Uk = U[:, :kk]
    sk = s[:kk]
    Vk = Vt[:kk, :].T

    Rpinv = Vk @ np.diag(1.0 / sk) @ Uk.T

    x = Rpinv @ y
    cov_x = Rpinv @ cov_y @ Rpinv.T
    return x, cov_x
