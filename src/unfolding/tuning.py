from __future__ import annotations

import numpy as np

from .linalg import _as_1d, _as_2d
from .tikhonov import tikhonov_unfold
from .toys import run_toy_mc


def scan_tikhonov_regularization(
    truth: np.ndarray,
    R: np.ndarray,
    regs: np.ndarray,
    *,
    n_toys: int,
    diff_order: int = 2,
    L: np.ndarray | None = None,
    nonneg: bool = False,
    rcond: float = 1e-12,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate a grid of Tikhonov strengths with toy-MC quality metrics.

    Returns arrays indexed by the same order as ``regs``:
    - ``regs``
    - ``mean_abs_rel_bias``
    - ``mean_abs_pull_mean``
    - ``mean_abs_pull_width_error``
    - ``score`` (smaller is better)

    The score is a simple sum of three terms:
    score = mean_abs_rel_bias + mean_abs_pull_mean + mean_abs_pull_width_error
    """

    truth = _as_1d(truth, "truth")
    R = _as_2d(R, "R")
    regs = _as_1d(regs, "regs")

    if R.shape[1] != truth.shape[0]:
        raise ValueError("truth incompatible with R")
    if regs.size == 0:
        raise ValueError("regs must be non-empty")
    if np.any(regs < 0):
        raise ValueError("all reg values must be non-negative")
    if n_toys <= 1:
        raise ValueError("n_toys must be > 1")

    mean_abs_rel_bias = np.empty(regs.size, dtype=float)
    mean_abs_pull_mean = np.empty(regs.size, dtype=float)
    mean_abs_pull_width_error = np.empty(regs.size, dtype=float)

    truth_scale = np.maximum(truth, 1.0)
    rng = np.random.default_rng() if rng is None else rng

    for i, reg in enumerate(regs):
        reg_val = float(reg)

        def unfold(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return tikhonov_unfold(
                y,
                R,
                reg=reg_val,
                L=L,
                diff_order=diff_order,
                nonneg=nonneg,
                rcond=rcond,
            )

        stats = run_toy_mc(truth, R, n_toys=n_toys, unfold=unfold, rng=rng)

        rel_bias = np.abs(stats["bias"]) / truth_scale
        mean_abs_rel_bias[i] = float(np.mean(rel_bias))
        mean_abs_pull_mean[i] = float(np.mean(np.abs(stats["pull_mean"])))
        mean_abs_pull_width_error[i] = float(np.mean(np.abs(stats["pull_std"] - 1.0)))

    score = mean_abs_rel_bias + mean_abs_pull_mean + mean_abs_pull_width_error

    return {
        "regs": regs.copy(),
        "mean_abs_rel_bias": mean_abs_rel_bias,
        "mean_abs_pull_mean": mean_abs_pull_mean,
        "mean_abs_pull_width_error": mean_abs_pull_width_error,
        "score": score,
    }


def choose_best_regularization(scan: dict[str, np.ndarray]) -> tuple[float, int]:
    """Pick the regularization value with the smallest scan score.

    Returns ``(best_reg, index)``.
    """

    if "regs" not in scan or "score" not in scan:
        raise ValueError("scan must contain 'regs' and 'score'")

    regs = _as_1d(scan["regs"], "scan['regs']")
    score = _as_1d(scan["score"], "scan['score']")

    if regs.shape != score.shape:
        raise ValueError("scan['regs'] and scan['score'] must have the same shape")
    if regs.size == 0:
        raise ValueError("scan arrays must be non-empty")

    idx = int(np.argmin(score))
    return float(regs[idx]), idx
