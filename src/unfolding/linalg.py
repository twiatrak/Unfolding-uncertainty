from __future__ import annotations

import numpy as np


def _as_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")
    return x


def _as_2d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {a.shape}")
    return a


def symmetric_psd_pinv(a: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Pseudo-inverse of a symmetric PSD matrix using eigen-decomposition."""
    a = _as_2d(a, "a")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be square, got {a.shape}")

    w, v = np.linalg.eigh(a)
    w_max = float(np.max(w)) if w.size else 0.0
    cutoff = rcond * w_max
    w_inv = np.zeros_like(w)
    mask = w > cutoff
    w_inv[mask] = 1.0 / w[mask]
    return (v * w_inv) @ v.T


def assert_shape(a: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    a = np.asarray(a)
    if a.shape != shape:
        raise ValueError(f"{name} has shape {a.shape}, expected {shape}")
