import numpy as np
import pytest

from unfolding.response import build_response_matrix
from unfolding.tuning import choose_best_regularization, scan_tikhonov_regularization


def test_scan_tikhonov_regularization_shapes_and_best_choice():
    truth = np.array([40, 60, 90, 70, 45, 30], dtype=float)
    R = build_response_matrix(n_truth=truth.size, sigma=1.0, seed=7)
    regs = np.array([0.0, 0.1, 0.3, 0.8], dtype=float)

    scan = scan_tikhonov_regularization(
        truth,
        R,
        regs,
        n_toys=80,
        diff_order=2,
        rng=np.random.default_rng(11),
    )

    assert scan["regs"].shape == regs.shape
    assert scan["mean_abs_rel_bias"].shape == regs.shape
    assert scan["mean_abs_pull_mean"].shape == regs.shape
    assert scan["mean_abs_pull_width_error"].shape == regs.shape
    assert scan["score"].shape == regs.shape
    assert np.all(np.isfinite(scan["score"]))

    best_reg, idx = choose_best_regularization(scan)
    assert 0 <= idx < regs.size
    assert np.isclose(best_reg, regs[idx])
    assert np.isclose(scan["score"][idx], np.min(scan["score"]))


def test_scan_tikhonov_regularization_input_validation():
    truth = np.array([10, 20, 30], dtype=float)
    R = build_response_matrix(3, sigma=1.0, seed=1)

    with pytest.raises(ValueError, match="non-empty"):
        scan_tikhonov_regularization(truth, R, np.array([]), n_toys=20)

    with pytest.raises(ValueError, match="non-negative"):
        scan_tikhonov_regularization(truth, R, np.array([0.1, -0.2]), n_toys=20)

    with pytest.raises(ValueError, match="> 1"):
        scan_tikhonov_regularization(truth, R, np.array([0.1]), n_toys=1)


def test_choose_best_regularization_validation():
    with pytest.raises(ValueError, match="contain"):
        choose_best_regularization({"regs": np.array([0.1])})

    with pytest.raises(ValueError, match="same shape"):
        choose_best_regularization({"regs": np.array([0.1, 0.2]), "score": np.array([1.0])})
