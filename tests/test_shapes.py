import numpy as np

from unfolding.response import build_response_matrix, smear_truth
from unfolding.tikhonov import tikhonov_unfold
from unfolding.tsvd import tsvd_unfold


def test_response_shapes():
    R = build_response_matrix(12, 10, sigma=1.2, seed=1)
    assert R.shape == (10, 12)
    assert np.allclose(R.sum(axis=0), 1.0)


def test_smear_truth_shape():
    R = build_response_matrix(8, 8, sigma=1.0, seed=2)
    x = np.linspace(10, 20, 8)
    y = smear_truth(x, R, poisson=False)
    assert y.shape == (8,)


def test_tikhonov_covariance_psdish():
    R = build_response_matrix(6, 6, sigma=1.0, seed=0)
    truth = np.array([20, 40, 60, 40, 20, 10], dtype=float)
    y = (R @ truth).astype(float)

    x_hat, cov_x = tikhonov_unfold(y, R, reg=0.5)
    assert x_hat.shape == truth.shape
    assert cov_x.shape == (truth.size, truth.size)

    # symmetry + non-negative diagonal
    assert np.allclose(cov_x, cov_x.T, atol=1e-10)
    assert np.all(np.diag(cov_x) >= -1e-10)


def test_tsvd_shapes():
    R = build_response_matrix(7, 7, sigma=1.1, seed=3)
    truth = np.linspace(5, 25, 7)
    y = (R @ truth).astype(float)

    x_hat, cov_x = tsvd_unfold(y, R, k=5)
    assert x_hat.shape == truth.shape
    assert cov_x.shape == (truth.size, truth.size)
