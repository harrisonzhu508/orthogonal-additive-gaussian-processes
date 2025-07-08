import numpy as np
import gpflow
import tensorflow as tf
import pytest

from oak.ortho_rbf_kernel import OrthogonalRBFKernel
from oak.input_measures import GaussianMeasure
from oak.oak_kernel import OAKKernel, KernelComponenent

@pytest.fixture
def simple_2d_data():
    # two points in 2D
    return np.array([[0.0, 0.0], [1.5, -2.0]]), np.array([[2.0, 3.0]])

def test_cov_X_s_and_var_s_2d_gaussian(simple_2d_data):
    X, _ = simple_2d_data

    # ARD lengthscales and variance
    lengthscales = np.array([1.], dtype=np.float64)
    variance = 2.0

    base = gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance)
    measure = GaussianMeasure(mu=0., var=1.)
    k = OrthogonalRBFKernel(base, measure, active_dims=[0, 1])

    cov = k.cov_X_s(tf.convert_to_tensor(X))  # shape [2,1]
    var_s = k.var_s()                          # scalar

    # --- closed-form expected cov_X_s ---
    # norm = σ² ∏ₙ ℓₙ / sqrt(ℓₙ² + σ_s²)
    norm = variance * lengthscales**2 / (lengthscales**2 + 1)
    # exponent = exp(-½ ∑ₙ (xₙ - μₙ)² / (ℓₙ² + σ_s²))
    denom = lengthscales**2 + 1
    diffs = X
    expected_cov = norm * np.exp(-0.5 * np.sum(diffs**2 / denom, axis=1, keepdims=True))

    np.testing.assert_allclose(cov.numpy(), expected_cov, rtol=1e-6)

    # --- closed-form expected var_s ---
    expected_var = variance * lengthscales**2 / (lengthscales**2 + 2 * 1)
    assert np.allclose(var_s.numpy(), expected_var, rtol=1e-6)


def test_K_and_K_diag_consistency_2d(simple_2d_data):
    X, X2 = simple_2d_data

    lengthscales = np.array([1.], dtype=np.float64)
    variance = 1.3

    base = gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance)
    measure = GaussianMeasure(mu=0, var=1.)
    k = OrthogonalRBFKernel(base, measure, active_dims=[0, 1])

    # full matrix
    Kxx = k.K(X, X2)
    # directly via definition: base - cov_X_s(X) cov_X_s(X2)ᵀ / var_s
    cov_X = k.cov_X_s(tf.convert_to_tensor(X))              # [N,1]
    cov_X2 = k.cov_X_s(tf.convert_to_tensor(X2))            # [M,1]
    var_s = k.var_s().numpy()
    K_manual = base.K(X, X2) - (cov_X.numpy() @ cov_X2.numpy().T) / var_s

    np.testing.assert_allclose(Kxx.numpy(), K_manual, rtol=1e-6)

    # K_diag match diag(K)
    Kfull = k.K(X)
    Kdiag = k.K_diag(X)
    assert np.allclose(np.diag(Kfull.numpy()), Kdiag.numpy(), rtol=1e-6)

def test_oakkernel_mixed_dim_blocks():
    # random data with 3 features
    np.random.seed(0)
    X = np.random.normal(size=(10, 3))

    # Block 1: 1D RBF on dim 0
    # Block 2: 2D ARD RBF on dims [1,2]
    active_dims = [[0], [1, 2]]
    base_kernels = [
        gpflow.kernels.RBF,  # 1D RBF
        lambda **kw: gpflow.kernels.RBF()
    ]

    k = OAKKernel(
        base_kernels=base_kernels,
        num_dims=3,
        max_interaction_depth=1,
        active_dims=active_dims,
        constrain_orthogonal=True,
    )
    # assign non-trivial variances
    k.variances[0].assign(0.4)  # constant term
    k.variances[1].assign(1.7)  # first-order interactions

    # full kernel evaluation
    K_full = k.K(X)

    # decompose: constant + block-0 + block-[1,2]
    K_const = KernelComponenent(k, []).K(X)
    K_dim0  = KernelComponenent(k, [0]).K(X)
    K_block = KernelComponenent(k, [1, 2]).K(X)

    np.testing.assert_allclose(
        K_full.numpy(),
        (K_const + K_dim0 + K_block).numpy(),
        atol=1e-6,
        err_msg="Mixed dims decomposition failed"
    )

    # K_diag consistency
    np.testing.assert_allclose(
        np.diag(K_full.numpy()),
        k.K_diag(X).numpy(),
        atol=1e-6,
        err_msg="Mixed dims K_diag mismatch"
    )
