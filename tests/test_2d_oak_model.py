import numpy as np
import pytest
import tensorflow as tf
import gpflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from oak.model_utils import oak_model

# add two new params: base_kernels and active_dims
@pytest.mark.parametrize("interaction_depth", [1, 2])
@pytest.mark.parametrize("use_sparsity_prior", [True, False])
@pytest.mark.parametrize("initialise_inducing_points", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("clip", [True, False])
@pytest.mark.parametrize(
    "base_kernels, active_dims",
    [
        # custom 1D+2D blocks
        (
            [
                gpflow.kernels.RBF,
                lambda **kw: gpflow.kernels.RBF(),
            ],
            # 3‐dim data: block0 → dim0, block1 → dims[1,2], block2 → dim2 (redundant single)
            # to keep it simple we can just do [[0],[1,2]]
            [[0], [1, 2]],
        ),
    ],
)
def test_oak_model_with_optional_blocks(
    interaction_depth: int,
    use_sparsity_prior: bool,
    initialise_inducing_points: bool,
    sparse: bool,
    clip: bool,
    base_kernels,
    active_dims,
):
    np.random.seed(44)
    tf.random.set_seed(44)

    N = 100
    X = np.random.normal(0, 1, (N, 3))
    y = X[:, 0] ** 2 + X[:, 1] + X[:, 1] * X[:, 2] + np.random.normal(0, 0.01, (N,))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y[:, None], test_size=0.2, random_state=42
    )

    # pass base_kernels and active_dims only when they're not None
    kwargs = dict(
        num_inducing=50,
        max_interaction_depth=interaction_depth,
        use_sparsity_prior=use_sparsity_prior,
        sparse=sparse,
    )
    if base_kernels is not None:
        kwargs["base_kernels"] = base_kernels
    if active_dims is not None:
        kwargs["active_dims"] = active_dims

    oak = oak_model(**kwargs)
    oak.fit(
        X_train,
        y_train,
        initialise_inducing_points=initialise_inducing_points,
        optimise=False,
    )

    y_pred = oak.predict(X_test, clip=clip)
    rss = mean_squared_error(y_pred, y_test[:, 0])
    # model should beat trivial mean predictor
    assert rss < mean_squared_error(
        y_test.mean() * np.ones_like(y_test[:, 0]), y_test[:, 0]
    )
    
@pytest.fixture
def small_3d_regression_data():
    np.random.seed(123)
    N = 40
    X = np.random.randn(N, 3).astype(np.float64)
    # simple nonlinear function of x0 and (x1,x2) interaction
    Y = (X[:, 0] + X[:, 1] * X[:, 2]).reshape(-1, 1).astype(np.float64)
    return X, Y

def test_sobol_with_1d_2d_blocks(small_3d_regression_data):
    X, Y = small_3d_regression_data
    tf.random.set_seed(123)

    # --- define a 1-D + 2-D block kernel -------------------------------
    base_kernels = [
        gpflow.kernels.RBF,                                    # block 0 → dim 0
        lambda **kw: gpflow.kernels.RBF()
    ]
    active_dims = [[0], [1, 2]]                               # block 1 → dims 1 & 2

    oak = oak_model(
        base_kernels=base_kernels,
        active_dims=active_dims,
        max_interaction_depth=2,
        use_sparsity_prior=False,
        sparse=False,
    )
    oak.fit(X, Y, optimise=False)

    # --- run Sobol ------------------------------------------------------
    sobol = oak.get_sobol(likelihood_variance=False)

    # basic sanity checks
    assert sobol.ndim == 1
    assert np.all(sobol >= 0)
    np.testing.assert_allclose(sobol.sum(), 1.0, atol=1e-6)

    # print for debug (pytest -s will show)
    print("Sobol indices:", sobol)

    # assert that sobol indices length are equal to number of blocks + 2nd order interactions between them
    assert len(sobol) == 3  # +1 for the interaction term