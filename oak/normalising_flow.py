import gpflow
from gpflow.config import default_float

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# Globals & aliases -----------------------------------------------------------
# -----------------------------------------------------------------------------
DTYPE = default_float()           # match GPflow’s global float type
SMALL = 1e-6                      # numerical jitter (same name as upstream file)
tfb = tfp.bijectors               # terse alias à la upstream

# -----------------------------------------------------------------------------
# Helper: per-dimension standardiser -----------------------------------------
# -----------------------------------------------------------------------------

def _standardiser(x: np.ndarray, eps: float = SMALL):
    """Return `[Scale, Shift]` bijectors so `(x-mean)*scale` has unit variance.

    Identical logic to the upstream 1-D helper; we keep the function private to
    avoid leaking symbols.
    """
    x = np.asarray(x)
    m, s = np.mean(x), np.std(x)
    # Prevent divide-by-zero and catastrophic *huge* scales.
    scale_init = 1.0 / (s + eps)
    return [
        tfb.Scale(gpflow.Parameter(scale_init, transform=tfb.Exp(), dtype=DTYPE)),
        tfb.Shift(gpflow.Parameter(-m, dtype=DTYPE)),
    ]


def make_sinharcsinh():
    return tfb.SinhArcsinh(
        skewness=gpflow.Parameter(0.0),
        tailweight=gpflow.Parameter(1.0, transform=tfb.Exp()),
    )


def make_standardizer(x):
    return [
        tfb.Scale(gpflow.Parameter(1.0 / np.std(x), transform=tfb.Exp())),
        tfb.Shift(gpflow.Parameter(-np.mean(x))),
    ]

class Normalizer(gpflow.base.Module):
    """
    :param x: input to transform
    :param log: whether to log x first before applying flows of transformations
    :return: flows of transformations to match x to standard Gaussian
    """

    def __init__(
        self,
        x,
        log=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x = x

        if log:
            offset = np.min(x) - 1.0
            self.bijector = tfb.Chain(
                [make_sinharcsinh() for _ in range(1)]
                + make_standardizer(np.log(x - offset))
                + [tfb.Log(), tfb.Shift(-offset)]
            )
        else:
            self.bijector = tfb.Chain(
                [make_sinharcsinh() for _ in range(1)] + make_standardizer(x)
            )

    def plot(self, title='Normalising Flow'):
        f = plt.figure()
        ax = f.add_axes([0.3, 0.3, 0.65, 0.65])
        x = self.x
        y = self.bijector(x).numpy()
        ax.plot(x, y, "k.", label="Gaussian")
        ax.legend()
        
        ax_x = f.add_axes([0.3, 0.05, 0.65, 0.25], sharex=ax)
        ax_x.hist(x, bins=20)
        ax_y = f.add_axes([0.05, 0.3, 0.25, 0.65], sharey=ax)
        ax_y.hist(y, bins=20, orientation="horizontal")
        ax_y.set_xlim(ax_y.get_xlim()[::-1])
        plt.title(title)
        

    def KL_objective(self):
        return 0.5 * tf.reduce_mean(
            tf.square(self.bijector(self.x))
        ) - tf.reduce_mean(
            self.bijector.forward_log_det_jacobian(self.x, event_ndims=0)
        )

    def kstest(self):
        # Kolmogorov-Smirnov test for normality of transformed data
        s, pvalue = stats.kstest(self.bijector(self.x).numpy()[:,0], "norm")
        print("KS test statistic is %.3f, p-value is %.8f" % (s, pvalue))
        return s, pvalue

# -----------------------------------------------------------------------------
# Normaliser 2-D — minimal extension of the 1-D upstream code -----------------
# -----------------------------------------------------------------------------
class Normalizer2D(gpflow.base.Module):
    """2‑D extension of Amazon OAK *Normalizer* **with optional decorrelation**.

    * By default (`decorrelate=False`) it behaves exactly like the upstream
      1‑D normaliser applied independently to x₁ and x₂.
    * With `decorrelate=True` it appends an **affine whitening bijector** based
      on the sample mean and Cholesky of the marginal covariance computed *after*
      the two 1‑D flows.  This makes the output approximately 𝒩(0, I).
    """

    def __init__(self, x, log=(True, True), decorrelate: bool = True,
                 eps: float = SMALL, name="normalizer2d", **kwargs):
        super().__init__(name=name, **kwargs)

        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == 2, "Expected input shape (N, 2)."
        self.x = x
        self.eps = eps
        self.decorrelate = decorrelate

        # ------------------------------------------------------------------
        # 1. Per‑dimension 1‑D flows (identical to upstream logic) ----------
        # ------------------------------------------------------------------
        bijectors = []
        for d in range(2):
            xd = x[:, d]
            if log[d]:
                offset = np.min(xd) - 1.0 - eps  # ensure positive inside Log
                chain = tfb.Chain([
                    *_standardiser(np.log(xd - offset), eps),
                    tfb.Log(),
                    tfb.Shift(-offset),
                    tfb.SinhArcsinh(
                        skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                        tailweight=gpflow.Parameter(1.0, transform=tfb.Exp(), dtype=DTYPE),
                    ),
                ], name=f"flow_dim{d}")
            else:
                chain = tfb.Chain([
                    *_standardiser(xd, eps),
                    tfb.SinhArcsinh(
                        skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                        tailweight=gpflow.Parameter(1.0, transform=tfb.Exp(), dtype=DTYPE),
                    ),
                ], name=f"flow_dim{d}")
            bijectors.append(chain)

        block = tfb.Blockwise(bijectors, block_sizes=[1, 1], name="per_dim_block")
        self._block = block  # keep around for debugging / access

        # ------------------------------------------------------------------
        # 2. Optional decorrelation (fixed affine) -------------------------
        # ------------------------------------------------------------------
        if decorrelate:
            z0 = block.forward(x).numpy()
            mean = z0.mean(axis=0)
            cov = np.cov(z0.T) + eps * np.eye(2)
            chol = np.linalg.cholesky(cov)
            chol_inv = np.linalg.inv(chol)

            self._mean = mean  # for reference
            self._chol_inv = chol_inv

            centre = tfb.Shift(-mean, name="centre")
            whiten = tfb.ScaleMatvecTriL(chol_inv, name="whiten")
            self.bijector = tfb.Chain([whiten, centre, block], name="flow_2d_decor")
        else:
            self.bijector = block

    # ------------------------------------------------------------------
    # Public API (unchanged) -------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, z):
        return self.bijector.forward(tf.cast(z, DTYPE))

    def inverse(self, y):
        return self.bijector.inverse(tf.cast(y, DTYPE))

    def log_det_jacobian(self, z):
        return self.bijector.forward_log_det_jacobian(tf.cast(z, DTYPE), event_ndims=1)

    # ------------------------------------------------------------------
    # Diagnostics -------------------------------------------------------
    # ------------------------------------------------------------------
    def KL_objective(self):
        z = self.x.astype(np.float64)
        y = self.forward(z)
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(y), axis=-1)) \
            - tf.reduce_mean(self.log_det_jacobian(z))

    def kstest(self):
        y = self.forward(self.x).numpy()
        test_results = []
        for d in range(2):
            s, p = stats.kstest(y[:, d], "norm")
            test_results.append((s, p))
            print(f"Dim {d}: KS stat = {s:.3f}, p = {p:.3g}")
        return test_results
        

    def plot(self, title='Normaliser 2D'):
        f = plt.figure()
        ax = f.add_axes([0.3, 0.3, 0.65, 0.65])
        x = self.x
        y = self.forward(x).numpy()
        ax.plot(x[:, 0], y[:, 0], "k.", label="Gaussian dim 1")
        ax.plot(x[:, 1], y[:, 1], "r.", label="Gaussian dim 2")
        ax.legend()

        ax_x = f.add_axes([0.3, 0.05, 0.65, 0.25], sharex=ax)
        ax_x.hist(x[:, 0], bins=20)
        ax_y = f.add_axes([0.05, 0.3, 0.25, 0.65], sharey=ax)
        ax_y.hist(y[:, 0], bins=20, orientation="horizontal")
        ax_y.set_xlim(ax_y.get_xlim()[::-1])
        plt.title(title)