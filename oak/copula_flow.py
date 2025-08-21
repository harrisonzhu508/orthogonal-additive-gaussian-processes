# normalizer_copula.py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats
import gpflow

tfd, tfb = tfp.distributions, tfp.bijectors
DTYPE = gpflow.default_float()
SMALL = 1e-6

def _standardiser(x, eps):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std  = np.std(x) + eps
    return [
        tfb.Scale(tf.constant(1.0 / std, dtype=DTYPE), name="scale"),
        tfb.Shift(tf.constant(-mean,     dtype=DTYPE), name="shift"),
    ]

class NormalizerGeneralizedCopula(gpflow.base.Module):
    """
    D-dimensional normalizer with:
      (1) per-dim marginal flows (gaussianize each coord),
      (2) optional fixed linear whitening (decorrelate=True),
      (3) optional learned copula flow (MAF stack) to remove nonlinear dependence.

    The pipeline is fully invertible; you can transform and inverse-transform.
    You can also invert a *single variable* only through the marginal flow
    (before whitening/copula), via inverse_marginal(...).

    x=Xtr_h,                        # <--- numpy array here
    log=[False]*len(hum_dims),
    decorrelate=True,
    use_copula_flow=True,
    n_maf=3, hidden_units=(128,128), activation="tanh"

    """

    def __init__(self,
                 x,
                 log=None,
                 decorrelate: bool = True,
                 use_copula_flow: bool = True,
                 n_maf: int = 3,
                 hidden_units=(128, 128),
                 activation="tanh",
                 eps: float = SMALL,
                 name="normalizer_copula",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        # --- accept both NumPy and TF arrays ---
        if isinstance(x, tf.Tensor):
            x = x.numpy()                      # convert to numpy if TF tensor
        x = np.asarray(x, dtype=np.float64)     # ensure double precision for stats
        self.x = x
        self.eps = eps

        D = x.shape[1]
        if log is None:
            log = [False] * D

        assert x.ndim == 2, "Expected input shape (N, D)."
        self.x = x
        self.eps = eps
        self.decorrelate = decorrelate
        self.use_copula_flow = use_copula_flow
        self.n_maf = int(n_maf)
        self.hidden_units = tuple(hidden_units)
        self.activation = activation

        D = x.shape[1]
        if log is None:
            log = [False] * D
        assert len(log) == D, "Length of `log` must match number of dimensions"

        # ------------------------------------------------------------------
        # 1) Per-dimension 1D marginal flows (frozen; monotone)
        # ------------------------------------------------------------------
        per_dim = []
        self._per_dim_offsets = [0.0] * D  # just for reference if you used logs
        for d in range(D):
            xd = x[:, d]
            if log[d]:
                # shift to positive, log, standardise, then a shape tweak
                offset = float(np.min(xd) - 1.0 - eps)
                self._per_dim_offsets[d] = offset
                chain = tfb.Chain([
                    *_standardiser(np.log(xd - offset), eps),
                    tfb.Log(name="log"),
                    tfb.Shift(-offset, name="unshift_to_zero"),
                    tfb.SinhArcsinh(
                        skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                        tailweight=gpflow.Parameter(1.0, transform=tfb.Exp(), dtype=DTYPE),
                        name="sinh_arcsinh",
                    ),
                ], name=f"flow_dim{d}")
            else:
                chain = tfb.Chain([
                    *_standardiser(xd, eps),
                    tfb.SinhArcsinh(
                        skewness=gpflow.Parameter(0.0, dtype=DTYPE),
                        tailweight=gpflow.Parameter(1.0, transform=tfb.Exp(), dtype=DTYPE),
                        name="sinh_arcsinh",
                    ),
                ], name=f"flow_dim{d}")
            per_dim.append(chain)

        block = tfb.Blockwise(per_dim, block_sizes=[1] * D, name="per_dim_block")
        self._block = block  # z = block(x)

        # ------------------------------------------------------------------
        # 2) Optional fixed affine whitening (linear decorrelation)
        # ------------------------------------------------------------------
        pieces = [block]
        self._centre = None
        self._whiten = None
        if decorrelate:
            z0 = block.forward(x).numpy()
            mean = z0.mean(axis=0)
            cov  = np.cov(z0.T) + eps * np.eye(D)
            chol = np.linalg.cholesky(cov)
            chol_inv = np.linalg.inv(chol)

            self._mean = mean
            self._chol_inv = chol_inv

            centre = tfb.Shift(-mean.astype(np.float64), name="centre")
            whiten = tfb.ScaleMatvecTriL(chol_inv.astype(np.float64), name="whiten")
            # In forward(): y = (... copula ...) ∘ whiten ∘ centre ∘ block (right-to-left)
            pieces = [whiten, centre] + pieces
            self._centre, self._whiten = centre, whiten

        # ------------------------------------------------------------------
        # 3) Optional learned copula flow (MAF stack; autoregressive bijector)
        # ------------------------------------------------------------------
        self._maf_stack = None
        if use_copula_flow and self.n_maf > 0:
            mafs = []
            for k in range(self.n_maf):
                made = tfb.AutoregressiveNetwork(
                    params=2,
                    hidden_units=list(self.hidden_units),
                    activation=self.activation,
                    kernel_initializer="glorot_uniform",
                    name=f"made_{k}",
                )
                maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made, name=f"maf_{k}")
                # Permute between layers to improve mixing
                perm = tfb.Permute(permutation=list(reversed(range(D))), name=f"perm_{k}")
                # Compose as (... maf ∘ perm ∘ ...) ∘ whiten ∘ centre ∘ block
                mafs.extend([maf, perm])
            self._maf_stack = tfb.Chain(mafs, name="copula_flow")
            pieces = [self._maf_stack] + pieces

        # Final bijector & base
        self.bijector = tfb.Chain(pieces, name="full_flow")
        self._base = tfd.MultivariateNormalDiag(loc=tf.zeros(D, DTYPE), scale_diag=tf.ones(D, DTYPE))

    # ----------------------------------------------------------------------
    # Forward / inverse
    # ----------------------------------------------------------------------
    def forward(self, x):
        """x -> y, aiming for y ~ iid N(0,1)."""
        return self.bijector.forward(tf.cast(x, DTYPE))

    def inverse(self, y):
        """Full inverse: y (≈N(0,1)) -> x (original scale)."""
        return self.bijector.inverse(tf.cast(y, DTYPE))

    def log_det_jacobian(self, x):
        """log |det J_f(x)| for f = full_flow."""
        return self.bijector.forward_log_det_jacobian(tf.cast(x, DTYPE), event_ndims=1)

    # Marginal-only helpers (before whitening/copula)
    def forward_marginals(self, x):
        """Apply only per-dim marginal flows: z = block(x)."""
        return self._block.forward(tf.cast(x, DTYPE))

    def inverse_marginal(self, z_i, dim: int):
        """Invert the 1-D marginal flow for a single variable (before whitening/copula)."""
        # Build a 1-D dummy tensor and run inverse of that dim only
        z_i = tf.convert_to_tensor(z_i, dtype=DTYPE)
        # Invert per-dim chain: x_i = chain^{-1}(z_i)
        return self._block.bijectors[dim].inverse(z_i)

    # ----------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------
    def KL_objective(self, x=None):
        """Proxy KL to N(0,I): E[0.5||y||^2] - E[log|detJ|] (up to constants). Lower is better."""
        if x is None:
            x = self.x
        x = tf.convert_to_tensor(x, dtype=DTYPE)
        y = self.forward(x)
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(y), axis=-1)) - tf.reduce_mean(self.log_det_jacobian(x))

    def kstest(self, x=None):
        """Univariate KS test vs N(0,1) after full transform (per coordinate)."""
        if x is None:
            x = self.x
        y = self.forward(x).numpy()
        res = []
        for d in range(y.shape[1]):
            s, p = stats.kstest(y[:, d], "norm")
            res.append((s, p))
            print(f"Dim {d}: KS stat = {s:.3f}, p = {p:.3g}")
        return res

    def pairwise_corr(self, x=None):
        """Sample correlation matrix after full transform (should be ~I if near-independent)."""
        if x is None:
            x = self.x
        y = self.forward(x).numpy()
        return np.corrcoef(y.T)

    # ----------------------------------------------------------------------
    # Training only the copula flow (MAF) — marginals/whitening stay fixed
    # ----------------------------------------------------------------------
    @tf.function
    def _nll_step(self, x_batch, optimizer):
        with tf.GradientTape() as tape:
            y = self.forward(x_batch)
            log_det = self.log_det_jacobian(x_batch)
            log_p = self._base.log_prob(y) + log_det
            nll = -tf.reduce_mean(log_p)
        vars_trainable = []
        if self._maf_stack is not None:
            for b in self._maf_stack.bijectors:
                vars_trainable += b.trainable_variables
        grads = tape.gradient(nll, vars_trainable)
        optimizer.apply_gradients(zip(grads, vars_trainable))
        return nll

    def fit_flow(self, x_train=None, epochs=200, batch_size=1024, lr=1e-3, verbose=True, seed=0):
        """
        Maximum-likelihood training of the copula flow only.
        Per-dim flows & whitening are fixed (not updated).
        """
        if not (self.use_copula_flow and self._maf_stack is not None):
            if verbose:
                print("Nothing to train: use_copula_flow is False or n_maf=0.")
            return

        if x_train is None:
            x_train = self.x
        x_train = tf.convert_to_tensor(x_train, dtype=DTYPE)
        n = tf.shape(x_train)[0]
        rng = tf.random.Generator.from_seed(seed)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf.function
        def get_batch():
            idx = rng.uniform(shape=(batch_size,), maxval=n, dtype=tf.int32)
            return tf.gather(x_train, idx)

        for e in range(1, epochs + 1):
            xb = get_batch()
            nll = self._nll_step(xb, opt)
            if verbose and (e == 1 or e % max(1, epochs // 10) == 0 or e == epochs):
                print(f"[{e:04d}/{epochs}] NLL: {float(nll):.4f} | KL proxy: {float(self.KL_objective(x_train)):.4f}")

# ---------- Distance correlation (Szekely et al.) ----------
def _pdist_centered(x):
    # x: (n,1)
    x = np.asarray(x, float)
    n = x.shape[0]
    D = np.abs(x - x.T)
    A = D - D.mean(axis=0, keepdims=True) - D.mean(axis=1, keepdims=True) + D.mean()
    return A

def distance_correlation(x, y):
    # x,y: (n,) vectors
    x = np.asarray(x, float).reshape(-1,1)
    y = np.asarray(y, float).reshape(-1,1)
    A = _pdist_centered(x)
    B = _pdist_centered(y)
    dcov2_xy = (A*B).mean()
    dcov2_xx = (A*A).mean()
    dcov2_yy = (B*B).mean()
    denom = np.sqrt(dcov2_xx*dcov2_yy) + 1e-12
    return np.sqrt(max(dcov2_xy, 0.0)) / denom

def dcor_matrix(X):
    # X: (n, d)
    n, d = X.shape
    M = np.zeros((d,d), float)
    for i in range(d):
        M[i,i] = 1.0
        for j in range(i+1, d):
            val = distance_correlation(X[:,i], X[:,j])
            M[i,j] = M[j,i] = val
    return M

# ---------- Gaussian-kernel HSIC (unbiased) with median heuristic ----------
def _rbf_kernel(z, sigma=None):
    # z: (n,1) array
    z = np.asarray(z, float).reshape(-1,1)
    sq = (z - z.T)**2
    if sigma is None:
        # median heuristic
        med = np.median(np.sqrt(np.abs(sq + np.triu(sq,1))))
        sigma = med if med > 0 else np.std(z) + 1e-12
    K = np.exp(-sq / (2*sigma**2))
    return K

def hsic_unbiased(x, y):
    # x,y: (n,) vectors
    x = np.asarray(x, float).reshape(-1,1)
    y = np.asarray(y, float).reshape(-1,1)
    n = x.shape[0]
    K = _rbf_kernel(x)
    L = _rbf_kernel(y)
    # Unbiased HSIC (Song et al., 2012)
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    term1 = (K*L).sum() / (n*(n-3))
    term2 = (K.sum(axis=0)*L.sum(axis=0)).sum() / (n*(n-3)*(n-1))
    term3 = (K.sum()*L.sum()) / (n*(n-1)*(n-2)*(n-3))
    hsic = term1 + term3 - 2*term2
    return max(hsic, 0.0)

def hsic_perm_test(x, y, n_perm=200, rng=None):
    # returns (hsic, p_value)
    if rng is None:
        rng = np.random.default_rng(0)
    obs = hsic_unbiased(x, y)
    n = len(x)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        val = hsic_unbiased(x, y[perm])
        count += (val >= obs)
    p = (count + 1) / (n_perm + 1)  # add-one smoothing
    return obs, p

# ---------- Convenience: run on your transformed block ----------
def independence_report(Z, check_hsic_topk=5, n_perm=200):
    """
    Z: (n, d) transformed features (e.g., Ztr_h)
    Returns: dCor matrix, and HSIC results for top dependent pairs.
    """
    M = dcor_matrix(Z)
    # rank pairs by dCor (excluding diagonal)
    d = Z.shape[1]
    pairs = []
    for i in range(d):
        for j in range(i+1, d):
            pairs.append((M[i,j], i, j))
    pairs.sort(reverse=True)
    top = pairs[:check_hsic_topk]

    hsic_results = []
    for _, i, j in top:
        hsic, p = hsic_perm_test(Z[:,i], Z[:,j], n_perm=n_perm)
        hsic_results.append({'i': i, 'j': j, 'hsic': hsic, 'p': p, 'dcor': M[i,j]})
    return M, hsic_results

# Ztr_h = hum_flow.forward(tf.convert_to_tensor(Xtr_h, dtype=default_float())).numpy()
# dcorM, hsic_top = independence_report(Ztr_h, check_hsic_topk=6, n_perm=200)
# print("Max off-diagonal dCor:", np.max(dcorM - np.eye(dcorM.shape[0])))
# for r in hsic_top:
#     print(r)
# # Heatmap:
# import matplotlib.pyplot as plt
# plt.imshow(dcorM, vmin=0, vmax=1); plt.colorbar(); plt.title("Distance correlation")
# plt.show()