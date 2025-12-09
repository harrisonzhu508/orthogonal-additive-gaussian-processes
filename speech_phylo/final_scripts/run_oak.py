import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.kernels import Kernel
from pathlib import Path
from typing import List, Union
import numpy as np
import geopandas as gpd
from oak.model_utils import oak_model, save_model, load_model
import os
import sys 
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import pandas as pd
from gpflow.config import default_float

class PhyloKernel(Kernel):
    """
    A phylogenetic covariance kernel with *learnable* Pagel's λ and a scale σ².
    Inputs X should have a column of integer tip-indices (0…n-1)
    specified via `active_dims` (default = [0]).
    """

    def __init__(
        self,
        V: Union[tf.Tensor, tf.Variable, np.ndarray],
        active_dims: List[int] = [0],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.V = tf.constant(V, dtype=tf.float64)
        self.variance = gpflow.Parameter(
            1.0,
            transform=tfp.bijectors.Softplus(),
            name="variance"
        )
        self.lambda_raw = gpflow.Parameter(
            0.5,
            transform=tfp.bijectors.Sigmoid(),
            name="lambda"
        )
        self.active_dims = active_dims

    @property
    def lambda_(self) -> tf.Tensor:
        return self.lambda_raw

    def V_lambda(self) -> tf.Tensor:
        lam = self.lambda_
        D = tf.linalg.diag(tf.linalg.diag_part(self.V))
        return lam * (self.V - D) + D

    def K(self, X: tf.Tensor, X2: tf.Tensor = None) -> tf.Tensor:
        idx1 = tf.cast(tf.reshape(tf.gather(X, self.active_dims, axis=1), [-1]), tf.int32)
        idx2 = idx1 if X2 is None else tf.cast(tf.reshape(tf.gather(X2, self.active_dims, axis=1), [-1]), tf.int32)
        V_l = self.V_lambda()
        rows = tf.gather(V_l, idx1, axis=0)
        W = tf.gather(rows, idx2, axis=1)
        return self.variance * W

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        idx = tf.cast(tf.reshape(tf.gather(X, self.active_dims, axis=1), [-1]), tf.int32)
        V_l = self.V_lambda()
        d = tf.gather(tf.linalg.diag_part(V_l), idx)
        return self.variance * d


# ─── Configuration ───────────────────────────────────────────────────────────
dir = "final_phyloregression_results"
seed = 0
np.random.seed(seed)

tree_names = ["heggarty2024", "long_v3_CCD0.0.25"]

for tree_name in tree_names:
    # ─── 1) Load V_raw ───────────────────────────────────────────────────────
    V_df = pd.read_csv(f"speech_phylo/{dir}/V_raw_{tree_name}.csv", index_col=0)
    assert all(V_df.index == V_df.columns)
    languages = V_df.index.tolist()
    V = V_df.values.astype(np.float64)

    # ─── 2) Load metadata ────────────────────────────────────────────────────
    meta = pd.read_csv(f"speech_phylo/{dir}/metadata_{tree_name}.csv", sep=",").reset_index()
    meta = meta.rename(columns={"level_0": "language"})
    if "language" not in meta.columns:
        raise ValueError("Your metadata CSV does not have a column named 'language'")
    meta["language"] = meta["language"].astype(str).str.strip()
    meta = meta.set_index("language").loc[languages]

    # ─── 3) Build X and y ────────────────────────────────────────────────────
    n = len(languages)
    X = np.hstack([
        meta["longitude"].to_numpy().reshape(-1, 1),
        meta["latitude"].to_numpy().reshape(-1, 1),
        np.log(meta["n_speakers"].to_numpy()).reshape(-1, 1),
        np.arange(n).reshape(-1, 1),  # tip index
    ])
    y = np.log(meta["rate_median"].to_numpy().reshape(-1, 1))

    # ─── 4) OAK model setup ──────────────────────────────────────────────────
    active_dims = [[0], [1], [2]]
    variable_names = ["lon", "lat", "log_speakers"]

    oak = oak_model(
        num_inducing=500,
        max_interaction_depth=2,
        use_sparsity_prior=True,
        sparse=False,
        base_kernels=[gpflow.kernels.RBF for _ in range(3)],
        active_dims=active_dims,
        noise_kernel=PhyloKernel(V, active_dims=[3]),
        lam_concurvity=0.0,
        empirical_measure=[[0], [1]],
    )

    # ─── 5) Fit ──────────────────────────────────────────────────────────────
    oak.fit(X, y, initialise_inducing_points=True, optimise=True)

    # ─── 6) Save model ───────────────────────────────────────────────────────
    model_dir = Path(f"speech_phylo/final_phyloregression_results/models")
    model_stub = f"oak_model_{tree_name}"
    save_model(oak.m, filename=model_dir / model_stub)

    # ─── 7) Evaluate ─────────────────────────────────────────────────────────
    y_pred = oak.predict(X).reshape(-1)
    y_true = y.reshape(-1)
    r_squared = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y)) ** 2)

    # ─── 8) Sobol indices ────────────────────────────────────────────────────
    sobol_df = oak.sobol_summary(covariate_names=variable_names, likelihood_variance=True, return_variance=True)
    sobol_df['sobol_index'] = sobol_df['sobol_index'].apply(lambda x: f"{x:.8f}")
    sobol_df.to_csv(f"speech_phylo/{dir}/oak_sobol_{tree_name}.csv", index=False)

    # ─── 10) Save metrics ────────────────────────────────────────────────────
    with open(f"speech_phylo/{dir}/oak_metrics_{tree_name}.json", "w") as f:
        json.dump({
            "lambda": oak.m.kernel.noise_kernel.lambda_.numpy().tolist(),
            "r_squared": r_squared
        }, f)