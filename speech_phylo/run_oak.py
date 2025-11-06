import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.kernels import Kernel
from pathlib import Path
from typing import List, Union
import numpy as np
import geopandas as gpd
from oak.model_utils import oak_model, save_model, load_model
# run_gpflow_shap.py
import os
import sys 
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from gpflow.config import default_float
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

class PhyloKernel(Kernel):
    """
    A phylogenetic covariance kernel with *learnable* Pagel’s λ and a scale σ².
    Inputs X should have a column of integer tip-indices (0…n-1)
    specified via `active_dims` (default = [0]).
    """

    def __init__(
        self,
        V: Union[tf.Tensor, tf.Variable, np.ndarray],
        active_dims: List[int] = [0],
        **kwargs
    ):
        """
        :param V:           (n×n) raw shared‐branch‐length matrix.
        :param active_dims: which column(s) of X hold the integer tip index.
        """
        super().__init__(**kwargs)
        # Store the raw phylogenetic covariance as a tf.constant
        self.V = tf.constant(V, dtype=tf.float64)

        # Learnable amplitude σ² ≥ 0
        self.variance = gpflow.Parameter(
            1.0,
            transform=tfp.bijectors.Softplus(),
            name="variance"
        )

        # Learnable Pagel's λ ∈ (0,1)
        self.lambda_raw = gpflow.Parameter(
            0.5,
            transform=tfp.bijectors.Sigmoid(),
            name="lambda"
        )

        # Which column(s) of X encode the tip‐index
        self.active_dims = active_dims

    @property
    def lambda_(self) -> tf.Tensor:
        """The actual λ ∈ (0,1) after Sigmoid transform."""
        return self.lambda_raw

    def V_lambda(self) -> tf.Tensor:
        """
        Build the λ‐adjusted covariance matrix:
          V_λ = λ*(V − diag(V)) + diag(V)
        """
        lam = self.lambda_
        # Extract diagonal of V
        D = tf.linalg.diag(tf.linalg.diag_part(self.V))
        # Scale off‐diagonals by λ, keep variances fixed
        return lam * (self.V - D) + D

    def K(self, X: tf.Tensor, X2: tf.Tensor = None) -> tf.Tensor:
        """
        Compute the Gram matrix on inputs X, X2.
        Column self.active_dims of X gives integer tip indices into V_λ.
        """
        # Extract tip‐indices from X and (optionally) X2
        idx1 = tf.cast(tf.reshape(tf.gather(X, self.active_dims, axis=1), [-1]), tf.int32)
        if X2 is None:
            idx2 = idx1
        else:
            idx2 = tf.cast(tf.reshape(tf.gather(X2, self.active_dims, axis=1), [-1]), tf.int32)

        # Build the λ‐adjusted covariance
        V_l = self.V_lambda()             # shape [n, n]
        # Gather rows and columns for the requested indices
        rows = tf.gather(V_l, idx1, axis=0)      # shape [N, n]
        W    = tf.gather(rows, idx2, axis=1)     # shape [N, M]
        # Multiply by the learnable scale σ²
        return self.variance * W

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """
        Diagonal elements of K(X,X):
          σ² * diag(V_λ)[tip-index]
        """
        idx = tf.cast(tf.reshape(tf.gather(X, self.active_dims, axis=1), [-1]), tf.int32)
        V_l = self.V_lambda()                      # [n, n]
        d   = tf.gather(tf.linalg.diag_part(V_l), idx)  # [N]
        return self.variance * d

import pandas as pd
import numpy as np
dir= "results_clean2_bayesian"
# dir= "results_clean2_nospeakers"

# for empirical_spatial_measure in [False, True]:
# for empirical_spatial_measure in [False]:
# for empirical_spatial_measure in [True]:
#     for tree_name in ["heggarty2024", "long_v2_CCD0.0.05", "long_v3_CCD0.0.10", "long_v3_CCD0.0.25"]:
#         for mercator in [False, True]:
#             for add_phylocov in [False, True]:
for empirical_spatial_measure in [True]:
    for tree_name in ["heggarty2024", "long_v3_CCD0.0.25"]:
        for mercator in [False]:
            for add_phylocov in [True]:
                # ─── 1) Load V_raw (this part was fine) ─────────────
                V_df = pd.read_csv(f"speech_phylo/{dir}/V_raw_{tree_name}.csv", index_col=0)
                assert all(V_df.index == V_df.columns)
                languages = V_df.index.tolist()
                V = V_df.values.astype(np.float64)

                # ─── 2) Load metadata, letting pandas read the header row ────
                meta = pd.read_csv(f"speech_phylo/{dir}/metadata_{tree_name}.csv", sep=",").reset_index()
                meta = meta.rename(columns={"level_0": "language_true"})
                # meta.drop(columns=meta.columns[1:7], errors='ignore')
                # meta = meta.drop(columns=meta.columns[1:7], errors='ignore')
                meta = meta.rename(columns={"language_true": "language"})

                # If the column really is called "language", do:
                if "language" not in meta.columns:
                    raise ValueError("Your metadata CSV does not have a column named 'language'")

                # Trim whitespace just in case
                meta["language"] = meta["language"].astype(str).str.strip()

                # Now set it as the index
                meta = meta.set_index("language")
                # ─── 3) Re-order the rows to match your V matrix ──────────
                # This will fail loudly if any languages are missing
                meta = meta.loc[languages]

                if mercator:
                    # ─── Reproject lat/lon to metric CRS ─────────────────────────
                    # Create GeoDataFrame in EPSG:4326 (lat/lon)
                    gdf = gpd.GeoDataFrame(
                        meta,
                        geometry=gpd.points_from_xy(meta["longitude"], meta["latitude"]),
                        crs="EPSG:4326"
                    )

                    # Choose your target projection (example: Web Mercator)
                    gdf_proj = gdf.to_crs(epsg=3857)  # you can change to EPSG:25832 for Denmark, etc.

                    # Replace longitude/latitude columns with projected coords in meters
                    meta["x"] = gdf_proj.geometry.x
                    meta["y"] = gdf_proj.geometry.y

                # ─── 4) Build X as before ───────────────────────────────
                longitudes   = meta["longitude"].to_numpy().reshape(-1, 1)
                latitudes    = meta["latitude"].to_numpy().reshape(-1, 1)
                log_speakers = np.log(meta["n_speakers"].to_numpy()).reshape(-1, 1)
                # hittite = meta["Hittite"].to_numpy().reshape(-1, 1)

                n       = len(languages)
                tip_idx = np.arange(n).reshape(-1, 1)
                if mercator:
                    # X       = np.hstack([meta["x"].to_numpy().reshape(-1, 1), meta["y"].to_numpy().reshape(-1, 1), log_speakers, hittite, tip_idx])
                    X       = np.hstack([meta["x"].to_numpy().reshape(-1, 1), meta["y"].to_numpy().reshape(-1, 1), log_speakers, tip_idx])
                else:
                    # X       = np.hstack([longitudes, latitudes, log_speakers, hittite, tip_idx])
                    X       = np.hstack([longitudes, latitudes, log_speakers, tip_idx])
                y = meta["rate_median"].to_numpy().reshape(-1, 1)
                y = np.log(y)
                print(y.shape)


                # -----------------------------------------------------------------------------
                # CLI
                # -----------------------------------------------------------------------------
                seed = 0
                np.random.seed(seed)
                # -----------------------------------------------------------------------------
                # Load data
                # -----------------------------------------------------------------------------


                # -----------------------------------------------------------------------------
                # OAK definition 
                # for interaction in [False, True]:
                for interaction in [True]:
                    if interaction:
                        max_interaction_depth=2
                    else:
                        max_interaction_depth=1

                    # for block_latlon in [False, True]:
                    for block_latlon in [False]:
                        if block_latlon:
                            active_dims = [
                                [0, 1], # latlon or xy
                                [2], # log_speakers
                                # [3], # hittite
                                # [3]  # tip index
                            ]

                            # get list of variable names
                            variable_names = [
                                "xy" if mercator else "lonlat",
                                "log_speakers",
                                # "hittite",
                                # "lon",
                                # "lat",
                                # "log_speakers",
                                # "hittite",
                            ]
                        

                        else:
                            active_dims = [
                                [0], # latlon or xy
                                [1],
                                [2], # log_speakers
                            ]

                            # get list of variable names
                            variable_names = [
                                "x" if mercator else "lon",
                                "y" if mercator else "lat",
                                "log_speakers",
                            ]
                            final_active_dims = active_dims

                        flattened_active_dims = [item for sublist in active_dims for item in sublist]

                        # reindex active_dims to account for removed dimensions
                        final_active_dims = []
                        final_variable_names = []
                        ind = 0
                        for i, this_active_dims in enumerate(active_dims):
                            new_active_dims = []
                            for d in this_active_dims:
                                new_active_dims.append(ind)
                                ind += 1
                            final_active_dims.append(new_active_dims)
                        base_kernels = [
                            gpflow.kernels.RBF for _ in range(len(active_dims))
                        ]

                        if add_phylocov:
                            noise_kernel = PhyloKernel(V, active_dims=[3])  # tip index
                        else:
                            noise_kernel = None


                        assert len(base_kernels) == len(active_dims), "Base kernels and active dimensions must match in length."


                        if empirical_spatial_measure:
                            if block_latlon is True:
                                empirical_measure = [ [0,1] ]
                            else:
                                empirical_measure = [[0], [1]]
                        else:
                            empirical_measure = None

                        oak = oak_model(
                            num_inducing=500,
                            max_interaction_depth=max_interaction_depth,
                            use_sparsity_prior=True,
                            sparse=False,
                            base_kernels=base_kernels,
                            active_dims=final_active_dims,
                            noise_kernel=noise_kernel,
                            lam_concurvity=0.0,
                            empirical_measure=empirical_measure,
                        )

                        oak.fit(
                            X,
                            y,
                            initialise_inducing_points=True,
                            # optimise=False,
                            optimise=True,
                        )

                        # save fitted model parameters for reuse
                        model_dir = Path(f"speech_phylo/final_figures/models")
                        model_stub = (
                            f"oak_model_{tree_name}_mercator{mercator}_interaction{interaction}"
                            f"_blocklatlon{block_latlon}_add_phylocov{add_phylocov}"
                            f"_empiricalspatial{empirical_spatial_measure}"
                        )
                        save_model(oak.m, filename=model_dir / model_stub)

                        y_pred = oak.predict(X)
                        # compute R_squared value
                        # reshape y and y_pred
                        y_true = y.reshape(-1).copy()
                        y_pred = y_pred.reshape(-1)
                        ss_res = np.sum((y_true - y_pred) ** 2)
                        ss_tot = np.sum((y_true - np.mean(y)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)

                        sobol_df = oak.sobol_summary(covariate_names=variable_names, likelihood_variance=True, return_variance=True)
                        sobol_df['sobol_index'] = sobol_df['sobol_index'].apply(lambda x: f"{x:.8f}")

                        shap = oak.get_shapley() #  numpy array, save shap to csv with variable names. 1 column being shap and 1 column being name
                        shap_df = pd.DataFrame({
                            "variable": variable_names,
                            "shapley_value": shap
                        })
                        shap_df['shapley_value'] = shap_df['shapley_value'].apply(lambda x: f"{x:.8f}")
                        shap_df.to_csv(f"speech_phylo/{dir}/oak_SHAPLEY_{tree_name}_mercator{mercator}_interaction{interaction}_blocklatlon{block_latlon}_add_phylocov{add_phylocov}_empiricalspatial{empirical_spatial_measure}.csv", index=False)

                        sobol_df.to_csv(f"speech_phylo/{dir}/oak_sobol_{tree_name}_mercator{mercator}_interaction{interaction}_blocklatlon{block_latlon}_add_phylocov{add_phylocov}_empiricalspatial{empirical_spatial_measure}.csv", index=False)
                        # save oak.m.noise_kernel.lambda_
                        if add_phylocov:
                            with open(f"speech_phylo/{dir}/oak_metrics_{tree_name}_mercator{mercator}_interaction{interaction}_blocklatlon{block_latlon}_add_phylocov{add_phylocov}_empiricalspatial{empirical_spatial_measure}.json", "w") as f:
                                json.dump({"lambda": oak.m.kernel.noise_kernel.lambda_.numpy().tolist(), "r_squared": r_squared}, f)