## Final Plots Notebook

The notebook assembles the figures for the speech phylogenetics analysis. It loads Bayesian phylogenetic regression outputs and OAK Sobol indices, builds geographic Gaussian Process maps, and exports publication figures to `final_figuresv2/`.

### Required inputs

- `final_phyloregression_results/`
  - `phylolm_*{tree}*.csv`: Bayesian phylo regression summaries; `tree` is one of `heggarty2024` or `long_v3_CCD0.0.25`. 
  - `oak_sobol_{tree}.csv`: Sobol indices from the OAK model (optional but needed for Sobol plots).
  - `metadata_{tree}.csv`: Per-language metadata with longitude/latitude, speaker counts, and posterior rates; columns with `x,y` comma pairs are split into `_lo/_hi`.
- Tree posterior files: `long_v3_44.trees` (speech) and `heggarty2024_fleurs_v4_889.trees` (cognates) for the rate-through-time plot.
- Delta/network stats: `final_figuresv2/cognates_delta.txt` and `final_figuresv2/speech_20pc_delta-1762509996545.txt` (tab-separated, id/name/delta/Q-residual).
- Language polygons: `dataset.geojson` (language areas; some names are renamed or split/merged inside the notebook). This is from the Nature data paper https://www.nature.com/articles/s41597-025-05828-6 and https://zenodo.org/records/15287258 under `raw/dataset.geojson`
- Font: `nimbus-sans-l/NimbusSanL-Regu.ttf` is added for the delta plot.
- External basemap: `geopandas` uses the built-in `naturalearth_lowres` dataset.

### What the notebook produces (saved in `final_figuresv2/`)

- `effects_comparison_{standard,log_plus_10,sqrt_plus_10}.svg`: Forest plots of standardized Bayesian regression coefficients per tree.
- `oak_stacked_by_tree.svg`: Stacked horizontal bar chart of OAK variance decomposition component and tree.
- `country_map_eu_india_cont_{tree}_posterior_product.(png|pdf)`: GP posterior mean × (1 − SD) over Europe/India using point observations.
- `rate_vs_longitude_{tree}.svg`: Scatter of log rate vs. longitude with CIs, colored by number of speakers.
- `rate_over_time_normalized.svg`: Time-sliced posterior rates from tree posteriors (speech vs. cognates) with density-weighted bands.
- `waveforms_pack.(png|svg)` and `waveform_XX.(png|svg)`: Stylized audio-waveform icons.
- `distance_matrix.png` and `binary_sequences.png`: Demo distance heatmap and binary-sequence raster.
- `delta_ranked_by_family.svg`: Delta scores ranked per language, colored by family (shared colors across speech/cognates).
- `language_polygons_continuous_{tree}.svg`: GP map constrained to language polygons (`dataset.geojson`).
- `language_polygons_gridtrain_{tree}.svg`: Variant GP map trained on polygon-gridded pseudo-observations.

### Notebook flow

1. Load Bayesian regression outputs and OAK Sobol indices from `final_phyloregression_results/`, normalize tree labels (`heggarty2024` → Cognates, `long_v3_CCD0.0.25` → Speech (25%)), and plot coefficient comparisons.
2. Plot Sobol variance fractions as stacked bars by tree.
3. Fit GP models on per-language rates and render posterior-product maps over Europe/India; export PNG/PDF per tree.
4. Scatter log rate vs. longitude with speaker-count coloring and CIs.
5. Rate-through-time comparison using posterior tree lists (`long_v3_44.trees`, `heggarty2024_fleurs_v4_889.trees`).
6. Generate auxiliary assets (waveform icons, distance matrix, binary sequences).
7. Plot delta rankings by family using the delta text files and Nimbus Sans font.
8. Additional continuous-map experiments using `dataset.geojson` to clip/weight GP predictions within language polygons (two variants saved per tree).

## Running notes for `final_plots.ipynb` and `final_scripts` scripts

- Start the kernel from `speech_phylo/` so relative paths resolve.
- R deps: `install.packages(c("ape", "brms", "dplyr", "tibble", "jsonlite", "doParallel", "foreach", "tidyr", "TreeDist")) # make sure RStan/Stan toolchain is configured for brms`
- To run the scripts, you need to be in the **parent path** of `speech_phylo/` and run `python speech_phylo/final_scripts/run_oak.py` or `Rscript speech_phylo/final_scripts/run_clean_bayesian_transformations.R` as needed.
- The OAK fork (modified from original) will also need to be installed in the environment for running `run_oak.py`: https://github.com/harrisonzhu508/orthogonal-additive-gaussian-processes/tree/my-oak-improvements. Python deps can be installed via:

```
pip install --upgrade pip
# Core + geo + GP stack
pip install numpy pandas matplotlib seaborn tqdm geopandas shapely scikit-learn \
            tensorflow tensorflow_probability gpflow dendropy jupyter ipython

# Install the OAK fork in editable mode so `oak.model_utils` is importable
pip install -e . # in ../speech_phylo (parent of this README)
```

### Full installation instructions

```
# =============================================================================
# 1. Download and install Miniconda
# =============================================================================

cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Initialize conda for your shell
~/miniconda3/bin/conda init bash
source ~/.bashrc

# =============================================================================
# 2. Create and activate environment
# =============================================================================

conda create -n speech_phylo python=3.7 -y
conda activate speech_phylo

# =============================================================================
# 3. Install Python dependencies
# =============================================================================

pip install --upgrade pip

# Core + geo + GP stack
pip install numpy pandas matplotlib seaborn tqdm geopandas shapely scikit-learn \
            tensorflow tensorflow_probability gpflow dendropy jupyter ipython

# =============================================================================
# 4. Clone and install the OAK fork
# =============================================================================

cd ~
cd orthogonal-additive-gaussian-processes

# Install OAK in editable mode
pip install -e .

# =============================================================================
# 5. Install R and R dependencies
# =============================================================================

# Install R via conda (easier than system R)
conda install -c conda-forge r-base r-essentials -y

# Install R packages
R -e 'install.packages(c("ape", "brms", "dplyr", "tibble", "jsonlite", "doParallel", "foreach", "tidyr", "TreeDist"), repos="https://cloud.r-project.org")'

# =============================================================================
# 7. Running the scripts
# =============================================================================

# For R Bayesian script:
Rscript speech_phylo/final_scripts/run_clean_bayesian_transformations.R

# For Python OAK script:
python speech_phylo/final_scripts/run_oak.py


# For Jupyter notebook - start kernel from speech_phylo/ directory:
cd ~/speech_phylo
jupyter notebook final_plots.ipynb
```