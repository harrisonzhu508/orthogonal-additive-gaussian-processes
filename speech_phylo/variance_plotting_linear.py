#!/usr/bin/env python3
"""
Plotting module for linear interaction models (no spline).

This module provides visualization functions for variance decomposition
results from the linear lat/lon phylogenetic regression models.

Usage:
    python speech_phylo/variance_plotting_linear.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations for linear interaction models
MODEL_CONFIGS = {
    "linear_main_only": {
        "file_suffix": "_linear_main_only",
        "display_name": "Linear Main Effects Only",
        "components": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm",
            "Phylogenetic", "Residual"
        ],
        "coeff_order": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm"
        ],
    },
    "linear_secondorder": {
        "file_suffix": "_linear_secondorder",
        "display_name": "Linear with 2nd Order Interactions",
        "components": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm",
            "longitude_norm:latitude_norm",
            "longitude_norm:log_n_speakers_norm",
            "longitude_norm:n_segments_norm",
            "longitude_norm:delta_norm",
            "latitude_norm:log_n_speakers_norm",
            "latitude_norm:n_segments_norm",
            "latitude_norm:delta_norm",
            "log_n_speakers_norm:n_segments_norm",
            "log_n_speakers_norm:delta_norm",
            "n_segments_norm:delta_norm",
            "Phylogenetic", "Residual"
        ],
        "coeff_order": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm",
            "longitude_norm:latitude_norm",
            "log_n_speakers_norm:n_segments_norm",
            "log_n_speakers_norm:delta_norm",
            "n_segments_norm:delta_norm"
        ],
    },
    "linear_thirdorder": {
        "file_suffix": "_linear_thirdorder",
        "display_name": "Linear with 3rd Order Interactions",
        "components": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm",
            "Phylogenetic", "Residual"
        ],
        "coeff_order": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm"
        ],
    },
    "linear_fourthorder": {
        "file_suffix": "_linear_fourthorder",
        "display_name": "Linear with 4th Order Interactions",
        "components": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm",
            "Phylogenetic", "Residual"
        ],
        "coeff_order": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm"
        ],
    },
    "linear_fifthorder": {
        "file_suffix": "_linear_fifthorder",
        "display_name": "Linear with 5th Order Interactions",
        "components": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm",
            "Phylogenetic", "Residual"
        ],
        "coeff_order": [
            "longitude_norm", "latitude_norm",
            "log_n_speakers_norm", "n_segments_norm", "delta_norm"
        ],
    },
}

# Color palette
COMPONENT_COLORS = {
    # Spatial - cool tones
    "longitude_norm": "#4477AA",
    "latitude_norm": "#66CCEE",
    "longitude_norm:latitude_norm": "#228833",
    
    # Main predictors - warm tones  
    "log_n_speakers_norm": "#E69F00",
    "n_segments_norm": "#56B4E9",
    "delta_norm": "#009E73",
    
    # 2nd order interactions
    "longitude_norm:log_n_speakers_norm": "#F0E442",
    "longitude_norm:n_segments_norm": "#0072B2",
    "longitude_norm:delta_norm": "#D55E00",
    "latitude_norm:log_n_speakers_norm": "#CC79A7",
    "latitude_norm:n_segments_norm": "#882255",
    "latitude_norm:delta_norm": "#AA4499",
    "log_n_speakers_norm:n_segments_norm": "#DDCC77",
    "log_n_speakers_norm:delta_norm": "#44AA99",
    "n_segments_norm:delta_norm": "#117733",
    
    # Structural
    "Phylogenetic": "#332288",
    "Residual": "#DDDDDD",
}

TREE_LABELS = {
    "heggarty2024": "Cognates",
    "long_v3_CCD0.0.25": "Speech",
}

TREE_COLORS = {
    "Cognates": "#7ad151",
    "Speech": "#414487",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_variance_samples(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    model_type: str = "linear_main_only",
    coord_method: str = "standard",
) -> pd.DataFrame:
    """Load variance decomposition samples for linear models."""
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["linear_main_only"])
    suffix = config["file_suffix"]
    
    pattern = os.path.join(csv_dir, f"variance_samples{suffix}_*_{coord_method}.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No variance samples found for model_type='{model_type}'")
        return pd.DataFrame()
    
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def load_coef_samples(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    model_type: str = "linear_main_only",
    coord_method: str = "standard",
) -> pd.DataFrame:
    """Load coefficient posterior samples for linear models."""
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["linear_main_only"])
    suffix = config["file_suffix"]
    
    pattern = os.path.join(csv_dir, f"coef_samples{suffix}_*_{coord_method}.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No coefficient samples found for model_type='{model_type}'")
        return pd.DataFrame()
    
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def load_regression_results(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    model_type: str = "linear_main_only",
    coord_method: str = "standard",
) -> pd.DataFrame:
    """Load summary regression results for linear models."""
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["linear_main_only"])
    suffix = config["file_suffix"]
    
    pattern = os.path.join(csv_dir, f"phylolm{suffix}_*_{coord_method}.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No regression results found for model_type='{model_type}'")
        return pd.DataFrame()
    
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


# ============================================================================
# VARIANCE STACKED BAR PLOT
# ============================================================================

def plot_variance_stacked(
    model_type: str = "linear_main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    tree_labels: dict = None,
    output_dir: str = "speech_phylo/final_linear_figuresv2",
    save: bool = False,
) -> None:
    """Plot stacked horizontal bar chart of variance decomposition."""
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    samples = load_variance_samples(csv_dir, model_type, coord_method)
    
    if samples.empty:
        return
    
    samples["tree_display"] = samples["tree"].map(tree_labels).fillna(samples["tree"])
    trees = list(samples["tree_display"].unique())
    
    # Ensure Speech is on top
    if "Speech" in trees:
        trees = [t for t in trees if t != "Speech"] + ["Speech"]
    
    # Find available Shapley columns
    shapley_cols = [c for c in samples.columns if c.startswith("shapley_norm_")]
    
    # Build component order
    comp_order = []
    for col in shapley_cols:
        comp_name = col.replace("shapley_norm_", "").replace("_", ":")
        comp_order.append(comp_name)
    comp_order.extend(["Phylogenetic", "Residual"])
    
    # Compute summary statistics
    summary_data = {}
    for tree_display in trees:
        summary_data[tree_display] = {}
        tree_samples = samples[samples["tree_display"] == tree_display]
        
        # Shapley components
        for col in shapley_cols:
            comp_name = col.replace("shapley_norm_", "").replace("_", ":")
            if col in tree_samples.columns:
                vals = tree_samples[col].dropna().values
                summary_data[tree_display][comp_name] = {
                    "mean": np.mean(vals),
                    "q2_5": np.percentile(vals, 2.5),
                    "q97_5": np.percentile(vals, 97.5),
                }
        
        # Phylogenetic
        if "prop_phylo_norm" in tree_samples.columns:
            vals = tree_samples["prop_phylo_norm"].dropna().values
            summary_data[tree_display]["Phylogenetic"] = {
                "mean": np.mean(vals),
                "q2_5": np.percentile(vals, 2.5),
                "q97_5": np.percentile(vals, 97.5),
            }
        
        # Residual
        if "prop_residual_norm" in tree_samples.columns:
            vals = tree_samples["prop_residual_norm"].dropna().values
            summary_data[tree_display]["Residual"] = {
                "mean": np.mean(vals),
                "q2_5": np.percentile(vals, 2.5),
                "q97_5": np.percentile(vals, 97.5),
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Variance Decomposition: {config['display_name']}")
    print(f"{'='*60}")
    for tree_display in trees:
        print(f"\n{tree_display}:")
        total = 0
        for comp in comp_order:
            if comp in summary_data[tree_display]:
                d = summary_data[tree_display][comp]
                total += d["mean"]
                if d["mean"] > 0.001:
                    print(f"  {comp:35s}: {d['mean']:6.1%} ({d['q2_5']:.0%}–{d['q97_5']:.0%})")
        print(f"  {'Total':35s}: {total:6.1%}")
    
    # Build plot
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.9 * len(trees))))
    
    y_positions = np.arange(len(trees))
    left = np.zeros(len(trees))
    bar_height = 0.65
    
    legend_handles = []
    legend_labels = []
    
    for comp in comp_order:
        values = np.array([summary_data[t].get(comp, {}).get("mean", 0) for t in trees])
        
        if np.allclose(values, 0.0):
            continue
        
        color = COMPONENT_COLORS.get(comp, "#888888")
        bars = ax.barh(y_positions, values, left=left, height=bar_height,
                       color=color, edgecolor="white", linewidth=0.5)
        
        legend_handles.append(bars[0])
        legend_labels.append(comp)
        
        # Add labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width >= 0.05:
                mean_pct = values[i]
                label = f"{mean_pct*100:.0f}%"
                x_pos = left[i] + width / 2
                ax.text(x_pos, y_positions[i], label, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        
        left += values
    
    # Professional styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(trees, fontsize=11)
    ax.set_xlabel("Proportion of Variance", fontsize=11, fontweight="medium")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    
    # Legend below plot
    ax.legend(legend_handles, legend_labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=min(5, len(legend_labels)), frameon=False, fontsize=8)
    
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = config["file_suffix"]
        path = os.path.join(output_dir, f"variance_stacked{suffix}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


# ============================================================================
# COEFFICIENT VIOLIN PLOT
# ============================================================================

def plot_effects_comparison(
    model_type: str = "linear_main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    tree_labels: dict = None,
    output_dir: str = "speech_phylo/final_linear_figuresv2",
    save: bool = False,
) -> None:
    """Plot coefficient posterior distributions as violin plots with 95% CI."""
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    coeff_order = config.get("coeff_order", [])
    
    coef_samples = load_coef_samples(csv_dir, model_type, coord_method)
    
    if coef_samples.empty:
        print(f"No coefficient samples found. Run R script first.")
        return
    
    coef_samples["tree_display"] = coef_samples["tree"].map(tree_labels).fillna(coef_samples["tree"])
    trees = list(coef_samples["tree_display"].unique())
    
    # Ensure Speech is on top
    if "Speech" in trees:
        trees = [t for t in trees if t != "Speech"] + ["Speech"]
    
    # Map column names to display names
    col_mapping = {}
    for col in coef_samples.columns:
        if col not in ["sample_id", "tree", "tree_display"]:
            col_mapping[col] = col.replace(".", ":")
    
    n_coeffs = len(coeff_order)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * n_coeffs)))
    
    offset = 0.2
    
    for t_idx, tree_display in enumerate(trees):
        tree_samples = coef_samples[coef_samples["tree_display"] == tree_display]
        color = TREE_COLORS.get(tree_display, "#888888")
        
        violin_data = []
        for coeff in coeff_order:
            # Find matching column
            col_name = None
            for csv_col, display in col_mapping.items():
                if display == coeff or csv_col == coeff or csv_col == coeff.replace(":", "."):
                    if csv_col in tree_samples.columns:
                        col_name = csv_col
                        break
            
            if col_name and col_name in tree_samples.columns:
                vals = tree_samples[col_name].dropna().values
                violin_data.append(vals if len(vals) > 0 else np.array([0]))
            else:
                violin_data.append(np.array([0]))
        
        positions = np.arange(n_coeffs) + (t_idx - 0.5) * offset * 2
        
        # Create violin plots
        parts = ax.violinplot(
            violin_data, positions=positions, vert=False,
            showmeans=False, showmedians=False, showextrema=False, widths=0.35,
        )
        
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
        
        # Add 95% CI lines with endpoint caps, mean marker, and text labels
        for i, (pos, data) in enumerate(zip(positions, violin_data)):
            if len(data) > 1:
                mean_val = np.mean(data)
                q2_5 = np.percentile(data, 2.5)
                q97_5 = np.percentile(data, 97.5)
                cap_height = 0.08
                
                # CI line
                ax.plot([q2_5, q97_5], [pos, pos], color=color, linewidth=1.5, alpha=0.8)
                # Endpoint caps
                ax.plot([q2_5, q2_5], [pos - cap_height, pos + cap_height],
                        color=color, linewidth=1.5, alpha=0.8)
                ax.plot([q97_5, q97_5], [pos - cap_height, pos + cap_height],
                        color=color, linewidth=1.5, alpha=0.8)
                # Mean marker
                ax.scatter([mean_val], [pos], color=color, s=25, zorder=5,
                          marker='|', linewidth=1.5)
                
                # Add text labels for CI endpoints
                ax.text(q2_5, pos + 0.15, f'{q2_5:.2f}', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
                ax.text(q97_5, pos + 0.15, f'{q97_5:.2f}', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
    
    # Professional styling
    ax.axvline(0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels(coeff_order, fontsize=10)
    ax.set_xlabel("Standardized Effect Size", fontsize=11, fontweight="medium")
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis="x", linestyle="-", alpha=0.2, color="#888888")
    ax.set_title(f"Coefficient Estimates (95% CI)\n{config['display_name']}",
                 fontsize=11, fontweight="medium")
    
    # Legend
    legend_elements = [
        Patch(facecolor=TREE_COLORS.get(t, "#888888"), alpha=0.55,
              edgecolor=TREE_COLORS.get(t, "#888888"), label=t)
        for t in trees
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              frameon=False, handlelength=1.2)
    
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = config["file_suffix"]
        path = os.path.join(output_dir, f"effects_comparison{suffix}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def plot_variance_violin(
    model_type: str = "linear_main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    tree_labels: dict = None,
    output_dir: str = "final_linear_figuresv2",
    save: bool = False,
) -> None:
    """Plot violin plots of variance decomposition posterior samples."""
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    samples = load_variance_samples(csv_dir, model_type, coord_method)
    
    if samples.empty:
        print(f"No variance samples found for model_type='{model_type}'")
        return
    
    samples["tree_display"] = samples["tree"].map(tree_labels).fillna(samples["tree"])
    trees = list(samples["tree_display"].unique())
    
    # Ensure Speech is on top
    if "Speech" in trees:
        trees = [t for t in trees if t != "Speech"] + ["Speech"]
    
    # Find available Shapley columns
    shapley_cols = [c for c in samples.columns if c.startswith("shapley_norm_")]
    
    # Build component order
    comp_order = []
    for col in shapley_cols:
        comp_name = col.replace("shapley_norm_", "").replace("_", ":")
        comp_order.append(comp_name)
    comp_order.extend(["Phylogenetic", "Residual"])
    
    n_comps = len(comp_order)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * n_comps)))
    
    offset = 0.18
    
    for t_idx, tree_display in enumerate(trees):
        tree_samples = samples[samples["tree_display"] == tree_display]
        color = TREE_COLORS.get(tree_display, "#888888")
        
        violin_data = []
        for comp in comp_order:
            if comp == "Phylogenetic":
                col_name = "prop_phylo_norm"
            elif comp == "Residual":
                col_name = "prop_residual_norm"
            else:
                col_name = "shapley_norm_" + comp.replace(":", "_")
            
            if col_name in tree_samples.columns:
                vals = tree_samples[col_name].dropna().values
                violin_data.append(vals if len(vals) > 0 else np.array([0]))
            else:
                violin_data.append(np.array([0]))
        
        positions = np.arange(n_comps) + (t_idx - 0.5) * offset * 2
        
        parts = ax.violinplot(
            violin_data, positions=positions, vert=False,
            showmeans=False, showmedians=False, showextrema=False, widths=0.32,
        )
        
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
        
        # Add 95% CI lines with endpoint caps, mean marker, and text labels
        for i, (pos, data) in enumerate(zip(positions, violin_data)):
            if len(data) > 1:
                mean_val = np.mean(data)
                q2_5 = np.percentile(data, 2.5)
                q97_5 = np.percentile(data, 97.5)
                cap_height = 0.08
                
                ax.plot([q2_5, q97_5], [pos, pos], color=color, linewidth=1.5, alpha=0.8)
                ax.plot([q2_5, q2_5], [pos - cap_height, pos + cap_height],
                        color=color, linewidth=1.5, alpha=0.8)
                ax.plot([q97_5, q97_5], [pos - cap_height, pos + cap_height],
                        color=color, linewidth=1.5, alpha=0.8)
                ax.scatter([mean_val], [pos], color=color, s=30, zorder=5,
                          marker='|', linewidth=2)
                
                # Add text labels for CI endpoints (as percentages for variance)
                ax.text(q2_5, pos + 0.15, f'{q2_5*100:.0f}%', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
                ax.text(q97_5, pos + 0.15, f'{q97_5*100:.0f}%', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
    
    # Professional styling
    ax.set_yticks(range(n_comps))
    ax.set_yticklabels(comp_order, fontsize=10)
    ax.set_xlabel("Proportion of Variance", fontsize=11, fontweight="medium")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis="x", linestyle="-", alpha=0.2, color="#888888")
    ax.set_title(f"Variance Decomposition (95% CI)\n{config['display_name']}",
                 fontsize=11, fontweight="medium")
    
    # Legend
    legend_elements = [
        Patch(facecolor=TREE_COLORS.get(t, "#888888"), alpha=0.55,
              edgecolor=TREE_COLORS.get(t, "#888888"), label=t)
        for t in trees
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              frameon=False, handlelength=1.2)
    
    # Add MCMC diagnostics subtitle if available
    reg_results = load_regression_results(csv_dir, model_type, coord_method)
    if not reg_results.empty and 'min_bulk_ess' in reg_results.columns:
        diag_parts = []
        for tree_name in samples["tree"].unique():
            tree_reg = reg_results[reg_results["tree"] == tree_name]
            if not tree_reg.empty:
                tree_disp = tree_labels.get(tree_name, tree_name)
                n_div = int(tree_reg["n_divergent"].iloc[0]) if "n_divergent" in tree_reg.columns else 0
                rhat = tree_reg["max_rhat"].iloc[0] if "max_rhat" in tree_reg.columns else 1.0
                ess_all = int(tree_reg["min_bulk_ess"].iloc[0])
                ess_fix = int(tree_reg["min_bulk_ess_fixed"].iloc[0]) if "min_bulk_ess_fixed" in tree_reg.columns else ess_all
                diag_parts.append(f"{tree_disp}: ESS_fix={ess_fix}/all={ess_all}, R̂={rhat:.2f}, div={n_div}")
        if diag_parts:
            fig.suptitle(" | ".join(diag_parts), fontsize=8, color="#666666", y=0.02)
    
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = config["file_suffix"]
        path = os.path.join(output_dir, f"variance_violin{suffix}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


def plot_all_linear_models(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    save: bool = False,
    output_dir: str = "speech_phylo/final_linear_figuresv2",
) -> None:
    """Plot variance decomposition and effects for all linear interaction levels."""
    model_types = [
        "linear_main_only", 
        "linear_secondorder", 
        # "linear_thirdorder[]",
        # "linear_fourthorder",
        # "linear_fifthorder"
    ]
    for model_type in model_types:
        if model_type not in MODEL_CONFIGS:
            continue
        print(f"\n{'='*70}")
        print(f"Model: {MODEL_CONFIGS[model_type]['display_name']}")
        print(f"{'='*70}")
        
        plot_variance_violin(model_type=model_type, csv_dir=csv_dir,
                             coord_method=coord_method, save=save, output_dir=output_dir)
        plot_effects_comparison(model_type=model_type, csv_dir=csv_dir,
                                coord_method=coord_method, save=save, output_dir=output_dir)


def create_combined_pdf_simple(
    output_dir: str = "speech_phylofinal_linear_figuresv2",
    output_filename: str = "all_linear_figures_combined.pdf",
    coord_method: str = "standard",
) -> None:
    """Combine saved SVG files for linear models into a single PDF, grouped by plot type."""
    import subprocess
    import shutil
    
    try:
        import cairosvg
        has_cairosvg = True
    except ImportError:
        has_cairosvg = False
    
    has_inkscape = shutil.which("inkscape") is not None
    
    if not has_cairosvg and not has_inkscape:
        print("Warning: Neither cairosvg nor inkscape found.")
        print("Install cairosvg: pip install cairosvg")
        return
    
    # Define plot type order for linear models
    plot_types = [
        ("variance_violin_linear", "Variance Violin Plots"),
        ("variance_stacked_linear", "Variance Stacked Bar Plots"),
        ("effects_comparison_linear", "Effects Comparison Plots"),
    ]
    
    model_order = ["main_only", "secondorder", "thirdorder", "fourthorder", "fifthorder"]
    
    svg_files_ordered = []
    
    for plot_prefix, plot_name in plot_types:
        matching = []
        for model in model_order:
            pattern = f"{plot_prefix}_{model}*{coord_method}.svg"
            files = sorted(glob.glob(os.path.join(output_dir, pattern)))
            matching.extend(files)
        
        # Also check with underscore variant
        all_matching = glob.glob(os.path.join(output_dir, f"{plot_prefix}*{coord_method}.svg"))
        for f in sorted(all_matching):
            if f not in matching:
                matching.append(f)
        
        svg_files_ordered.extend(matching)
        if matching:
            print(f"\n{plot_name}:")
            for f in matching:
                print(f"  - {os.path.basename(f)}")
    
    if not svg_files_ordered:
        print(f"No SVG files found in {output_dir}")
        return
    
    print(f"\nTotal: {len(svg_files_ordered)} SVG files to combine")
    
    temp_pdfs = []
    temp_dir = os.path.join(output_dir, ".temp_pdfs")
    os.makedirs(temp_dir, exist_ok=True)
    
    for svg_path in svg_files_ordered:
        pdf_name = os.path.basename(svg_path).replace(".svg", ".pdf")
        pdf_path = os.path.join(temp_dir, pdf_name)
        
        try:
            if has_cairosvg:
                cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            elif has_inkscape:
                subprocess.run(
                    ["inkscape", svg_path, "--export-type=pdf", f"--export-filename={pdf_path}"],
                    capture_output=True, check=True
                )
            temp_pdfs.append(pdf_path)
            print(f"  ✔ Converted: {os.path.basename(svg_path)}")
        except Exception as e:
            print(f"  ✗ Error converting {os.path.basename(svg_path)}: {e}")
    
    if not temp_pdfs:
        print("No PDFs were created")
        return
    
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        for pdf in temp_pdfs:
            merger.append(pdf)
        merger.write(output_path)
        merger.close()
        print(f"\n✅ Combined PDF saved to: {output_path}")
    except ImportError:
        try:
            import pikepdf
            pdf_output = pikepdf.Pdf.new()
            for pdf in temp_pdfs:
                with pikepdf.Pdf.open(pdf) as src:
                    pdf_output.pages.extend(src.pages)
            pdf_output.save(output_path)
            print(f"\n✅ Combined PDF saved to: {output_path}")
        except ImportError:
            if shutil.which("gs"):
                cmd = ["gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
                       f"-sOutputFile={output_path}"] + temp_pdfs
                subprocess.run(cmd, check=True)
                print(f"\n✅ Combined PDF saved to: {output_path}")
            else:
                print("Error: No PDF merger available. Install PyPDF2 or pikepdf:")
                print("  pip install PyPDF2")
    
    import shutil as shutil_mod
    shutil_mod.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Plotting linear interaction models...")
    plot_all_linear_models(
        csv_dir="speech_phylo/final_phyloregression_results",
        coord_method="standard",
        save=True,
        output_dir="speech_phylo/final_linear_figuresv2",
    )
    
    # Create combined PDF
    create_combined_pdf_simple(
        output_dir="speech_phylo/final_linear_figuresv2",
        output_filename="all_linear_figures_combined.pdf"
    )
