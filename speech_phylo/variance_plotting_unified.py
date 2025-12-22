"""
Unified Variance Decomposition Plotting Module
===============================================
Handles all three model variants:
  - "main_only": 3 main effects only
  - "secondorder": 3 main + 3 pairwise interactions
  - "allinteractions": 3 main + 3 pairwise + 1 triple interaction

Usage:
    from variance_plotting_unified import (
        plot_variance_violin,
        plot_variance_stacked,
        plot_spline_surface,
        load_variance_samples,
        load_regression_results,
        load_spline_surface,
        plot_all_models,
    )

    # Plot all models
    plot_all_models(csv_dir="final_phyloregression_results", save=True)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    "main_only": {
        "file_suffix": "",
        "components": [
            "log_n_speakers", "n_segments", "delta",
            "Spatial (spline)", "Phylogenetic", "Residual"
        ],
        "shapley_cols": {
            "shapley_norm_log_n_speakers": "log_n_speakers",
            "shapley_norm_n_segments": "n_segments",
            "shapley_norm_delta": "delta",
            "prop_spline_norm": "Spatial (spline)",
            "prop_phylo_norm": "Phylogenetic",
            "prop_residual_norm": "Residual",
        },
        "display_name": "Main Effects Only",
        "coeff_order": ["log_n_speakers", "n_segments", "delta"],
        "coef_columns": {
            "log_n_speakers": ("coef_log_n_speakers", "ci_lower_log_n_speakers", "ci_upper_log_n_speakers"),
            "n_segments": ("coef_n_segments", "ci_lower_n_segments", "ci_upper_n_segments"),
            "delta": ("coef_delta", "ci_lower_delta", "ci_upper_delta"),
        },
    },
    "secondorder": {
        "file_suffix": "_secondorder",
        "components": [
            "log_n_speakers", "n_segments", "delta",
            "log_n_speakers×n_segments", "log_n_speakers×delta", "n_segments×delta",
            "Spatial (spline)", "Phylogenetic", "Residual"
        ],
        "shapley_cols": {
            "shapley_norm_log_n_speakers": "log_n_speakers",
            "shapley_norm_n_segments": "n_segments",
            "shapley_norm_delta": "delta",
            "shapley_norm_log_n_speakers_n_segments": "log_n_speakers×n_segments",
            "shapley_norm_log_n_speakers_delta": "log_n_speakers×delta",
            "shapley_norm_n_segments_delta": "n_segments×delta",
            "prop_spline_norm": "Spatial (spline)",
            "prop_phylo_norm": "Phylogenetic",
            "prop_residual_norm": "Residual",
        },
        "display_name": "With 2nd-Order Interactions",
    },
    "allinteractions": {
        "file_suffix": "_allinteractions",
        "components": [
            "log_n_speakers", "n_segments", "delta",
            "log_n_speakers×n_segments", "log_n_speakers×delta", "n_segments×delta",
            "Triple Interaction",
            "Spatial (spline)", "Phylogenetic", "Residual"
        ],
        "shapley_cols": {
            "shapley_norm_log_n_speakers": "log_n_speakers",
            "shapley_norm_n_segments": "n_segments",
            "shapley_norm_delta": "delta",
            "shapley_norm_log_n_speakers_n_segments": "log_n_speakers×n_segments",
            "shapley_norm_log_n_speakers_delta": "log_n_speakers×delta",
            "shapley_norm_n_segments_delta": "n_segments×delta",
            "shapley_norm_triple": "Triple Interaction",
            "prop_spline_norm": "Spatial (spline)",
            "prop_phylo_norm": "Phylogenetic",
            "prop_residual_norm": "Residual",
        },
        "display_name": "With All Interactions (2nd + 3rd Order)",
        # Coefficient columns for effects comparison plot
        "coeff_order": ["log_n_speakers", "n_segments", "delta"],
        "coef_columns": {
            "log_n_speakers": ("coef_log_n_speakers", "ci_lower_log_n_speakers", "ci_upper_log_n_speakers"),
            "n_segments": ("coef_n_segments", "ci_lower_n_segments", "ci_upper_n_segments"),
            "delta": ("coef_delta", "ci_lower_delta", "ci_upper_delta"),
        },
    },
    "secondorder": {
        "file_suffix": "_secondorder",
        "components": [
            "log_n_speakers", "n_segments", "delta",
            "log_n_speakers×n_segments", "log_n_speakers×delta", "n_segments×delta",
            "Spatial (spline)", "Phylogenetic", "Residual"
        ],
        "shapley_cols": {
            "shapley_norm_log_n_speakers": "log_n_speakers",
            "shapley_norm_n_segments": "n_segments",
            "shapley_norm_delta": "delta",
            "shapley_norm_log_n_speakers_n_segments": "log_n_speakers×n_segments",
            "shapley_norm_log_n_speakers_delta": "log_n_speakers×delta",
            "shapley_norm_n_segments_delta": "n_segments×delta",
            "prop_spline_norm": "Spatial (spline)",
            "prop_phylo_norm": "Phylogenetic",
            "prop_residual_norm": "Residual",
        },
        "display_name": "With 2nd-Order Interactions",
        "coeff_order": [
            "log_n_speakers", "n_segments", "delta",
            "log_n_speakers×n_segments", "log_n_speakers×delta", "n_segments×delta",
        ],
        "coef_columns": {
            "log_n_speakers": ("coef_log_n_speakers", "ci_lower_log_n_speakers", "ci_upper_log_n_speakers"),
            "n_segments": ("coef_n_segments", "ci_lower_n_segments", "ci_upper_n_segments"),
            "delta": ("coef_delta", "ci_lower_delta", "ci_upper_delta"),
            "log_n_speakers×n_segments": ("coef_log_n_speakers_n_segments", "ci_lower_log_n_speakers_n_segments", "ci_upper_log_n_speakers_n_segments"),
            "log_n_speakers×delta": ("coef_log_n_speakers_delta", "ci_lower_log_n_speakers_delta", "ci_upper_log_n_speakers_delta"),
            "n_segments×delta": ("coef_n_segments_delta", "ci_lower_n_segments_delta", "ci_upper_n_segments_delta"),
        },
    },
    "allinteractions": {
        "file_suffix": "_allinteractions",
        "components": [
            "log_n_speakers", "n_segments", "delta",
            "log_n_speakers×n_segments", "log_n_speakers×delta", "n_segments×delta",
            "Triple Interaction",
            "Spatial (spline)", "Phylogenetic", "Residual"
        ],
        "shapley_cols": {
            "shapley_norm_log_n_speakers": "log_n_speakers",
            "shapley_norm_n_segments": "n_segments",
            "shapley_norm_delta": "delta",
            "shapley_norm_log_n_speakers_n_segments": "log_n_speakers×n_segments",
            "shapley_norm_log_n_speakers_delta": "log_n_speakers×delta",
            "shapley_norm_n_segments_delta": "n_segments×delta",
            "shapley_norm_triple": "Triple Interaction",
            "prop_spline_norm": "Spatial (spline)",
            "prop_phylo_norm": "Phylogenetic",
            "prop_residual_norm": "Residual",
        },
        "display_name": "With All Interactions (2nd + 3rd Order)",
        "coeff_order": [
            "log_n_speakers", "n_segments", "delta",
            "log_n_speakers×n_segments", "log_n_speakers×delta", "n_segments×delta",
            "Triple Interaction",
        ],
        "coef_columns": {
            "log_n_speakers": ("coef_log_n_speakers", "ci_lower_log_n_speakers", "ci_upper_log_n_speakers"),
            "n_segments": ("coef_n_segments", "ci_lower_n_segments", "ci_upper_n_segments"),
            "delta": ("coef_delta", "ci_lower_delta", "ci_upper_delta"),
            "log_n_speakers×n_segments": ("coef_log_n_speakers_n_segments", "ci_lower_log_n_speakers_n_segments", "ci_upper_log_n_speakers_n_segments"),
            "log_n_speakers×delta": ("coef_log_n_speakers_delta", "ci_lower_log_n_speakers_delta", "ci_upper_log_n_speakers_delta"),
            "n_segments×delta": ("coef_n_segments_delta", "ci_lower_n_segments_delta", "ci_upper_n_segments_delta"),
            "Triple Interaction": ("coef_log_n_speakers_n_segments_delta", "ci_lower_log_n_speakers_n_segments_delta", "ci_upper_log_n_speakers_n_segments_delta"),
        },
    },
}

# Professional Nature/Science color palette
# Inspired by ColorBrewer "Set2" and muted earth tones suitable for print
COMPONENT_COLORS = {
    # Main effects - warm tones
    "log_n_speakers": "#E69F00",      # Orange (population)
    "n_segments": "#56B4E9",          # Sky blue (phonemes)
    "delta": "#009E73",               # Teal/green (speech rate)
    
    # Interactions - cooler tones
    "log_n_speakers×n_segments": "#F0E442",  # Yellow
    "log_n_speakers×delta": "#0072B2",       # Deep blue
    "n_segments×delta": "#D55E00",           # Vermillion
    "Triple Interaction": "#CC79A7",         # Pink/mauve
    
    # Structural components
    "Spatial (spline)": "#999999",    # Medium gray
    "Phylogenetic": "#332288",        # Deep purple
    "Residual": "#DDDDDD",            # Light gray
    
    # Legacy
    "Linear Effects": "#009E73",
}

TREE_COLORS = {
    "Cognates": "#7ad151",
    "Speech": "#414487",
}

TREE_LABELS = {
    "long_v3_CCD0.0.25": "Speech",
    "heggarty2024": "Cognates",
}

PLOT_STYLE = {
    "figsize": (12, 8),
    "row_spacing": 2.0,
    "capsize": 5,
    "capthick": 2,
    "markersize": 9,
    "linewidth": 2.2,
    "ylabel_fontsize": 14,
    "xlabel_fontsize": 14,
    "tick_fontsize": 12,
    "legend_fontsize": 14,
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_variance_samples(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    model_type: str = "main_only",
    coord_method: str = "standard",
) -> pd.DataFrame:
    """Load posterior samples for variance decomposition."""
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["main_only"])
    suffix = config["file_suffix"]
    
    pattern = os.path.join(csv_dir, f"variance_samples{suffix}_*_{coord_method}.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No files found matching: {pattern}")
        return pd.DataFrame()
    
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def load_regression_results(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    model_type: str = "main_only",
    coord_method: str = "standard",
) -> pd.DataFrame:
    """Load regression coefficient results."""
    suffix = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["main_only"])["file_suffix"]
    
    if suffix == "":
        pattern = os.path.join(csv_dir, f"phylolm_with_inventory_*_{coord_method}.csv")
        # Exclude files with _allinteractions or _secondorder
        csv_files = [f for f in glob.glob(pattern) 
                     if "_allinteractions_" not in f and "_secondorder_" not in f]
    else:
        suffix_clean = suffix.lstrip("_")
        pattern = os.path.join(csv_dir, f"phylolm_with_inventory_{suffix_clean}_*_{coord_method}.csv")
        csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No regression results found for model_type='{model_type}'")
        return pd.DataFrame()
    
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def load_spline_surface(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    model_type: str = "main_only",
    tree_name: str = "heggarty2024",
    coord_method: str = "standard",
) -> pd.DataFrame:
    """Load spline surface predictions for a given tree."""
    suffix = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["main_only"])["file_suffix"]
    
    if suffix == "":
        filename = f"spline_surface_{tree_name}_{coord_method}.csv"
    else:
        suffix_clean = suffix.lstrip("_")
        filename = f"spline_surface_{suffix_clean}_{tree_name}_{coord_method}.csv"
    
    path = os.path.join(csv_dir, filename)
    
    if not os.path.exists(path):
        print(f"Spline surface not found: {path}")
        return pd.DataFrame()
    
    return pd.read_csv(path)


def load_coef_samples(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    model_type: str = "main_only",
    coord_method: str = "standard",
) -> pd.DataFrame:
    """Load coefficient posterior samples for violin plots."""
    suffix = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["main_only"])["file_suffix"]
    
    if suffix == "":
        pattern = os.path.join(csv_dir, f"coef_samples_*_{coord_method}.csv")
        # Exclude secondorder and allinteractions files
        csv_files = [f for f in glob.glob(pattern) 
                     if "secondorder" not in f and "allinteractions" not in f]
    else:
        suffix_clean = suffix.lstrip("_")
        pattern = os.path.join(csv_dir, f"coef_samples_{suffix_clean}_*_{coord_method}.csv")
        csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No coefficient samples found for model_type='{model_type}'")
        return pd.DataFrame()
    
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


# ============================================================================
# VIOLIN PLOT
# ============================================================================

def plot_variance_violin(
    model_type: str = "main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    tree_labels: dict = None,
    output_dir: str = "speech_phylo/final_figuresv2",
    save: bool = False,
) -> None:
    """Plot overlapping violin plots for posterior variance distributions.
    
    Professional styling inspired by Nature/Science journal figures.
    """
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    samples = load_variance_samples(csv_dir, model_type, coord_method)
    
    if samples.empty:
        return
    
    # Load regression results to get MCMC diagnostics
    reg_results = load_regression_results(csv_dir, model_type, coord_method)
    
    samples["tree_display"] = samples["tree"].map(tree_labels).fillna(samples["tree"])
    trees = list(samples["tree_display"].unique())
    # Ensure Speech is always on top (matplotlib draws bottom-to-top, so reverse)
    if "Speech" in trees:
        trees = [t for t in trees if t != "Speech"] + ["Speech"]
    
    col_map = config["shapley_cols"]
    comp_order = config["components"]
    available_cols = {k: v for k, v in col_map.items() if k in samples.columns}
    
    n_comps = len(comp_order)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * n_comps)))
    
    offset = 0.18
    
    for t_idx, tree_display in enumerate(trees):
        tree_data = samples[samples["tree_display"] == tree_display]
        color = TREE_COLORS.get(tree_display, "#888888")
        
        violin_data = []
        for comp in comp_order:
            col_name = [k for k, v in available_cols.items() if v == comp]
            if col_name and col_name[0] in tree_data.columns:
                vals = tree_data[col_name[0]].dropna().values
                violin_data.append(vals if len(vals) > 0 else np.array([0]))
            else:
                violin_data.append(np.array([0]))
        
        positions = np.arange(n_comps) + (t_idx - 0.5) * offset * 2
        
        # Create violin plots without built-in extrema (we'll add 95% CI manually)
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
                cap_height = 0.12  # Height of endpoint caps (increased)
                
                # Draw 95% CI line (thicker)
                ax.plot([q2_5, q97_5], [pos, pos], color=color, linewidth=2.5, alpha=0.9)
                # Draw endpoint caps (thicker)
                ax.plot([q2_5, q2_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=2.5, alpha=0.9)
                ax.plot([q97_5, q97_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=2.5, alpha=0.9)
                # Draw mean marker (thicker)
                ax.scatter([mean_val], [pos], color=color, s=40, zorder=5, 
                          marker='|', linewidth=3)
                
                # Add text labels for CI endpoints (as percentages for variance)
                ax.text(q2_5, pos + 0.15, f'{q2_5*100:.0f}%', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
                ax.text(q97_5, pos + 0.15, f'{q97_5*100:.0f}%', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
    
    # Add subtle highlight band for delta row to emphasize key finding
    if "delta" in comp_order:
        delta_idx = comp_order.index("delta")
        ax.axhspan(delta_idx - 0.4, delta_idx + 0.4, color='#fff3cd', alpha=0.4, zorder=0)
    
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
    
    # Professional legend
    legend_elements = [
        Patch(facecolor=TREE_COLORS.get(t, "#888888"), alpha=0.55,
              edgecolor=TREE_COLORS.get(t, "#888888"), label=t)
        for t in trees
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              frameon=False, handlelength=1.2)
    
    # Add MCMC diagnostics subtitle if available
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
        suffix = config["file_suffix"] or "_mainonly"
        path = os.path.join(output_dir, f"variance_violin{suffix}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


# ============================================================================
# STACKED BAR PLOT
# ============================================================================

def plot_variance_stacked(
    model_type: str = "main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    tree_labels: dict = None,
    output_dir: str = "speech_phylo/final_figuresv2",
    save: bool = False,
) -> None:
    """Plot stacked horizontal bar chart of variance decomposition.
    
    Professional styling inspired by Nature/Science journal figures.
    """
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    samples = load_variance_samples(csv_dir, model_type, coord_method)
    
    if samples.empty:
        return
    
    samples["tree_display"] = samples["tree"].map(tree_labels).fillna(samples["tree"])
    trees = list(samples["tree_display"].unique())
    # Ensure Speech is always on top (matplotlib draws bottom-to-top, so reverse)
    if "Speech" in trees:
        trees = [t for t in trees if t != "Speech"] + ["Speech"]
    
    col_map = config["shapley_cols"]
    comp_order = config["components"]
    
    # Compute summary statistics
    summary_data = {}
    for tree_display in trees:
        tree_samples = samples[samples["tree_display"] == tree_display]
        summary_data[tree_display] = {}
        for col, comp_name in col_map.items():
            if col in tree_samples.columns:
                vals = tree_samples[col].dropna().values
                summary_data[tree_display][comp_name] = {
                    "mean": np.mean(vals),
                    "q2_5": np.percentile(vals, 2.5),
                    "q97_5": np.percentile(vals, 97.5),
                }
            else:
                summary_data[tree_display][comp_name] = {"mean": 0, "q2_5": 0, "q97_5": 0}
    
    # Normalize each tree's components to sum to exactly 1.0
    for tree_display in trees:
        total = sum(summary_data[tree_display].get(c, {}).get("mean", 0) for c in comp_order)
        if total > 0 and abs(total - 1.0) > 0.0001:
            for comp in comp_order:
                if comp in summary_data[tree_display]:
                    summary_data[tree_display][comp]["mean"] /= total
    
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
                    print(f"  {comp:25s}: {d['mean']:6.1%} ({d['q2_5']:.0%}–{d['q97_5']:.0%})")
        print(f"  {'Total':25s}: {total:6.1%}")
    
    # Build professional plot
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.9 * len(trees))))
    
    y_positions = np.arange(len(trees))
    left = np.zeros(len(trees))
    bar_height = 0.65
    
    # Track components for legend
    legend_handles = []
    legend_labels = []
    
    for comp in comp_order:
        values = np.array([summary_data[t].get(comp, {}).get("mean", 0) for t in trees])
        q2_5 = np.array([summary_data[t].get(comp, {}).get("q2_5", 0) for t in trees])
        q97_5 = np.array([summary_data[t].get(comp, {}).get("q97_5", 0) for t in trees])
        
        if np.allclose(values, 0.0):
            continue
        
        color = COMPONENT_COLORS.get(comp, "#888888")
        bars = ax.barh(y_positions, values, left=left, height=bar_height,
                       color=color, edgecolor="white", linewidth=0.5)
        
        legend_handles.append(bars[0])
        legend_labels.append(comp)
        
        # Add labels inside bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width < 0.005:
                continue
            
            mean_pct = values[i]
            ci_low = q2_5[i]
            ci_high = q97_5[i]
            
            if width >= 0.12:
                # Large segment: percentage with CI on second line
                label = f"{mean_pct*100:.0f}%\n[{ci_low*100:.0f}–{ci_high*100:.0f}]"
                fontsize = 8
                text_color = "white"
                x_pos = left[i] + width / 2
            elif width >= 0.05:
                # Medium segment: just percentage
                label = f"{mean_pct*100:.0f}%"
                fontsize = 8
                text_color = "white"
                x_pos = left[i] + width / 2
            else:
                # Skip very small segments
                continue
            
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    label, ha="center", va="center", fontsize=fontsize,
                    color=text_color, fontweight="semibold")
        
        left += values
    
    # Professional styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(trees, fontsize=12, fontweight="medium")
    ax.set_xlabel("Proportion of Variance Explained", fontsize=11, fontweight="medium")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    
    # Professional legend below plot
    ax.legend(legend_handles, legend_labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=min(4, len(legend_labels)), frameon=False, fontsize=9,
              handlelength=1.5, handletextpad=0.5, columnspacing=1.5)
    
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = config["file_suffix"] or "_mainonly"
        path = os.path.join(output_dir, f"variance_stacked{suffix}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


# ============================================================================
# SPLINE SURFACE PLOT
# ============================================================================

def plot_spline_surface(
    model_type: str = "main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    tree_name: str = "heggarty2024",
    tree_labels: dict = None,
    output_dir: str = "speech_phylo/final_figuresv2",
    save: bool = False,
    show_observations: bool = True,
    show_basemap: bool = True,
    geojson_path: str = "speech_phylo/dataset.geojson",
) -> None:
    """Plot the spline surface for a given tree with language polygon masking.
    
    Uses dataset.geojson for language boundaries, showing the spline effect
    only within Indo-European language regions.
    """
    import geopandas as gpd
    from shapely.geometry import box, Point
    from shapely.prepared import prep
    
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    df = load_spline_surface(csv_dir, model_type, tree_name, coord_method)
    
    if df.empty:
        return
    
    tree_display = tree_labels.get(tree_name, tree_name)
    
    # Reshape for plotting
    lon = df["longitude"].values
    lat = df["latitude"].values
    z = df["spline_mean"].values
    
    n_grid = int(np.sqrt(len(lon)))
    
    try:
        lon_grid = lon.reshape(n_grid, n_grid)
        lat_grid = lat.reshape(n_grid, n_grid)
        z_grid = z.reshape(n_grid, n_grid)
    except ValueError:
        print(f"Cannot reshape data. Expected {n_grid}x{n_grid} grid, got {len(lon)} points.")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define ROI bounds from data with padding (Europe + India region)
    PAD_DEG = 15
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ire = world.loc[world['name'] == 'Ireland'].total_bounds
    bgd = world.loc[world['name'] == 'Bangladesh'].total_bounds
    ROI_MINX = min(ire[0], bgd[0]) - PAD_DEG
    ROI_MAXX = max(ire[2], bgd[2]) + PAD_DEG
    ROI_MINY = min(ire[1], bgd[1]) - PAD_DEG
    ROI_MAXY = max(ire[3], bgd[3]) + PAD_DEG
    ROI_BOX = box(ROI_MINX, ROI_MINY, ROI_MAXX, ROI_MAXY)
    
    # Load language GeoJSON for masking
    gdf_language = None
    language_union = None
    if os.path.exists(geojson_path):
        try:
            gdf_language = gpd.read_file(geojson_path)
            
            # Rename languages to match metadata
            renaming = {
                "Punjabi (Panjabi)": "Punjabi",
                "Norwegian": "NorwegianBokmal",
                "Persian (Farsi)": "PersianTehran",
                "Belarusian (Belorussian)": "Belarusian",
                "Kurdish": "KurdishCJafiri",
                "Netherlandic": "Dutch",
                "Serbian / Croatian / Bosnian": "SerboCroatian",
            }
            gdf_language["name"] = gdf_language["name"].replace(renaming)
            
            # Load metadata to filter languages
            metadata_path = os.path.join(csv_dir, f"metadata_{tree_name}_with_inventory_delta.csv")
            if os.path.exists(metadata_path):
                obs_df = pd.read_csv(metadata_path)
                valid_langs = set(obs_df.get("language", []))
                gdf_language = gdf_language[gdf_language["name"].isin(valid_langs)].copy()
            
            # Clip to ROI
            gdf_language = gpd.clip(gdf_language, ROI_BOX)
            
            if not gdf_language.empty:
                language_union = gdf_language.unary_union
        except Exception as e:
            print(f"Warning: Could not load language GeoJSON: {e}")
    
    # Load basemap
    if show_basemap:
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            land_union = world.unary_union
            land_gdf = gpd.GeoDataFrame(geometry=[land_union], crs=world.crs)
            land_roi = gpd.clip(land_gdf, ROI_BOX)
            
            # Ocean background
            ax.set_facecolor("#d5e9ff")
            
            # Land (white)
            land_roi.plot(ax=ax, color='white', edgecolor='none', zorder=0)
            
            # Country boundaries (light gray)
            land_roi.boundary.plot(ax=ax, color='lightgray', linewidth=0.6, zorder=0.5)
        except Exception as e:
            print(f"Warning: Could not load basemap: {e}")
    
    # Create masked spline surface (only within language polygons)
    z_masked = z_grid.copy()
    if language_union is not None:
        language_prep = prep(language_union)
        grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        
        # Mask points outside language polygons
        in_language_mask = np.array([
            language_prep.contains(Point(p[0], p[1])) for p in grid_points
        ]).reshape(lon_grid.shape)
        
        z_masked = np.where(in_language_mask, z_grid, np.nan)
    
    # Spline surface with pcolormesh (smoother for masked data)
    vmax = max(abs(np.nanmin(z_masked)), abs(np.nanmax(z_masked)))
    if np.isnan(vmax) or vmax == 0:
        vmax = max(abs(z.min()), abs(z.max()))
    
    mesh = ax.pcolormesh(lon_grid, lat_grid, z_masked, shading="auto",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, zorder=1)
    
    # Language polygon boundaries (white outlines)
    if gdf_language is not None and not gdf_language.empty:
        gdf_language.boundary.plot(ax=ax, color='white', linewidth=0.8, zorder=2)
    
    # Add observation points
    if show_observations:
        metadata_path = os.path.join(csv_dir, f"metadata_{tree_name}_with_inventory_delta.csv")
        if os.path.exists(metadata_path):
            obs_df = pd.read_csv(metadata_path)
            if "longitude" in obs_df.columns and "latitude" in obs_df.columns:
                ax.scatter(obs_df["longitude"], obs_df["latitude"], 
                          c="white", s=60, edgecolor="black", linewidth=0.7,
                          alpha=1.0, zorder=10, label="Language locations")
    
    # Set axis limits
    ax.set_xlim(ROI_MINX, ROI_MAXX)
    ax.set_ylim(ROI_MINY, ROI_MAXY)
    
    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label("Spline Effect (log scale)", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)
    
    # Load regression results for MCMC diagnostics
    reg_results = load_regression_results(csv_dir, model_type, coord_method)
    
    # Build diagnostics subtitle
    diag_subtitle = ""
    if not reg_results.empty and 'min_bulk_ess' in reg_results.columns:
        tree_reg = reg_results[reg_results["tree"] == tree_name]
        if not tree_reg.empty:
            n_div = int(tree_reg["n_divergent"].iloc[0]) if "n_divergent" in tree_reg.columns else 0
            rhat = tree_reg["max_rhat"].iloc[0] if "max_rhat" in tree_reg.columns else 1.0
            ess_all = int(tree_reg["min_bulk_ess"].iloc[0])
            ess_fix = int(tree_reg["min_bulk_ess_fixed"].iloc[0]) if "min_bulk_ess_fixed" in tree_reg.columns else ess_all
            diag_subtitle = f"ESS_fix={ess_fix}/all={ess_all}, R̂={rhat:.2f}, divergences={n_div}"
    
    # Professional styling
    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.6)
    ax.set_title(f"Spatial Spline Effect: {tree_display}\n({config['display_name']})" + 
                 (f"\n{diag_subtitle}" if diag_subtitle else ""),
                 fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = config["file_suffix"] or "_mainonly"
        path = os.path.join(output_dir, f"spline_surface{suffix}_{tree_name}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


# ============================================================================
# SPLINE SURFACE WITH COEFFICIENT SIGNIFICANCE OVERLAY
# ============================================================================

def plot_spline_with_significance(
    model_type: str = "main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    tree_name: str = "heggarty2024",
    coord_method: str = "standard",
    tree_labels: dict = None,
    output_dir: str = "speech_phylo/final_spline_figuresv2",
    show_observations: bool = True,
    show_basemap: bool = True,
    prob_threshold: float = 0.9,
    save: bool = False,
) -> None:
    """
    Plot per-location spline significance using same style as spline surface plot.
    
    prob_threshold: Proportion of posterior samples that must be on one side of 0.
                   (0.9 = 90% of samples must be positive or 90% negative)
    
    At each grid point, shows binary significance:
    - Blue (0): Less than prob_threshold of samples on one side
    - Red (1): At least prob_threshold of samples on one side (positive or negative)
    
    Uses same language polygon masking and styling as plot_spline_surface.
    """
    import geopandas as gpd
    from shapely.geometry import box, Point
    from shapely.prepared import prep
    
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    
    # Load spline surface
    df = load_spline_surface(csv_dir, model_type, tree_name, coord_method)
    if df.empty:
        print(f"No spline surface data for {tree_name}")
        return
    
    tree_display = tree_labels.get(tree_name, tree_name)
    
    # Get grid data
    lon = df["longitude"].values
    lat = df["latitude"].values
    
    # Use probability column if available, otherwise fall back to CI-based
    if "spline_prob_positive" in df.columns:
        prob_positive = df["spline_prob_positive"].values
        # Convert to percentage (0-100%)
        prob_pct = prob_positive * 100
        method_label = "% posterior samples > 0"
    else:
        # Fall back to mean-based estimate if prob column not available
        print(f"Warning: spline_prob_positive column not found")
        prob_pct = np.full_like(lon, 50.0)  # Default to 50%
        method_label = "probability unavailable"
    
    n_grid = int(np.sqrt(len(lon)))
    
    try:
        lon_grid = lon.reshape(n_grid, n_grid)
        lat_grid = lat.reshape(n_grid, n_grid)
        z_grid = prob_pct.reshape(n_grid, n_grid)
    except ValueError:
        print(f"Cannot reshape data for {tree_name}")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # ROI bounds using Ireland to Bangladesh extent
    PAD_DEG = 15
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ire = world.loc[world['name'] == 'Ireland'].total_bounds
    bgd = world.loc[world['name'] == 'Bangladesh'].total_bounds
    ROI_MINX = min(ire[0], bgd[0]) - PAD_DEG
    ROI_MAXX = max(ire[2], bgd[2]) + PAD_DEG
    ROI_MINY = min(ire[1], bgd[1]) - PAD_DEG
    ROI_MAXY = max(ire[3], bgd[3]) + PAD_DEG
    ROI_BOX = box(ROI_MINX, ROI_MINY, ROI_MAXX, ROI_MAXY)
    
    # Load language polygons from geojson (same as spline plot)
    gdf_language = None
    language_union = None
    # GeoJSON is in parent directory of csv_dir
    geojson_path = os.path.join(os.path.dirname(csv_dir), "dataset.geojson")
    if not os.path.exists(geojson_path):
        geojson_path = os.path.join(csv_dir, "dataset.geojson")  # fallback
    
    if os.path.exists(geojson_path):
        try:
            gdf_language = gpd.read_file(geojson_path)
            
            renaming = {
                "Punjabi (Panjabi)": "Punjabi",
                "Norwegian": "NorwegianBokmal",
                "Persian (Farsi)": "PersianTehran",
                "Belarusian (Belorussian)": "Belarusian",
                "Kurdish": "KurdishCJafiri",
                "Netherlandic": "Dutch",
                "Serbian / Croatian / Bosnian": "SerboCroatian",
            }
            gdf_language["name"] = gdf_language["name"].replace(renaming)
            
            metadata_path = os.path.join(csv_dir, f"metadata_{tree_name}_with_inventory_delta.csv")
            if os.path.exists(metadata_path):
                obs_df = pd.read_csv(metadata_path)
                valid_langs = set(obs_df.get("language", []))
                gdf_language = gdf_language[gdf_language["name"].isin(valid_langs)].copy()
            
            gdf_language = gpd.clip(gdf_language, ROI_BOX)
            
            if not gdf_language.empty:
                language_union = gdf_language.unary_union
        except Exception as e:
            print(f"Warning: Could not load language GeoJSON: {e}")
    
    # Load basemap (same as spline plot)
    if show_basemap:
        try:
            land_union = world.unary_union
            land_gdf = gpd.GeoDataFrame(geometry=[land_union], crs=world.crs)
            land_roi = gpd.clip(land_gdf, ROI_BOX)
            
            ax.set_facecolor("#d5e9ff")  # Ocean
            land_roi.plot(ax=ax, color='white', edgecolor='none', zorder=0)
            land_roi.boundary.plot(ax=ax, color='lightgray', linewidth=0.6, zorder=0.5)
        except Exception:
            pass
    
    # Mask probability grid to language regions
    z_masked = z_grid.copy()
    if language_union is not None:
        language_prep = prep(language_union)
        grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        
        in_language_mask = np.array([
            language_prep.contains(Point(p[0], p[1])) for p in grid_points
        ]).reshape(lon_grid.shape)
        
        z_masked = np.where(in_language_mask, z_grid, np.nan)
    
    # Bin the probability into discrete categories
    from matplotlib.colors import BoundaryNorm, ListedColormap
    
    # Define 8 bins: 12.5% intervals
    bounds = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
    # Colors: dark blue → light blue → white area → light red → dark red
    colors = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", 
              "#fcbba1", "#fc9272", "#cb181d", "#67000d"]
    cmap_binned = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap_binned.N)
    
    mesh = ax.pcolormesh(lon_grid, lat_grid, z_masked, shading="auto",
                         cmap=cmap_binned, norm=norm, zorder=1)
    
    # Language polygon boundaries (white outlines)
    if gdf_language is not None and not gdf_language.empty:
        gdf_language.boundary.plot(ax=ax, color='white', linewidth=0.8, zorder=2)
    
    # Add observation points (same as spline plot)
    if show_observations:
        metadata_path = os.path.join(csv_dir, f"metadata_{tree_name}_with_inventory_delta.csv")
        if os.path.exists(metadata_path):
            obs_df = pd.read_csv(metadata_path)
            if "longitude" in obs_df.columns and "latitude" in obs_df.columns:
                ax.scatter(obs_df["longitude"], obs_df["latitude"], 
                          c="white", s=60, edgecolor="black", linewidth=0.7,
                          alpha=1.0, zorder=10, label="Language locations")
    
    ax.set_xlim(ROI_MINX, ROI_MAXX)
    ax.set_ylim(ROI_MINY, ROI_MAXY)
    
    # Colorbar with bin labels
    tick_positions = [6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75]
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9, ticks=tick_positions)
    cbar.ax.set_yticklabels([
        "87.5-100% below 0",
        "75-87.5% below 0",
        "62.5-75% below 0", 
        "50-62.5% below 0",
        "50-62.5% above 0",
        "62.5-75% above 0",
        "75-87.5% above 0",
        "87.5-100% above 0"
    ])
    cbar.set_label("Posterior Probability", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    
    # Professional styling (same as spline plot)
    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.6)
    
    ax.set_title(f"Posterior Probability of Positive Spline Effect: {tree_display}\n" +
                 f"({config['display_name']})",
                 fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = config["file_suffix"] or "_mainonly"
        path = os.path.join(output_dir, f"spline_significance{suffix}_{tree_name}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


def plot_all_significance_plots(
    model_types: list = None,
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    output_dir: str = "speech_phylo/final_spline_figuresv2",
    save: bool = True,
) -> list:
    """
    Generate per-location spline significance plots for all trees and model types.
    Returns list of saved file paths.
    """
    if model_types is None:
        model_types = ["main_only", "secondorder", "allinteractions"]
    
    trees = ["heggarty2024", "long_v3_CCD0.0.25"]
    saved_paths = []
    
    for model_type in model_types:
        for tree_name in trees:
            print(f"Generating: {model_type} × {tree_name}")
            try:
                plot_spline_with_significance(
                    model_type=model_type,
                    csv_dir=csv_dir,
                    tree_name=tree_name,
                    coord_method=coord_method,
                    output_dir=output_dir,
                    save=save,
                    prob_threshold=0.9,
                )
                if save:
                    config = MODEL_CONFIGS[model_type]
                    suffix = config["file_suffix"] or "_mainonly"
                    path = os.path.join(output_dir, f"spline_significance{suffix}_{tree_name}_{coord_method}.svg")
                    saved_paths.append(path)
            except Exception as e:
                print(f"  Error: {e}")
    
    return saved_paths


# ============================================================================
# EFFECTS COMPARISON (COEFFICIENT VIOLIN PLOT)
# ============================================================================

def plot_effects_comparison(
    model_type: str = "main_only",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    tree_labels: dict = None,
    output_dir: str = "speech_phylo/final_figuresv2",
    save: bool = False,
) -> None:
    """Plot coefficient posterior distributions as violin plots with 95% CI.
    
    Uses posterior samples for true violin plots. Falls back to bar visualization
    if samples not available.
    """
    if tree_labels is None:
        tree_labels = TREE_LABELS
    
    config = MODEL_CONFIGS[model_type]
    coeff_order = config.get("coeff_order", ["log_n_speakers", "n_segments", "delta"])
    
    # Try to load coefficient posterior samples
    coef_samples = load_coef_samples(csv_dir, model_type, coord_method)
    
    if coef_samples.empty:
        print(f"No coefficient samples found. Run R script first to generate coef_samples_*.csv files.")
        return
    
    coef_samples["tree_display"] = coef_samples["tree"].map(tree_labels).fillna(coef_samples["tree"])
    trees = list(coef_samples["tree_display"].unique())
    
    # Ensure Speech is on top (last in list for matplotlib)
    if "Speech" in trees:
        trees = [t for t in trees if t != "Speech"] + ["Speech"]
    
    # Map column names to display names
    # Column names in CSV are like: Intercept, log_n_speakers_norm, n_segments_norm, delta_norm
    col_mapping = {
        "log_n_speakers_norm": "log_n_speakers",
        "n_segments_norm": "n_segments", 
        "delta_norm": "delta",
        "log_n_speakers_norm.n_segments_norm": "log_n_speakers×n_segments",
        "log_n_speakers_norm.delta_norm": "log_n_speakers×delta",
        "n_segments_norm.delta_norm": "n_segments×delta",
        "log_n_speakers_norm.n_segments_norm.delta_norm": "Triple Interaction",
    }
    
    n_coeffs = len(coeff_order)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.7 * n_coeffs)))
    
    offset = 0.2
    
    for t_idx, tree_display in enumerate(trees):
        tree_samples = coef_samples[coef_samples["tree_display"] == tree_display]
        color = TREE_COLORS.get(tree_display, "#888888")
        
        violin_data = []
        for coeff in coeff_order:
            # Find the column name
            col_name = None
            for csv_col, display in col_mapping.items():
                if display == coeff and csv_col in tree_samples.columns:
                    col_name = csv_col
                    break
            
            if col_name and col_name in tree_samples.columns:
                vals = tree_samples[col_name].dropna().values
                violin_data.append(vals if len(vals) > 0 else np.array([0]))
            else:
                violin_data.append(np.array([0]))
        
        positions = np.arange(n_coeffs) + (t_idx - 0.5) * offset * 2
        
        # Create violin plots without extrema
        parts = ax.violinplot(
            violin_data, positions=positions, vert=False,
            showmeans=False, showmedians=False, showextrema=False, widths=0.35,
        )
        
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
        
        # Add actual 95% CI lines with endpoint caps, mean marker, and text labels
        for i, (pos, data) in enumerate(zip(positions, violin_data)):
            if len(data) > 1:
                mean_val = np.mean(data)
                q2_5 = np.percentile(data, 2.5)
                q97_5 = np.percentile(data, 97.5)
                cap_height = 0.12  # Height of endpoint caps (increased)
                
                # Draw 95% CI line (thicker)
                ax.plot([q2_5, q97_5], [pos, pos], color=color, linewidth=2.5, alpha=0.9)
                # Draw endpoint caps (thicker)
                ax.plot([q2_5, q2_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=2.5, alpha=0.9)
                ax.plot([q97_5, q97_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=2.5, alpha=0.9)
                # Draw mean marker (thicker)
                ax.scatter([mean_val], [pos], color=color, s=40, zorder=5, 
                          marker='|', linewidth=3)
                
                # Add text labels for CI endpoints
                ax.text(q2_5, pos + 0.15, f'{q2_5:.2f}', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
                ax.text(q97_5, pos + 0.15, f'{q97_5:.2f}', ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.9)
                
    # Add subtle highlight band for delta row to emphasize key finding
    if "delta" in coeff_order:
        delta_idx = coeff_order.index("delta")
        ax.axhspan(delta_idx - 0.4, delta_idx + 0.4, color='#fff3cd', alpha=0.4, zorder=0)
    
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
    
    # Professional legend
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
        suffix = config["file_suffix"] or "_mainonly"
        path = os.path.join(output_dir, f"effects_comparison{suffix}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def plot_all_models(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    save: bool = False,
) -> None:
    """Plot violin, stacked bar, and effects comparison plots for all three model types."""
    for model_type in ["main_only", "secondorder", "allinteractions"]:
        print(f"\n{'='*70}")
        print(f"Model: {MODEL_CONFIGS[model_type]['display_name']}")
        print(f"{'='*70}")
        
        plot_variance_violin(model_type=model_type, csv_dir=csv_dir,
                             coord_method=coord_method, save=save, output_dir="speech_phylo/final_spline_figuresv2")
        # plot_variance_stacked(model_type=model_type, csv_dir=csv_dir,
        #                       coord_method=coord_method, save=save)
        plot_effects_comparison(model_type=model_type, csv_dir=csv_dir,
                                coord_method=coord_method, save=save, output_dir="speech_phylo/final_spline_figuresv2")


def plot_all_spline_surfaces(
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
    save: bool = False,
) -> None:
    """Plot spline surfaces for all model types and trees."""
    trees = ["heggarty2024", "long_v3_CCD0.0.25"]
    
    for model_type in ["main_only", "secondorder", "allinteractions"]:
        for tree in trees:
            plot_spline_surface(model_type=model_type, csv_dir=csv_dir,
                                tree_name=tree, coord_method=coord_method, save=save, output_dir="speech_phylo/final_spline_figuresv2")


def create_combined_pdf(
    output_dir: str = "speech_phylo/final_figuresv2",
    output_filename: str = "all_figures_combined.pdf",
    coord_method: str = "standard",
) -> None:
    """Combine all saved SVG/PNG figures into a single multi-page PDF.
    
    This function:
    1. Reads all saved figures from output_dir
    2. Combines them into a multi-page PDF
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    import io
    
    pdf_path = os.path.join(output_dir, output_filename)
    
    # Find all figure files
    svg_files = sorted(glob.glob(os.path.join(output_dir, f"*_{coord_method}.svg")))
    
    if not svg_files:
        print(f"No SVG files found in {output_dir}")
        return
    
    print(f"Found {len(svg_files)} figures to combine...")
    
    with PdfPages(pdf_path) as pdf:
        for svg_file in svg_files:
            try:
                # Convert SVG to figure
                drawing = svg2rlg(svg_file)
                if drawing is None:
                    print(f"  Skipping {os.path.basename(svg_file)} - could not read")
                    continue
                
                # Get dimensions
                width = drawing.width
                height = drawing.height
                
                # Create figure and render
                fig, ax = plt.subplots(figsize=(width/72, height/72))
                ax.axis('off')
                
                # Render SVG to PDF page
                pdf_io = io.BytesIO()
                renderPDF.drawToFile(drawing, pdf_io, fmt='PDF')
                
                # Alternative: just save figure
                fig.text(0.5, 0.5, os.path.basename(svg_file).replace('.svg', ''), 
                         ha='center', va='center', fontsize=20,
                         transform=ax.transAxes)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  Added: {os.path.basename(svg_file)}")
                
            except Exception as e:
                print(f"  Error processing {svg_file}: {e}")
    
    print(f"\n✅ Combined PDF saved to: {pdf_path}")


def create_combined_pdf_simple(
    output_dir: str = "speech_phylo/final_figuresv2",
    output_filename: str = "all_figures_combined.pdf",
    csv_dir: str = "speech_phylo/final_phyloregression_results",
    coord_method: str = "standard",
) -> None:
    """Combine saved SVG files into a single PDF, grouped by plot type.
    
    Order: All variance violins -> All variance stacked -> All effects -> All splines
    """
    import subprocess
    import shutil
    
    # Check for cairosvg or inkscape
    has_cairosvg = False
    has_inkscape = shutil.which("inkscape") is not None
    
    try:
        import cairosvg
        has_cairosvg = True
    except ImportError:
        pass
    
    if not has_cairosvg and not has_inkscape:
        print("Warning: Neither cairosvg nor inkscape found.")
        print("Install cairosvg: pip install cairosvg")
        print("Or install inkscape: apt install inkscape")
        return
    
    # Define plot type order
    plot_types = [
        ("variance_violin", "Variance Violin Plots"),
        ("variance_stacked", "Variance Stacked Bar Plots"),
        ("effects_comparison", "Effects Comparison Plots"),
        ("spline_surface", "Spline Surface Plots"),
        ("spline_significance", "Spline + Significance Plots"),
    ]
    
    # Model order
    model_order = ["mainonly", "secondorder", "allinteractions"]
    
    # Collect and sort SVG files by plot type
    svg_files_ordered = []
    
    for plot_prefix, plot_name in plot_types:
        matching = []
        for model in model_order:
            pattern = f"{plot_prefix}*{model}*{coord_method}.svg"
            files = sorted(glob.glob(os.path.join(output_dir, pattern)))
            matching.extend(files)
        
        # Also get any that don't match model patterns (e.g. splines with tree names)
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
    
    # Convert SVGs to temporary PDFs
    temp_pdfs = []
    temp_dir = os.path.join(output_dir, ".temp_pdfs")
    os.makedirs(temp_dir, exist_ok=True)
    
    for svg_path in svg_files_ordered:
        pdf_name = os.path.basename(svg_path).replace(".svg", ".pdf")
        pdf_path = os.path.join(temp_dir, pdf_name)
        
        try:
            if has_inkscape:
                # Prefer inkscape - more robust for complex SVGs
                result = subprocess.run(
                    ["inkscape", svg_path, "--export-type=pdf", f"--export-filename={pdf_path}"],
                    capture_output=True
                )
                if result.returncode == 0:
                    temp_pdfs.append(pdf_path)
                    print(f"  ✔ Converted: {os.path.basename(svg_path)}")
                else:
                    print(f"  ✗ Inkscape error for {os.path.basename(svg_path)}")
            elif has_cairosvg:
                cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
                temp_pdfs.append(pdf_path)
                print(f"  ✔ Converted: {os.path.basename(svg_path)}")
        except subprocess.TimeoutExpired:
            print(f"  ⏱ Timeout (skipped): {os.path.basename(svg_path)}")
        except Exception as e:
            print(f"  ✗ Error converting {os.path.basename(svg_path)}: {e}")
    
    if not temp_pdfs:
        print("No PDFs were created")
        return
    
    # Combine PDFs using PyPDF2 or pikepdf
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
            # Fallback: use ghostscript
            if shutil.which("gs"):
                cmd = ["gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
                       f"-sOutputFile={output_path}"] + temp_pdfs
                subprocess.run(cmd, check=True)
                print(f"\n✅ Combined PDF saved to: {output_path}")
            else:
                print("Error: No PDF merger available. Install PyPDF2 or pikepdf:")
                print("  pip install PyPDF2")
    
    # Clean up temp files
    import shutil as shutil_mod
    shutil_mod.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    plot_all_models(csv_dir="speech_phylo/final_phyloregression_results", save=True)
    # plot_all_spline_surfaces(csv_dir="speech_phylo/final_phyloregression_results", save=True)
    # plot_all_significance_plots(csv_dir="speech_phylo/final_phyloregression_results", save=True, output_dir="speech_phylo/final_spline_figuresv2")
    
    # Create combined PDF with all figures
    # create_combined_pdf_simple(
    #     output_dir="speech_phylo/final_spline_figuresv2",
    #     output_filename="all_spline_figures_combined.pdf",
    #     csv_dir="speech_phylo/final_phyloregression_results"
    # )
