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
                cap_height = 0.08  # Height of endpoint caps
                
                # Draw 95% CI line
                ax.plot([q2_5, q97_5], [pos, pos], color=color, linewidth=1.5, alpha=0.8)
                # Draw endpoint caps
                ax.plot([q2_5, q2_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=1.5, alpha=0.8)
                ax.plot([q97_5, q97_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=1.5, alpha=0.8)
                # Draw mean marker
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
    
    # Professional legend
    legend_elements = [
        Patch(facecolor=TREE_COLORS.get(t, "#888888"), alpha=0.55,
              edgecolor=TREE_COLORS.get(t, "#888888"), label=t)
        for t in trees
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              frameon=False, handlelength=1.2)
    
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
) -> None:
    """Plot the spline surface for a given tree with optional observation points."""
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    vmax = max(abs(z.min()), abs(z.max()))
    contour = ax.contourf(lon_grid, lat_grid, z_grid, levels=20,
                          cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    
    # Add observation points
    if show_observations:
        metadata_path = os.path.join(csv_dir, f"metadata_{tree_name}_with_inventory_delta.csv")
        if os.path.exists(metadata_path):
            obs_df = pd.read_csv(metadata_path)
            if "longitude" in obs_df.columns and "latitude" in obs_df.columns:
                ax.scatter(obs_df["longitude"], obs_df["latitude"], 
                          c="white", s=40, edgecolor="black", linewidth=0.8,
                          alpha=0.8, zorder=5, label="Observations")
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Spline Effect (log scale)", fontsize=12)
    
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.set_title(f"Spatial Spline Effect: {tree_display}\n({config['display_name']})",
                 fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = config["file_suffix"] or "_mainonly"
        path = os.path.join(output_dir, f"spline_surface{suffix}_{tree_name}_{coord_method}.svg")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    
    plt.show()


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
                cap_height = 0.08  # Height of endpoint caps
                
                # Draw 95% CI line
                ax.plot([q2_5, q97_5], [pos, pos], color=color, linewidth=1.5, alpha=0.8)
                # Draw endpoint caps
                ax.plot([q2_5, q2_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=1.5, alpha=0.8)
                ax.plot([q97_5, q97_5], [pos - cap_height, pos + cap_height], 
                        color=color, linewidth=1.5, alpha=0.8)
                # Draw mean marker
                ax.scatter([mean_val], [pos], color=color, s=30, zorder=5, 
                          marker='|', linewidth=2)
                
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
    plot_all_spline_surfaces(csv_dir="speech_phylo/final_phyloregression_results", save=True)
    
    # Create combined PDF with all figures
    create_combined_pdf_simple(
        output_dir="speech_phylo/final_figuresv2",
        output_filename="all_spline_figures_combined.pdf",
        csv_dir="speech_phylo/final_phyloregression_results"
    )
