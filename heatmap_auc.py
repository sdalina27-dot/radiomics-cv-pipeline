#!/usr/bin/env python3
"""
heatmap_auc.py
==============
Generate AUC heatmap from pipeline results.

Usage:
    conda activate gpu-radiomics
    python heatmap_auc.py

Configuration:
    Modify WORKSPACE_DIR, CLASSIFIER, etc. at the top of the file.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── User-configurable settings ───────────────────────────────────────────────
WORKSPACE_DIR: Path = Path(__file__).parent / "workspace" / "OO+wavelet_r0.7_RobustScaler_top5_NoRdundancyInFitness"
RESULTS_FILE: str = "pipeline_results.csv"
CLASSIFIER: str = "svm"  # Options: 'svm', 'rf', 'xgboost', 'softvote-3'
# svm_QuantileTransformer / svm_StandardScaler / svm_RobustScaler

# RS filter for the heatmap. List the outer_rs to average over; set to None
# to use all available RS (original behavior).
TARGET_RS: list | None = [42, 123, 456]

# Output settings
OUTPUT_DIR: Path = WORKSPACE_DIR
FIGURE_SIZE: tuple = (12, 8)
DPI: int = 150
CMAP: str = "YlOrRd"  # Colormap: 'viridis', 'plasma', 'inferno', 'magma', 'YlOrRd', 'RdYlGn_r'


def compute_heatmap_data(df: pd.DataFrame, classifier: str,
                         target_rs: list | None = None) -> pd.DataFrame:
    """
    Compute heatmap data: mean AUC ± std across RS.
    
    For each (BW, K):
    1. Compute mean AUC across 8 folds for each RS
    2. Compute std across RS values
    
    If `target_rs` is provided, only those outer_rs are used; otherwise all
    available RS are averaged over.
    
    Returns DataFrame with columns: bw, k, mean_auc, std_auc, display_text
    """
    # Filter by classifier
    df_clf = df[df["classifier"] == classifier].copy()
    
    if df_clf.empty:
        print(f"Error: No data found for classifier '{classifier}'")
        print(f"Available classifiers: {df['classifier'].unique().tolist()}")
        sys.exit(1)
    
    # Filter to the requested RS (None = use all RS)
    if target_rs is not None:
        df_clf = df_clf[df_clf["outer_rs"].isin(target_rs)]
        if df_clf.empty:
            print(f"Error: No data found for classifier '{classifier}' "
                  f"with outer_rs in {target_rs}")
            print(f"Available outer_rs: {sorted(df['outer_rs'].unique().tolist())}")
            sys.exit(1)
    
    # Get unique values
    bws = sorted(df_clf["bw"].unique())
    ks = sorted(df_clf["k"].unique())
    rss = sorted(df_clf["outer_rs"].unique())
    
    print(f"Classifier: {classifier}")
    print(f"BWs: {bws}")
    print(f"K values: {ks}")
    print(f"RS values: {rss}")
    
    # Compute mean AUC per (bw, k, outer_rs) - average across 8 folds
    rs_means = df_clf.groupby(["bw", "k", "outer_rs"])["auc"].mean().reset_index()
    rs_means.rename(columns={"auc": "rs_mean_auc"}, inplace=True)
    
    # Compute mean and std across RS for each (bw, k)
    heatmap_data = rs_means.groupby(["bw", "k"])["rs_mean_auc"].agg(["mean", "std"]).reset_index()
    heatmap_data.rename(columns={"mean": "mean_auc", "std": "std_auc"}, inplace=True)
    
    # Fill NaN std (when only 1 RS) with 0
    heatmap_data["std_auc"] = heatmap_data["std_auc"].fillna(0)
    
    # Create display text
    heatmap_data["display_text"] = heatmap_data.apply(
        lambda r: f"{r['mean_auc']:.3f}\n±{r['std_auc']:.3f}", axis=1
    )
    
    return heatmap_data, bws, ks


def plot_heatmap(heatmap_data: pd.DataFrame, bws: list, ks: list, 
                 classifier: str, output_path: Path):
    """Plot and save heatmap."""
    # Set Arial font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Pivot for heatmap
    pivot_mean = heatmap_data.pivot(index="bw", columns="k", values="mean_auc")
    pivot_std = heatmap_data.pivot(index="bw", columns="k", values="std_auc")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot heatmap with mean values
    sns.heatmap(
        pivot_mean,
        annot=False,  # Disable default annotation
        cmap=CMAP,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Mean AUC"},
        vmin=0.6,
        vmax=1.0,
    )
    
    # Add custom annotation with same style
    for i, bw in enumerate(bws):
        for j, k in enumerate(ks):
            mean_val = pivot_mean.loc[bw, k] if bw in pivot_mean.index and k in pivot_mean.columns else 0
            std_val = pivot_std.loc[bw, k] if bw in pivot_std.index and k in pivot_std.columns else 0
            # Single annotation with mean ± std
            ax.text(
                j + 0.5, i + 0.5,
                f"{mean_val:.3f}\n±{std_val:.3f}",
                ha="center", va="center",
                fontsize=9,
                color="black",
                fontfamily="Arial",
            )
    
    # Labels and title
    ax.set_title(f"AUC Heatmap ({classifier.upper()})\nMean ± Std across RS (std of per-RS 8-fold means)", 
                 fontsize=14, fontweight="bold", fontfamily="Arial")
    ax.set_xlabel("K (Top-K Features)", fontsize=12, fontfamily="Arial")
    ax.set_ylabel("Bin Width (BW)", fontsize=12, fontfamily="Arial")
    
    # Set tick label font
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily="Arial")
    ax.set_yticklabels(ax.get_yticklabels(), fontfamily="Arial", rotation=0)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"\nHeatmap saved to: {output_path}")
    plt.close(fig)


def print_summary_table(heatmap_data: pd.DataFrame, classifier: str):
    """Print summary table to console."""
    print(f"\n{'='*80}")
    print(f"  AUC Summary Table ({classifier.upper()})")
    print(f"{'='*80}")
    
    # Pivot for display
    pivot_mean = heatmap_data.pivot(index="bw", columns="k", values="mean_auc")
    pivot_std = heatmap_data.pivot(index="bw", columns="k", values="std_auc")
    
    # Print header
    ks = sorted(heatmap_data["k"].unique())
    print(f"\n{'BW':>6s}", end="")
    for k in ks:
        print(f"{'k='+str(k):>12s}", end="")
    print()
    print("-" * (6 + 12 * len(ks)))
    
    # Print rows
    for bw in sorted(heatmap_data["bw"].unique()):
        print(f"{bw:>6d}", end="")
        for k in ks:
            mean_val = pivot_mean.loc[bw, k] if bw in pivot_mean.index and k in pivot_mean.columns else np.nan
            std_val = pivot_std.loc[bw, k] if bw in pivot_std.index and k in pivot_std.columns else np.nan
            print(f"{mean_val:.3f}±{std_val:.2f}", end="  ")
        print()
    
    # Find best configuration
    best_idx = heatmap_data["mean_auc"].idxmax()
    best = heatmap_data.loc[best_idx]
    print(f"\nBest: BW={int(best['bw'])}, K={int(best['k'])} → AUC={best['mean_auc']:.4f} ± {best['std_auc']:.4f}")


def main():
    # Load data
    results_path = WORKSPACE_DIR / RESULTS_FILE
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    df = pd.read_csv(results_path)
    print(f"Loaded: {results_path}")
    print(f"Shape: {df.shape}")
    
    # Compute heatmap data
    heatmap_data, bws, ks = compute_heatmap_data(df, CLASSIFIER, TARGET_RS)
    
    # Print summary table
    print_summary_table(heatmap_data, CLASSIFIER)
    
    # Plot heatmap
    output_path = OUTPUT_DIR / f"heatmap_auc_{CLASSIFIER}.png"
    plot_heatmap(heatmap_data, bws, ks, CLASSIFIER, output_path)
    
    # Also save CSV
    csv_path = OUTPUT_DIR / f"heatmap_data_{CLASSIFIER}.csv"
    heatmap_data.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")


if __name__ == "__main__":
    main()
