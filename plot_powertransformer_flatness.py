#!/usr/bin/env python3
"""
Visualize the distribution of Cine_3D_ES_original_shape_Sphericity
before and after PowerTransformer (Yeo-Johnson, BW=10).

For presentation use - generates a clean, publication-ready figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH = "/home/ser/pipeline/data/Cine_output_20260606_binWidth_30.csv"
FEATURE  = "Cine_3D_ES_original_shape_Sphericity"
OUT_PATH = "/home/ser/pipeline/powertransformer_sphericity_bw30.png"

# ── Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
raw = df[FEATURE].values.reshape(-1, 1)

# ── Apply PowerTransformer (Yeo-Johnson, standardize=True) ─────────────
pt = PowerTransformer(method="yeo-johnson", standardize=True)
transformed = pt.fit_transform(raw)

raw_flat   = raw.ravel()
trans_flat = transformed.ravel()

# ── Stats ───────────────────────────────────────────────────────────────
def desc(x):
    return {
        "mean":  np.mean(x),
        "std":   np.std(x, ddof=1),
        "skew":  stats.skew(x),
        "min":   np.min(x),
        "max":   np.max(x),
    }

s_raw = desc(raw_flat)
s_tr  = desc(trans_flat)

# ── Shared axis limits ──────────────────────────────────────────────────
# Use the same y-axis (count) range for both plots
all_data = np.concatenate([raw_flat, trans_flat])
y_max = len(raw_flat) * 0.30  # ~30% of n for the tallest bin
    
# ── Plot ─�───────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before
ax = axes[0]
sns.histplot(raw_flat, bins=20, color="#4C72B0", ax=ax,
             alpha=0.7, edgecolor="white", linewidth=0.5)
ax.axvline(s_raw["mean"], color="#C44E52", linestyle="--", linewidth=2,
           label=f"Mean = {s_raw['mean']:.3f}")
ax.set_title("Before PowerTransformer", fontsize=14, fontweight="bold")
ax.set_xlabel(f"{FEATURE}", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_ylim(0, y_max)
ax.legend(fontsize=11, loc="upper right")

# After
ax = axes[1]
sns.histplot(trans_flat, bins=20, color="#55A868", ax=ax,
             alpha=0.7, edgecolor="white", linewidth=0.5)
ax.axvline(s_tr["mean"], color="#C44E52", linestyle="--", linewidth=2,
           label=f"Mean = {s_tr['mean']:.3f}")
ax.set_title("After PowerTransformer\n(Yeo-Johnson, standardized)", fontsize=14, fontweight="bold")
ax.set_xlabel("Transformed value", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_ylim(0, y_max)
ax.legend(fontsize=11, loc="upper right")

# ── Annotation box (top-right corner, small) ────────────────────────────
info_text = (
    f"n = {len(raw_flat)}    BW = 10\n\n"
    f"Before:  skew = {s_raw['skew']:+.3f}  |  std = {s_raw['std']:.3f}\n"
    f"After:   skew = {s_tr['skew']:+.3f}  |  std = {s_tr['std']:.3f}"
)
fig.text(0.98, 0.95, info_text, ha="right", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="#999999", alpha=0.9),
         family="monospace")

fig.suptitle(f"PowerTransformer Effect on {FEATURE}\n(BW=10, n=80 cases)",
             fontsize=15, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
print(f"\nBefore: skew={s_raw['skew']:+.4f}, std={s_raw['std']:.4f}, mean={s_raw['mean']:.4f}")
print(f"After:  skew={s_tr['skew']:+.4f}, std={s_tr['std']:.4f}, mean={s_tr['mean']:.4f}")
