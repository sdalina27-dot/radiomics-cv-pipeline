#!/usr/bin/env python3
"""
plot_old_vs_new_heatmap.py
==========================
Manuscript-style heatmap for the GA old-vs-new LOO comparison (mirrors
plot_mi_ga_heatmap.py). Rows = bin width (5..50), columns = old/new GA x
SS/RS/PT, cell = pooled LOO AUC (SVM, k=4). A star marks the better run within a
scaler (DeLong, two-sided, paired 80 cases): `*` p<0.05, `**` p<0.01, `***` p<0.001.

Reads delong_results_{S}.csv produced by analyze_old_vs_new_GA.py.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "fig_ga_old_vs_new_loomap.png"

SCALERS = ["StandardScaler", "RobustScaler", "PowerTransformer"]
SHORT = {"StandardScaler": "SS", "RobustScaler": "RS", "PowerTransformer": "PT"}
BW = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

COLUMNS = []
for s in SCALERS:
    COLUMNS.append((s, "old"))
    COLUMNS.append((s, "new"))


def load_delong(s: str) -> pd.DataFrame:
    return pd.read_csv(HERE / f"delong_results_{SHORT[s]}.csv").set_index("bw")


def star_for_p(p: float) -> str:
    if pd.isna(p):
        return ""
    for thr, st in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p < thr:
            return st
    return ""


def compute_matrix():
    auc = pd.DataFrame(index=BW, columns=[f"{t} {SHORT[s]}" for s, t in COLUMNS], dtype=float)
    mark = pd.DataFrame("", index=BW, columns=auc.columns)

    for s in SCALERS:
        d = load_delong(s)
        for bw in BW:
            a_old = float(d.loc[bw, "auc_old"])
            a_new = float(d.loc[bw, "auc_new"])
            p = float(d.loc[bw, "delong_p"])
            auc.loc[bw, f"old {SHORT[s]}"] = a_old
            auc.loc[bw, f"new {SHORT[s]}"] = a_new
            st = star_for_p(p)
            if st:
                better = "old" if a_old > a_new else "new"
                mark.loc[bw, f"{better} {SHORT[s]}"] = st
    return auc, mark


def main():
    for f in font_manager.findSystemFonts():
        if "arial" in f.lower():
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = "Arial"

    auc, mark = compute_matrix()

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    data = auc.values.astype(float)
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.5, vmax=0.9, aspect="auto")

    ax.set_xticks(range(len(auc.columns)))
    ax.set_xticklabels(auc.columns, fontsize=13, color="black")
    ax.set_yticks(range(len(auc.index)))
    ax.set_yticklabels([f"BW {b}" for b in auc.index], fontsize=12, color="black")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, colors="black")
    ax.set_xlabel("GA configuration / Scaler", fontsize=14, labelpad=10, color="black")
    ax.set_ylabel("Bin Width", fontsize=14, color="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    for i, bw in enumerate(auc.index):
        for j, col in enumerate(auc.columns):
            val = data[i, j]
            m = mark.loc[bw, col]
            txt_color = "white" if val > 0.8 else "black"
            ax.text(j, i, f"{val:.3f}\n{m}", ha="center", va="center",
                    fontsize=11, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pooled AUC (SVM, k=4)", fontsize=13, color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="black")

    ax.set_title("GA old (5x8, 3 inner) vs new (5x4, 1 inner) — LOO pooled AUC",
                 fontsize=14, color="black")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
