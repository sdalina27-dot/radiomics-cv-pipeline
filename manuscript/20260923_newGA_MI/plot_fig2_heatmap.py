#!/usr/bin/env python3
"""
plot_fig2_heatmap.py
====================
Figure 2. Pooled LOO AUC (SVM, k=4) for the current GA (no-MI-init, 4-seed/1-inner)
and MI (4-seed) runs, 10 bin widths x 3 scalers. Matches the manuscript heatmap style.

Stars mark the better method within a scaler (DeLong, two-sided); daggers mark an
SS/RS cell better than the same method under PowerTransformer (†), or a PT cell
better than the same method under SS/RS (‡).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import ga_mi_config as cfg

OUT = Path(__file__).resolve().parent / "figure2_pooled_auc_heatmap.png"
COLUMNS = [(s, m) for s in cfg.SCALERS for m in ("GA", "MI")]


def build_matrix():
    auc = pd.DataFrame(index=cfg.BWS,
                       columns=[f"{m} {cfg.SHORT[s]}" for s, m in COLUMNS], dtype=float)
    mark = pd.DataFrame("", index=cfg.BWS, columns=auc.columns)
    for s in cfg.SCALERS:
        ga, mi = cfg.GA_NEW.format(s=s), cfg.MI_NEW.format(s=s)
        for bw in cfg.BWS:
            _, p, a_ga, a_mi = cfg.delong_pair(ga, mi, bw)
            auc.loc[bw, f"GA {cfg.SHORT[s]}"] = cfg.bootstrap_all(ga)[bw][0]
            auc.loc[bw, f"MI {cfg.SHORT[s]}"] = cfg.bootstrap_all(mi)[bw][0]
            st = cfg.star_for_p(p)
            if st:
                mark.loc[bw, f"{'GA' if a_ga > a_mi else 'MI'} {cfg.SHORT[s]}"] = st
    for tpl, meth in [(cfg.GA_NEW, "GA"), (cfg.MI_NEW, "MI")]:
        for s in ("StandardScaler", "RobustScaler"):
            m, pt = tpl.format(s=s), tpl.format(s="PowerTransformer")
            for bw in cfg.BWS:
                _, p, a_m, a_p = cfg.delong_pair(m, pt, bw)
                if not cfg.dagger_for_p(p):
                    continue
                if a_m > a_p:
                    mark.loc[bw, f"{meth} {cfg.SHORT[s]}"] += cfg.dagger_for_p(p)
                else:
                    mark.loc[bw, f"{meth} PT"] += cfg.dagger_for_p(p).replace("†", "‡")
    return auc, mark


def main():
    for f in font_manager.findSystemFonts():
        if "arial" in f.lower():
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = "Arial"

    auc, mark = build_matrix()
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    data = auc.values.astype(float)
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.5, vmax=0.9, aspect="auto")

    ax.set_xticks(range(len(auc.columns)))
    ax.set_xticklabels(auc.columns, fontsize=13, color="black")
    ax.set_yticks(range(len(auc.index)))
    ax.set_yticklabels([f"BW {b}" for b in auc.index], fontsize=12, color="black")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, colors="black")
    ax.set_xlabel("Feature-Selection Method / Scaler", fontsize=14, labelpad=10, color="black")
    ax.set_ylabel("Bin Width", fontsize=14, color="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    for i, bw in enumerate(auc.index):
        for j, col in enumerate(auc.columns):
            val = data[i, j]
            m = mark.loc[bw, col]
            txt_color = "white" if val > 0.8 else "black"
            ax.text(j, i, f"{val:.3f}\n{m}", ha="center", va="center", fontsize=11, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pooled AUC (SVM, k=4)", fontsize=13, color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="black")

    ax.set_title("MI vs GA — LOO pooled AUC (GA no-MI-init, 4 seeds)", fontsize=15, color="black")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
