#!/usr/bin/env python3
"""
plot_fig3_roc.py
================
Figure 3. ROC curves of radiomics (current GA/MI runs at the matched configs
RobustScaler BW10 and StandardScaler BW20) and clinical models (LOO, SVM).
Clinical curves are unchanged from manuscript/20260906.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, auc

import ga_mi_config as cfg

OUT = Path(__file__).resolve().parent / "figure3_roc.png"

RADIO = [
    ("GA RS, BW10", cfg.GA_NEW.format(s="RobustScaler"), 10, "#d62728", "-"),
    ("MI RS, BW10", cfg.MI_NEW.format(s="RobustScaler"), 10, "#ff7f0e", "-"),
    ("GA SS, BW20", cfg.GA_NEW.format(s="StandardScaler"), 20, "#1f77b4", "-"),
    ("MI SS, BW20", cfg.MI_NEW.format(s="StandardScaler"), 20, "#2ca02c", "-"),
]
CLINICAL = [
    ("LVSVi", "LVSVi", "#9467bd", "--"),
    ("LVWT", "LVWT", "#8c564b", "-."),
    ("Clinical4", "Volume4", "#e377c2", "--"),
]
CLINICAL_SCALER = "RobustScaler"


def load_clinical(feature_set):
    df = pd.read_csv(cfg.CLINICAL_PRED)
    df = df[(df["classifier"] == "svm") & (df["scaler"] == CLINICAL_SCALER)
            & (df["feature_set"] == feature_set)]
    return df["y_true"].values, df["y_score"].values


def load_clinical_ci(feature_set):
    df = pd.read_csv(cfg.CLINICAL_SUMMARY)
    r = df[(df["classifier"] == "svm") & (df["scaler"] == CLINICAL_SCALER)
           & (df["feature_set"] == feature_set)].iloc[0]
    return float(r["auc_lo"]), float(r["auc_hi"])


def main():
    for f in font_manager.findSystemFonts():
        if "arial" in f.lower():
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for label, exp, bw, color, ls in RADIO:
        yt, ys = cfg.load_scores(exp, bw)
        fpr, tpr, _ = roc_curve(yt, ys)
        lo, hi = cfg.bootstrap_all(exp)[bw][1], cfg.bootstrap_all(exp)[bw][2]
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2,
                label=f"{label}: AUC = {auc(fpr, tpr):.3f} (95% CI {lo:.3f}-{hi:.3f})")
    for label, fset, color, ls in CLINICAL:
        yt, ys = load_clinical(fset)
        fpr, tpr, _ = roc_curve(yt, ys)
        lo, hi = load_clinical_ci(fset)
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2.2,
                label=f"{label}: AUC = {auc(fpr, tpr):.3f} (95% CI {lo:.3f}-{hi:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("1 - Specificity", fontsize=14)
    ax.set_ylabel("Sensitivity", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_title("ROC — Radiomics (GA/MI) vs Clinical Models (LOO, SVM)", fontsize=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
