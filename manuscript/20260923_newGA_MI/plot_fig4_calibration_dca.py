#!/usr/bin/env python3
"""
plot_fig4_calibration_dca.py
============================
Figure 4. (A) Calibration curves and (B) decision curve analysis for the current
GA/MI runs at the matched configs plus the clinical models. Clinical curves are
unchanged from manuscript/20260906.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.integrate import trapezoid

import ga_mi_config as cfg

OUT = Path(__file__).resolve().parent / "figure4_calibration_dca.png"

N_BINS = 5
THRESHOLDS = np.linspace(0.05, 0.85, 161)

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


def load_models():
    models = []
    for label, exp, bw, color, ls in RADIO:
        yt, ys = cfg.load_scores(exp, bw)
        models.append((label, yt.astype(int), ys.astype(float), color, ls))
    cp = pd.read_csv(cfg.CLINICAL_PRED)
    for label, fset, color, ls in CLINICAL:
        sub = cp[(cp["classifier"] == "svm") & (cp["scaler"] == CLINICAL_SCALER)
                 & (cp["feature_set"] == fset)].sort_values("case_id")
        models.append((label, sub["y_true"].values.astype(int), sub["y_score"].values.astype(float), color, ls))
    return models


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calib_bins(y, p, n=N_BINS):
    order = np.argsort(p, kind="mergesort")
    yt, ys = y[order], p[order]
    groups = np.array_split(np.arange(len(ys)), n)
    return np.array([ys[g].mean() for g in groups]), np.array([yt[g].mean() for g in groups])


def net_benefit(y, p, thr=THRESHOLDS):
    n = len(y)
    nb = np.empty(thr.shape)
    for i, pt in enumerate(thr):
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb[i] = tp / n - (fp / n) * (pt / (1 - pt))
    return nb


def aucd(thr, nb):
    return float(trapezoid(nb, thr))


def main():
    for f in font_manager.findSystemFonts():
        if "arial" in f.lower():
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = "Arial"

    models = load_models()
    y_ref = models[0][1]
    fig, (ax_cal, ax_dca) = plt.subplots(1, 2, figsize=(14.4, 6.6))

    ax_cal.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1.4, label="Ideal", zorder=1)
    for label, y, p, color, ls in models:
        mp, obs = calib_bins(y, p)
        ax_cal.plot(mp, obs, color=color, linestyle=ls, lw=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=f"{label} (Brier = {brier(y, p):.3f})", zorder=3)
    ax_cal.set_xlim(-0.02, 1.02); ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.set_xticks(np.arange(0, 1.01, 0.2)); ax_cal.set_yticks(np.arange(0, 1.01, 0.2))
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=14)
    ax_cal.set_ylabel("Observed Proportion (FD)", fontsize=14)
    ax_cal.set_title("A  Calibration Curve", fontsize=15, loc="left")
    ax_cal.tick_params(axis="both", labelsize=13)
    ax_cal.legend(loc="lower right", fontsize=8.5, framealpha=0.9, handlelength=2.2)

    prev = float(np.mean(y_ref))
    nb_all = prev - (1 - prev) * (THRESHOLDS / (1 - THRESHOLDS))
    ax_dca.plot(THRESHOLDS, nb_all, color="black", lw=2, label=f"Treat All (AUCD = {aucd(THRESHOLDS, nb_all):.3f})")
    ax_dca.plot(THRESHOLDS, np.zeros_like(THRESHOLDS), color="grey", linestyle=":", lw=1.4, label="Treat None (AUCD = 0.000)")
    for label, y, p, color, ls in models:
        nb = net_benefit(y, p)
        ax_dca.plot(THRESHOLDS, nb, color=color, linestyle=ls, lw=2, label=f"{label} (AUCD = {aucd(THRESHOLDS, nb):.3f})")
    ax_dca.set_xlim(0.05, 0.85)
    ax_dca.set_ylim(min(-0.05, float(np.min(nb_all))), max(prev + 0.05, 0.55))
    ax_dca.set_xticks(np.arange(0.05, 0.86, 0.1))
    ax_dca.set_xlabel("Threshold Probability", fontsize=14)
    ax_dca.set_ylabel("Net Benefit", fontsize=14)
    ax_dca.set_title("B  Decision Curve Analysis", fontsize=15, loc="left")
    ax_dca.tick_params(axis="both", labelsize=13)
    ax_dca.legend(loc="lower left", fontsize=8.5, framealpha=0.9, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")
    print("\nBrier scores:")
    for label, y, p, _, _ in models:
        print(f"  {label:14s} Brier={brier(y, p):.4f}")
    print("AUCD:")
    for label, y, p, _, _ in models:
        print(f"  {label:14s} AUCD={aucd(THRESHOLDS, net_benefit(y, p)):.4f}")


if __name__ == "__main__":
    main()
