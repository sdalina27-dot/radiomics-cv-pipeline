#!/usr/bin/env python3
"""
ga_mi_config.py
===============
Shared configuration + helpers for the 20260923 manuscript regeneration.

CURRENT (new) runs:
  GA  loo_r0.7_{Scaler}_maxk4_noMI_inner1   5 GA folds x 4 random seeds, 1 inner holdout
  MI  mi_loo_r0.7_{Scaler}_inner1           5 folds x 4 random seeds

PREVIOUS (old) runs used by the 20260915 manuscript:
  GA  loo_r0.7_{Scaler}_noMIinit            5 GA folds x 8 random seeds, 3 inner folds
  MI  mi_loo_r0.7_{Scaler}                  5 folds x 8 random seeds

All metrics are RBF-SVM @ k=4, pooled over the 80 leave-one-out cases.
"""
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from MLstatkit import Delong_test

warnings.filterwarnings("ignore")

BASE = Path("/home/ser/pipeline")
WS = BASE / "workspace"
DATA = BASE / "data"
CLINICAL_PRED = BASE / "manuscript" / "20260906" / "clinical_demographic_predictions.csv"
CLINICAL_SUMMARY = BASE / "manuscript" / "20260906" / "clinical_demographic_summary.csv"
BASELINE_SUMMARY = WS / "baselines" / "baseline_summary.csv"
BASELINE_FIXED = WS / "baselines" / "fixed_triad_predictions.csv"
BASELINE_MWU = WS / "baselines" / "univariate_mwu_predictions.csv"

SCALERS = ["StandardScaler", "RobustScaler", "PowerTransformer"]
SHORT = {"StandardScaler": "SS", "RobustScaler": "RS", "PowerTransformer": "PT"}
BWS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

GA_NEW = "loo_r0.7_{s}_maxk4_noMI_inner1"
MI_NEW = "mi_loo_r0.7_{s}_inner1"
GA_OLD = "loo_r0.7_{s}_noMIinit"
MI_OLD = "mi_loo_r0.7_{s}"

# Each method at its own peak (new runs): GA RobustScaler BW10, MI StandardScaler BW20.
GA_BW = {"RobustScaler": 10, "StandardScaler": 20, "PowerTransformer": 10}
MI_BW = {"RobustScaler": 10, "StandardScaler": 20, "PowerTransformer": 10}
# Representative configs used in Tables 3/5/6 and Figures 3/4.
CONFIG_GA_RS10 = (GA_NEW.format(s="RobustScaler"), 10)
CONFIG_MI_RS10 = (MI_NEW.format(s="RobustScaler"), 10)
CONFIG_GA_SS20 = (GA_NEW.format(s="StandardScaler"), 20)
CONFIG_MI_SS20 = (MI_NEW.format(s="StandardScaler"), 20)

CLASSIFIER = "svm"
K = 4
PRED = "pipeline_predictions.csv"

FEATURE_TYPES = {"shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"}
NAME_PATTERN = re.compile(r"Cine_3D_(ED|ES)_(.+)")
ELONG = "Cine_3D_ES_original_shape_Elongation"
SPHER = "Cine_3D_ES_original_shape_Sphericity"
FLAT = "Cine_3D_ES_original_shape_Flatness"


def parse_feature_name(name):
    m = NAME_PATTERN.match(name)
    if not m:
        return "?", "?", name
    toks = m.group(2).split("_")
    return m.group(1), toks[-2] if toks[-2] in FEATURE_TYPES else "?", toks[-1]


def short_name(name):
    return name.replace("Cine_3D_", "").replace("_original_", " ").replace("_", " ")


def _pred_df(exp):
    df = pd.read_csv(WS / exp / PRED)
    return df[(df["classifier"] == CLASSIFIER) & (df["k"] == K)]


def load_scores(exp, bw):
    sub = _pred_df(exp)
    sub = sub[sub["bw"] == bw].sort_values("case_id")
    return sub["y_true"].values, sub["y_score"].values


_BOOT = {}


def bootstrap_all(exp, n_iter=1000, seed=42):
    """Per-BW pooled AUC + percentile bootstrap CI (one shared RNG, ascending BW)."""
    if exp in _BOOT:
        return _BOOT[exp]
    rng = np.random.default_rng(seed)
    df = _pred_df(exp)
    out = {}
    for bw in sorted(df["bw"].unique()):
        sub = df[df["bw"] == bw]
        yt, ys = sub["y_true"].values, sub["y_score"].values
        n = len(sub)
        boot = []
        for _ in range(n_iter):
            idx = rng.integers(0, n, size=n)
            try:
                boot.append(roc_auc_score(yt[idx], ys[idx]))
            except ValueError:
                boot.append(np.nan)
        boot = np.array(boot)
        boot = boot[~np.isnan(boot)]
        out[bw] = (float(roc_auc_score(yt, ys)),
                   float(np.percentile(boot, 2.5)),
                   float(np.percentile(boot, 97.5)))
    _BOOT[exp] = out
    return out


def pooled_auc(exp):
    return {bw: v[0] for bw, v in bootstrap_all(exp).items()}


def delong_pair(exp1, exp2, bw):
    yt, s1 = load_scores(exp1, bw)
    _, s2 = load_scores(exp2, bw)
    z, p, a1, a2 = Delong_test(yt, s1, s2, return_ci=False, return_auc=True, verbose=0)
    return float(z), float(p), float(a1), float(a2)


def delong_arrays(y_true, s1, s2):
    z, p, a1, a2 = Delong_test(y_true, s1, s2, return_ci=False, return_auc=True, verbose=0)
    return float(z), float(p), float(a1), float(a2)


def star_for_p(p):
    if pd.isna(p):
        return ""
    for thr, st in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p < thr:
            return st
    return ""


def dagger_for_p(p):
    if pd.isna(p):
        return ""
    for thr, dg in [(0.001, "†††"), (0.01, "††"), (0.05, "†")]:
        if p < thr:
            return dg
    return ""


def fold_dirs(exp, bw):
    d = WS / exp / "artifacts" / f"BW_{bw}" / "outer_rs_loo"
    return sorted(d.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))


def fold_top4(exp, bw, fold):
    f = (WS / exp / "artifacts" / f"BW_{bw}" / "outer_rs_loo" /
         f"fold_{fold}" / "feature_frequencies.csv")
    return pd.read_csv(f).sort_values("rank").head(4)["feature"].tolist()


def collect_feature_stats(exp, bw):
    """(top4 fold counts, summed raw_count, n_folds)."""
    sel, raw = {}, {}
    dirs = fold_dirs(exp, bw)
    for fd in dirs:
        f = fd / "feature_frequencies.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        for feat in df.sort_values("rank").head(4)["feature"]:
            sel[feat] = sel.get(feat, 0) + 1
        for _, r in df.iterrows():
            raw[r["feature"]] = raw.get(r["feature"], 0) + int(r["raw_count"])
    return sel, raw, len(dirs)


def feature_group_stats(bw):
    d = pd.read_csv(DATA / f"Cine_output_20260606_binWidth_{bw}.csv")
    return d
