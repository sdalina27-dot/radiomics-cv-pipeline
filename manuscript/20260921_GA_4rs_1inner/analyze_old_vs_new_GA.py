#!/usr/bin/env python3
"""
analyze_old_vs_new_GA.py
========================
Old vs new GA LOO comparison (LOO, SVM, k=4, 80 cases/BW).

  OLD : loo_r0.7_{Scaler}_noMIinit          5 GA folds x 8 random seeds, inner = 3-fold AUC average
  NEW : loo_r0.7_{Scaler}_maxk4_noMI_inner1 5 GA folds x 4 random seeds, inner = single holdout (1 AUC)

Both runs use USE_MI_INITIALIZATION=False (pure random GA init), so the only
differences are the GA random-seed count and the inner fitness split. Everything
is SVM @ k=4 on pooled LOO per-case scores.

Outputs (this folder):
  auc_comparison_{S}.md/.csv        per-BW pooled AUC (95% bootstrap CI) + DeLong old vs new
  delong_results_{S}.csv            raw per-BW DeLong z/p + both AUCs
  auc_matrix.csv                    combined 10 BW x 6 (old/new x SS/RS/PT) pooled AUC
  feature_comparison_{S}_BW10.md/.csv  top-4 selected features at BW10 (sel. prob. + sel. freq.)
  feature_overlap_{S}.csv           Spearman rho + top-4 Jaccard per BW
  README.md                         summary
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from MLstatkit import Delong_test

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/ser/pipeline/workspace")
DATA_DIR = Path("/home/ser/pipeline/data")
OUT = Path(__file__).resolve().parent

SCALERS = ["RobustScaler", "StandardScaler", "PowerTransformer"]
SHORT = {"RobustScaler": "RS", "StandardScaler": "SS", "PowerTransformer": "PT"}
BWS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CLASSIFIER = "svm"
K = 4
PRED_FILE = "pipeline_predictions.csv"

OLD_TPL = "loo_r0.7_{s}_noMIinit"
NEW_TPL = "loo_r0.7_{s}_maxk4_noMI_inner1"
OLD_LABEL = "old (5x8, 3 inner)"
NEW_LABEL = "new (5x4, 1 inner)"

# GA selection-vote denominators: folds x n_random_seeds x n_GA_folds x TOP_N_FREQUENCY
OLD_N_RS, NEW_N_RS = 8, 4
N_GA_FOLDS = 5
TOP_N = 5
FEATURE_BW = 10

FEATURE_TYPES = {"shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"}
NAME_PATTERN = re.compile(r"Cine_3D_(ED|ES)_(.+)")


def parse_feature_name(name):
    m = NAME_PATTERN.match(name)
    if not m:
        return "?", "?", name
    toks = m.group(2).split("_")
    return m.group(1), toks[-2] if toks[-2] in FEATURE_TYPES else "?", toks[-1]


def compute_ci(exp, n_iter=1000, seed=42):
    """Pooled AUC + percentile bootstrap CI (matches bootstrap_ci_svm_k4)."""
    rng = np.random.default_rng(seed)
    df = pd.read_csv(WORKSPACE / exp / PRED_FILE)
    df = df[(df["classifier"] == CLASSIFIER) & (df["k"] == K)]
    recs = []
    for bw in sorted(df["bw"].unique()):
        sub = df[df["bw"] == bw]
        y_true, y_score = sub["y_true"].values, sub["y_score"].values
        n = len(sub)
        try:
            pooled = roc_auc_score(y_true, y_score)
        except ValueError:
            pooled = np.nan
        boot = []
        for _ in range(n_iter):
            idx = rng.integers(0, n, size=n)
            try:
                boot.append(roc_auc_score(y_true[idx], y_score[idx]))
            except ValueError:
                boot.append(np.nan)
        boot = np.array(boot)
        boot = boot[~np.isnan(boot)]
        lo = float(np.percentile(boot, 2.5)) if len(boot) else np.nan
        hi = float(np.percentile(boot, 97.5)) if len(boot) else np.nan
        recs.append({"bw": bw, "auc_pooled": pooled, "auc_lo": lo, "auc_hi": hi})
    return pd.DataFrame(recs).set_index("bw")


def load_scores(exp, bw):
    df = pd.read_csv(WORKSPACE / exp / PRED_FILE)
    df = df[(df["classifier"] == CLASSIFIER) & (df["k"] == K) & (df["bw"] == bw)]
    df = df.sort_values("case_id")
    return df["y_true"].values, df["y_score"].values


def delong(y_true, p1, p2):
    try:
        z, p, a1, a2 = Delong_test(y_true, p1, p2, return_ci=False, return_auc=True, verbose=0)
        return float(z), float(p), float(a1), float(a2)
    except Exception as exc:
        print(f"  [WARNING] DeLong failed: {exc}")
        return float("nan"), float("nan"), float("nan"), float("nan")


def star_for_p(p):
    if pd.isna(p):
        return ""
    for thr, st in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p < thr:
            return st
    return ""


# ── 1. Per-scaler AUC comparison + DeLong ───────────────────────────────────
def analyze_scaler(scaler):
    s = SHORT[scaler]
    m_old = OLD_TPL.format(s=scaler)
    m_new = NEW_TPL.format(s=scaler)
    print(f"\n{'='*70}\n[{s}] {m_old}  vs  {m_new}\n{'='*70}")

    ci_old = compute_ci(m_old)
    ci_new = compute_ci(m_new)
    rows = []
    for bw in BWS:
        y_true, s_old = load_scores(m_old, bw)
        _, s_new = load_scores(m_new, bw)
        z, p, a_old, a_new = delong(y_true, s_old, s_new)
        rows.append({
            "bw": bw,
            "auc_old": float(ci_old.loc[bw, "auc_pooled"]),
            "ci_lo_old": float(ci_old.loc[bw, "auc_lo"]),
            "ci_hi_old": float(ci_old.loc[bw, "auc_hi"]),
            "auc_new": float(ci_new.loc[bw, "auc_pooled"]),
            "ci_lo_new": float(ci_new.loc[bw, "auc_lo"]),
            "ci_hi_new": float(ci_new.loc[bw, "auc_hi"]),
            "delta_auc": float(ci_new.loc[bw, "auc_pooled"]) - float(ci_old.loc[bw, "auc_pooled"]),
            "delong_z": z, "delong_p": p,
        })
    ac = pd.DataFrame(rows)
    ac.to_csv(OUT / f"auc_comparison_{s}.csv", index=False)
    ac[["bw", "auc_old", "auc_new", "delta_auc", "delong_z", "delong_p"]].to_csv(
        OUT / f"delong_results_{s}.csv", index=False)

    lines = [f"# GA {scaler} — old vs new LOO config (SVM, k=4)", "",
             f"OLD = `{m_old}` ({OLD_LABEL}); NEW = `{m_new}` ({NEW_LABEL}).",
             "Pooled LOO AUC with 95% bootstrap CI (1000 iters). ΔAUC = new − old.",
             "DeLong's test (two-sided, paired 80 cases). Star marks the significantly better run "
             "(`*` p<0.05, `**` p<0.01, `***` p<0.001).",
             "", "| BW | old AUC (95% CI) | new AUC (95% CI) | ΔAUC | DeLong p |",
             "|---:|---:|---:|---:|---:|"]
    for _, r in ac.iterrows():
        b1 = f"{r['auc_old']:.3f} ({r['ci_lo_old']:.3f}, {r['ci_hi_old']:.3f})"
        b2 = f"{r['auc_new']:.3f} ({r['ci_lo_new']:.3f}, {r['ci_hi_new']:.3f})"
        st = star_for_p(r["delong_p"])
        if r["delta_auc"] > 0 and st:
            b2 += st
        elif r["delta_auc"] < 0 and st:
            b1 += st
        lines.append(f"| {int(r['bw'])} | {b1} | {b2} | {r['delta_auc']:+.3f} | {r['delong_p']:.4g} |")
    lines += ["", f"*Generated by `{Path(__file__).name}`.*"]
    (OUT / f"auc_comparison_{s}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: auc_comparison_{s}.md/.csv, delong_results_{s}.csv")
    return ac


# ── 2. Combined AUC matrix (for the heatmap) ────────────────────────────────
def build_auc_matrix(per_scaler):
    cols = []
    for s in SCALERS:
        cols += [f"old {SHORT[s]}", f"new {SHORT[s]}"]
    mat = pd.DataFrame(index=BWS, columns=cols, dtype=float)
    for s in SCALERS:
        ac = per_scaler[s].set_index("bw")
        mat[f"old {SHORT[s]}"] = ac["auc_old"]
        mat[f"new {SHORT[s]}"] = ac["auc_new"]
    mat.index.name = "bw"
    mat.to_csv(OUT / "auc_matrix.csv")
    print("Saved: auc_matrix.csv")


# ── 3. Selected-feature comparison (BW10) ───────────────────────────────────
def collect_feature_stats(exp, bw):
    """Return (sel_count_top4, raw_count_sum, n_folds) across the 80 LOO folds."""
    fold_dir = WORKSPACE / exp / "artifacts" / f"BW_{bw}" / "outer_rs_loo"
    fold_dirs = sorted(fold_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    sel_count, raw_sum = {}, {}
    for fd in fold_dirs:
        freq = pd.read_csv(fd / "feature_frequencies.csv")
        if freq.empty:
            continue
        for f in freq.sort_values("rank").head(4)["feature"]:
            sel_count[f] = sel_count.get(f, 0) + 1
        for _, r in freq.iterrows():
            raw_sum[r["feature"]] = raw_sum.get(r["feature"], 0) + int(r["raw_count"])
    return sel_count, raw_sum, len(fold_dirs)


def feature_comparison(scaler, bw=FEATURE_BW):
    s = SHORT[scaler]
    m_old = OLD_TPL.format(s=scaler)
    m_new = NEW_TPL.format(s=scaler)
    old_cnt, old_raw, n_old = collect_feature_stats(m_old, bw)
    new_cnt, new_raw, n_new = collect_feature_stats(m_new, bw)

    denom_old = n_old * OLD_N_RS * N_GA_FOLDS * TOP_N
    denom_new = n_new * NEW_N_RS * N_GA_FOLDS * TOP_N

    data = pd.read_csv(DATA_DIR / f"Cine_output_20260606_binWidth_{bw}.csv")
    features = sorted(set(old_cnt) | set(new_cnt))
    rows = []
    for feat in features:
        if feat not in data.columns:
            continue
        phase, ftype, fname = parse_feature_name(feat)
        g1, g0 = data.loc[data["Label"] == 1, feat], data.loc[data["Label"] == 0, feat]
        try:
            p = float(mannwhitneyu(g1, g0, alternative="two-sided").pvalue)
        except Exception:
            p = float("nan")
        rows.append({
            "phase": phase, "type": ftype, "feature": fname, "full_name": feat,
            "fd_mean": float(g1.mean()), "fd_std": float(g1.std(ddof=1)),
            "hcm_mean": float(g0.mean()), "hcm_std": float(g0.std(ddof=1)),
            "p_value": p,
            "old_sel_prob": old_cnt.get(feat, 0) / n_old,
            "new_sel_prob": new_cnt.get(feat, 0) / n_new,
            "old_sel_freq": old_raw.get(feat, 0) / denom_old,
            "new_sel_freq": new_raw.get(feat, 0) / denom_new,
            "old_sel_folds": old_cnt.get(feat, 0),
            "new_sel_folds": new_cnt.get(feat, 0),
        })
    out = pd.DataFrame(rows)
    out["delta_sel_prob"] = out["new_sel_prob"] - out["old_sel_prob"]
    out = out.sort_values(["old_sel_prob", "new_sel_prob"], ascending=[False, False]).reset_index(drop=True)
    out.to_csv(OUT / f"feature_comparison_{s}_BW{bw}.csv", index=False)

    lines = [f"# Selected features — GA {scaler}, BW{bw}: old vs new (SVM, k=4, LOO)", "",
             f"OLD = `{m_old}` ({OLD_LABEL}); NEW = `{m_new}` ({NEW_LABEL}).",
             "",
             "- **sel. prob.** = fraction of the 80 LOO folds where the feature ranks in that fold's top-4 "
             "(comparable across runs).",
             f"- **sel. freq.** = total `raw_count` / (80 folds x n_RS x {N_GA_FOLDS} GA folds x {TOP_N} top chromosomes) "
             f"= selection-vote frequency (OLD denom {denom_old}, NEW denom {denom_new}).",
             "- Features listed appear in the top-4 of at least one run. FD = 40 label-1, HCM = 40 label-0; "
             "p = Mann-Whitney U (two-sided).",
             "",
             "| phase | type | feature | FD | HCM | p | old sel. prob. | new sel. prob. | old sel. freq. | new sel. freq. |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in out.iterrows():
        lines.append(
            f"| {r['phase']} | {r['type']} | {r['feature']} | "
            f"{r['fd_mean']:.3f} ± {r['fd_std']:.3f} | {r['hcm_mean']:.3f} ± {r['hcm_std']:.3f} | "
            f"{r['p_value']:.6g} | {r['old_sel_prob']:.3f} | {r['new_sel_prob']:.3f} | "
            f"{r['old_sel_freq']:.3f} | {r['new_sel_freq']:.3f} |")
    lines += ["", f"*Generated by `{Path(__file__).name}`.*"]
    (OUT / f"feature_comparison_{s}_BW{bw}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: feature_comparison_{s}_BW{bw}.md/.csv ({len(out)} features)")
    return old_cnt, new_cnt, old_raw, new_raw, n_old, n_new


# ── 4. Feature-ranking overlap per BW ───────────────────────────────────────
def agg_raw_count(exp, bw):
    fold_dir = WORKSPACE / exp / "artifacts" / f"BW_{bw}" / "outer_rs_loo"
    fold_dirs = sorted(fold_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    agg = {}
    for fd in fold_dirs:
        freq = pd.read_csv(fd / "feature_frequencies.csv")
        if freq.empty:
            continue
        for _, r in freq.iterrows():
            agg[r["feature"]] = agg.get(r["feature"], 0) + int(r["raw_count"])
    return pd.Series(agg).sort_values(ascending=False)


def build_overlap(scaler):
    s = SHORT[scaler]
    m_old = OLD_TPL.format(s=scaler)
    m_new = NEW_TPL.format(s=scaler)
    rows = []
    for bw in BWS:
        c1, _, _ = collect_feature_stats(m_old, bw)
        c2, _, _ = collect_feature_stats(m_new, bw)
        t1, t2 = set(c1), set(c2)
        inter, union = t1 & t2, t1 | t2
        jaccard = len(inter) / len(union) if union else float("nan")
        s1, s2 = agg_raw_count(m_old, bw), agg_raw_count(m_new, bw)
        common = s1.index.intersection(s2.index)
        rho = float(spearmanr(s1.loc[common], s2.loc[common]).statistic) if len(common) >= 2 else float("nan")
        rows.append({
            "bw": bw, "spearman_rawcount": rho, "n_common_features": len(common),
            "top4_jaccard": jaccard,
            "top4_old": ", ".join(sorted(t1)),
            "top4_new": ", ".join(sorted(t2)),
            "top4_overlap": ", ".join(sorted(inter)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"feature_overlap_{s}.csv", index=False)
    print(f"Saved: feature_overlap_{s}.csv")
    return out


# ── README ──────────────────────────────────────────────────────────────────
def write_readme(per_scaler, overlaps):
    lines = ["# GA old vs new LOO config — all scalers (LOO, SVM, k=4)", "",
             "**OLD:** `loo_r0.7_{Scaler}_noMIinit` — 5 GA folds x 8 random seeds, "
             "inner fitness = mean AUC over a 3-fold inner CV.",
             "**NEW:** `loo_r0.7_{Scaler}_maxk4_noMI_inner1` — 5 GA folds x 4 random seeds, "
             "inner fitness = single stratified holdout (1 AUC per individual per generation).",
             "Both use `USE_MI_INITIALIZATION=False`, so the only changes are the GA random-seed "
             "count (8 -> 4) and the inner split (3-fold average -> 1 holdout).", "",
             "## Headline", ""]
    for s in SCALERS:
        ac = per_scaler[s]
        n_sig = int((ac["delong_p"] < 0.05).sum())
        bw10 = ac[ac["bw"] == 10].iloc[0]
        ov = overlaps[s]
        lines.append(
            f"- **{SHORT[s]}**: mean ΔAUC (new − old) {ac['delta_auc'].mean():+.3f}; "
            f"DeLong significant at α=0.05 in {n_sig}/10 BWs; "
            f"BW10 old {bw10['auc_old']:.3f} -> new {bw10['auc_new']:.3f} "
            f"(Δ{bw10['delta_auc']:+.3f}, p={bw10['delong_p']:.3g}); "
            f"Spearman ρ (feature ranking) {ov['spearman_rawcount'].min():.3f}–{ov['spearman_rawcount'].max():.3f}.")
    lines += ["", "## Files", "",
              "- `auc_comparison_{SS,RS,PT}.md/.csv` — per-BW pooled AUC (95% CI) + DeLong old vs new.",
              "- `delong_results_{SS,RS,PT}.csv` — raw per-BW z/p + both AUCs.",
              "- `auc_matrix.csv` — combined 10 BW x 6 matrix behind the heatmap.",
              "- `feature_comparison_{SS,RS,PT}_BW10.md/.csv` — top-4 selected features with sel. prob. and sel. freq.",
              "- `feature_overlap_{SS,RS,PT}.csv` — Spearman ρ + top-4 Jaccard per BW.",
              "- `fig_ga_old_vs_new_loomap.png` — 6-column heatmap (old/new x SS/RS/PT), stars = DeLong old vs new.",
              "",
              f"*Generated by `analyze_old_vs_new_GA.py` on {pd.Timestamp.now():%Y-%m-%d %H:%M}.*"]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved: README.md")


def main():
    print("=" * 70)
    print("GA old (5x8, 3 inner) vs new (5x4, 1 inner) — all scalers (LOO, SVM, k=4)")
    print("=" * 70)
    per_scaler = {s: analyze_scaler(s) for s in SCALERS}
    build_auc_matrix(per_scaler)
    for s in SCALERS:
        feature_comparison(s, FEATURE_BW)
    overlaps = {s: build_overlap(s) for s in SCALERS}
    write_readme(per_scaler, overlaps)
    print("\nDone.")


if __name__ == "__main__":
    main()
