#!/usr/bin/env python3
"""
delong_comparison.py
====================
Compare two feature-selection methods using DeLong's test.

Reads feature frequencies from two workspace folders, evaluates predictions
at each method's best-K per bin-width (or a single unified K across all bin
widths), and performs DeLong's test (via the MLstatkit package) to compare
AUCs.

The comparison is done per (BW, outer_rs): predictions are pooled across all
8 outer folds (each case appears in exactly one test fold), then a single
DeLong's test is run per (BW, RS) pair.

Usage:
    conda activate gpu-radiomics
    /home/ser/miniconda3/envs/gpu-radiomics/bin/python3 delong_comparison.py
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from MLstatkit import Delong_test

warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────
METHOD1_DIR: Path = Path(
    "/home/ser/pipeline/workspace/OO_r0.7_StandardScaler_top5_NoRdundancyInFitness"
)
METHOD2_DIR: Path = Path(
    "/home/ser/pipeline/workspace/mi_multirs_r0.7_StandardScaler"
)
CLASSIFIER_TYPE: str = "svm"
USE_UNIFIED_K: bool = False  # True: single best-K across all BWs; False: best-K per BW
OUTPUT_DIR: Path = Path(
    "/home/ser/pipeline/delong test/"
    f"{METHOD1_DIR.name} vs {METHOD2_DIR.name}"
)

# ── Load only definitions from cv_pipeline.ipynb (skip execution cells) ─────
_CV_PIPELINE_NB: Path = Path("/home/ser/pipeline/cv_pipeline.ipynb")

import json as _json

_nb_data: dict = _json.loads(_CV_PIPELINE_NB.read_text(encoding="utf-8"))
_code_cells: list[str] = []
for _cell in _nb_data["cells"]:
    if _cell.get("cell_type") != "code":
        continue
    _src: str = "".join(_cell["source"])
    # Stop before the execution / integration-test cells
    if "FULL PIPELINE EXECUTION" in _src or "PART 1 INTEGRATION TEST" in _src:
        break
    _code_cells.append(_src)

_definitions_source: str = "\n\n".join(_code_cells)
_cv_ns: dict = {"__name__": "__cv_pipeline_loaded__", "__file__": str(_CV_PIPELINE_NB)}
exec(compile(_definitions_source, str(_CV_PIPELINE_NB), "exec"), _cv_ns)

# override the notebook's
# current value (which may differ) to match the actual experiment configs.
_cv_ns["SCALER_TYPE_OUTER"] = "StandardScaler"

# ── Extract symbols ─────────────────────────────────────────────────────────
OuterCVGenerator = _cv_ns["OuterCVGenerator"]
get_holdout_classifier = _cv_ns["get_holdout_classifier"]
make_scaler = _cv_ns["make_scaler"]
select_top_k_features = _cv_ns["select_top_k_features"]
SCALER_TYPE_OUTER: str = _cv_ns["SCALER_TYPE_OUTER"]
del _cv_ns, _definitions_source, _nb_data, _code_cells


# ── Helper functions ────────────────────────────────────────────────────────

def find_best_k_per_bw(
    results_df: pd.DataFrame, classifier: str, bin_widths: list[int]
) -> dict[int, int]:
    """For each BW, find the K with highest mean AUC (across RS and folds)."""
    best_k: dict[int, int] = {}
    for bw in bin_widths:
        sub = results_df[
            (results_df["bw"] == bw) & (results_df["classifier"] == classifier)
        ]
        if sub.empty:
            print(f"  [WARNING] No results for bw={bw}, classifier={classifier}")
            continue
        mean_auc = sub.groupby("k")["auc"].mean()
        best_k[bw] = int(mean_auc.idxmax())
        print(f"  bw={bw:>3d}  best_k={best_k[bw]:>3d}  "
              f"mean_auc={mean_auc.max():.4f}")
    return best_k


def find_best_unified_k(
    results_df: pd.DataFrame, classifier: str
) -> int:
    """Find the single K with highest mean AUC across all BWs, RS, and folds."""
    sub = results_df[results_df["classifier"] == classifier]
    if sub.empty:
        print(f"  [WARNING] No results for classifier={classifier}")
        return 5  # fallback
    mean_auc = sub.groupby("k")["auc"].mean()
    best_k = int(mean_auc.idxmax())
    print(f"  unified best_k={best_k}  mean_auc={mean_auc.max():.4f}")
    return best_k


def get_predictions(
    bw: int,
    outer_rs: int,
    fold_idx: int,
    freq_dir: Path,
    k: int,
    classifier: str,
) -> Optional[dict]:
    """
    Train a classifier on the top-K features selected by a method and return
    prediction probabilities on the holdout test set.

    Returns dict with keys: case_ids, y_true, probas — or None on failure.
    """
    # Load data splits (same for both methods since they share the data dir)
    splits = OuterCVGenerator.load_splits_in_memory(bw, outer_rs)
    if fold_idx not in splits:
        print(f"  [WARNING] fold {fold_idx} not in splits for bw={bw}, rs={outer_rs}")
        return None
    X_train, y_train, X_test, y_test, test_case_ids, _ = splits[fold_idx]

    # Load feature frequencies
    freq_path = (
        freq_dir / f"BW_{bw}" / f"outer_rs_{outer_rs}" / f"fold_{fold_idx}"
        / "feature_frequencies.csv"
    )
    if not freq_path.exists():
        print(f"  [WARNING] Missing artifact: {freq_path}")
        return None
    freq_df = pd.read_csv(freq_path)
    if freq_df.empty:
        print(f"  [WARNING] Empty feature frequencies: {freq_path}")
        return None

    # Select top-K features
    selected_features = select_top_k_features(freq_df, k)
    available = [
        f for f in selected_features if f in X_train.columns and f in X_test.columns
    ]
    if len(available) < 2:
        print(f"  [WARNING] < 2 overlapping features for k={k}, bw={bw}, "
              f"rs={outer_rs}, fold={fold_idx}")
        return None

    # Scale and train
    scaler = make_scaler(SCALER_TYPE_OUTER)
    X_tr_s = scaler.fit_transform(X_train[available])
    X_te_s = scaler.transform(X_test[available])
    clf = get_holdout_classifier(classifier)
    clf.fit(X_tr_s, y_train)
    probas = clf.predict_proba(X_te_s)[:, 1]

    return {
        "case_ids": test_case_ids.values,
        "y_true": y_test.values,
        "probas": probas,
    }


def run_delong(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray):
    """
    Run DeLong's test via MLstatkit.

    Returns (z, p, auc_a, auc_b) or (nan, nan, nan, nan) on failure.
    z is signed as z = (AUC_B - AUC_A) / SE.
    """
    try:
        result = Delong_test(
            y_true, prob_a, prob_b,
            return_ci=False, return_auc=True, verbose=0,
        )
        z, p, auc_a, auc_b = result
        return float(z), float(p), float(auc_a), float(auc_b)
    except Exception as exc:
        print(f"  [WARNING] DeLong test failed: {exc}")
        return float("nan"), float("nan"), float("nan"), float("nan")


# ── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  DeLong's Test Comparison")
    print(f"  Method 1: {METHOD1_DIR.name}")
    print(f"  Method 2: {METHOD2_DIR.name}")
    print(f"  Classifier: {CLASSIFIER_TYPE}")
    print(f"  Scaler: {SCALER_TYPE_OUTER}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Read pipeline results ──────────────────────────────────────────────
    results1 = pd.read_csv(METHOD1_DIR / "pipeline_results.csv")
    results2 = pd.read_csv(METHOD2_DIR / "pipeline_results.csv")

    bin_widths = sorted(results1["bw"].unique())
    outer_rs_list = sorted(results1["outer_rs"].unique())
    n_folds = sorted(results1["fold_idx"].unique())

    print(f"\nBWs: {bin_widths}")
    print(f"RS:  {outer_rs_list}")
    print(f"Folds: {n_folds}")

    # ── Find best K per BW (or unified K) for each method ────────────────────
    if USE_UNIFIED_K:
        print("\n── Unified Best K (Method 1) ──")
        k1_unified = find_best_unified_k(results1, CLASSIFIER_TYPE)
        print("\n── Unified Best K (Method 2) ──")
        k2_unified = find_best_unified_k(results2, CLASSIFIER_TYPE)
        best_k1 = {bw: k1_unified for bw in bin_widths}
        best_k2 = {bw: k2_unified for bw in bin_widths}
        k_mode = "unified"
        k_label = f"Unified K (k1={k1_unified}, k2={k2_unified})"
    else:
        print("\n── Best K per BW (Method 1) ──")
        best_k1 = find_best_k_per_bw(results1, CLASSIFIER_TYPE, bin_widths)
        print("\n── Best K per BW (Method 2) ──")
        best_k2 = find_best_k_per_bw(results2, CLASSIFIER_TYPE, bin_widths)
        k_mode = "per_bw"
        k_label = "Best K per BW"

    # Save best K table
    best_k_df = pd.DataFrame({
        "bw": bin_widths,
        f"{METHOD1_DIR.name}_k": [best_k1.get(bw) for bw in bin_widths],
        f"{METHOD2_DIR.name}_k": [best_k2.get(bw) for bw in bin_widths],
    })
    best_k_df.to_csv(OUTPUT_DIR / "best_k_per_bw.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'best_k_per_bw.csv'}")
    print(f"K selection mode: {k_label}")

    # ── Per-case predictions + DeLong results ──────────────────────────────
    per_case_rows: list[dict] = []
    delong_rows: list[dict] = []

    freq_dir1 = METHOD1_DIR / "artifacts"
    freq_dir2 = METHOD2_DIR / "artifacts"

    for bw in bin_widths:
        k1 = best_k1.get(bw)
        k2 = best_k2.get(bw)
        if k1 is None or k2 is None:
            print(f"\n[SKIP] bw={bw} — missing best K for one or both methods")
            continue

        print(f"\n── BW={bw}  (k1={k1}, k2={k2}) ──")

        for outer_rs in outer_rs_list:
            # Pool predictions across folds for this (BW, RS)
            all_y_true: list = []
            all_prob1: list = []
            all_prob2: list = []
            all_case_ids: list = []

            for fold_idx in n_folds:
                pred1 = get_predictions(
                    bw, outer_rs, fold_idx, freq_dir1, k1, CLASSIFIER_TYPE
                )
                pred2 = get_predictions(
                    bw, outer_rs, fold_idx, freq_dir2, k2, CLASSIFIER_TYPE
                )

                if pred1 is None or pred2 is None:
                    continue

                # Verify y_test alignment between methods
                if not np.array_equal(pred1["y_true"], pred2["y_true"]):
                    print(f"  [WARNING] y_test mismatch at bw={bw}, rs={outer_rs}, "
                          f"fold={fold_idx}")
                    continue

                # Save per-case predictions
                for i in range(len(pred1["case_ids"])):
                    per_case_rows.append({
                        "bw": bw,
                        "outer_rs": outer_rs,
                        "fold_idx": fold_idx,
                        "case_id": pred1["case_ids"][i],
                        "true_label": int(pred1["y_true"][i]),
                        f"{METHOD1_DIR.name}_prob": float(pred1["probas"][i]),
                        f"{METHOD2_DIR.name}_prob": float(pred2["probas"][i]),
                    })

                # Pool for DeLong test
                all_y_true.extend(pred1["y_true"])
                all_prob1.extend(pred1["probas"])
                all_prob2.extend(pred2["probas"])
                all_case_ids.extend(pred1["case_ids"])

            if len(all_y_true) < 3:
                print(f"  rs={outer_rs}: insufficient data ({len(all_y_true)} cases)")
                continue

            y_true = np.array(all_y_true)
            prob1 = np.array(all_prob1)
            prob2 = np.array(all_prob2)

            # Run DeLong's test
            z, p, auc1, auc2 = run_delong(y_true, prob1, prob2)

            # method1 is significantly better if p < 0.05 AND auc1 > auc2
            # (z = (AUC_B - AUC_A) / SE, so z < 0 means AUC_A > AUC_B)
            sig_better_m1 = bool(
                not np.isnan(p) and p < 0.05 and auc1 > auc2
            )
            # method2 is significantly better if p < 0.05 AND auc2 > auc1
            sig_better_m2 = bool(
                not np.isnan(p) and p < 0.05 and auc2 > auc1
            )

            delong_rows.append({
                "bw": bw,
                "outer_rs": outer_rs,
                f"{METHOD1_DIR.name}_auc": auc1,
                f"{METHOD2_DIR.name}_auc": auc2,
                "z_stat": z,
                "p_value": p,
                f"{METHOD1_DIR.name}_better": sig_better_m1,
                f"{METHOD2_DIR.name}_better": sig_better_m2,
                "n_cases": len(y_true),
            })

            print(f"  rs={outer_rs:>4d}: AUC1={auc1:.4f}  AUC2={auc2:.4f}  "
                  f"z={z:.4f}  p={p:.6f}  "
                  f"M1 sig={'YES' if sig_better_m1 else 'no'}  "
                  f"M2 sig={'YES' if sig_better_m2 else 'no'}  "
                  f"(n={len(y_true)})")

    # ── Save results ───────────────────────────────────────────────────────
    per_case_df = pd.DataFrame(per_case_rows)
    per_case_df.to_csv(OUTPUT_DIR / "per_case_predictions.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'per_case_predictions.csv'} "
          f"({len(per_case_df)} rows)")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(OUTPUT_DIR / "delong_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'delong_results.csv'} "
          f"({len(delong_df)} rows)")

    # ── Summary report ─────────────────────────────────────────────────────
    write_summary_report(
        OUTPUT_DIR, delong_df, best_k_df, bin_widths, outer_rs_list,
        METHOD1_DIR.name, METHOD2_DIR.name, CLASSIFIER_TYPE, SCALER_TYPE_OUTER,
        k_mode,
    )
    print(f"Saved: {OUTPUT_DIR / 'summary_report.md'}")
    print(f"\n{'=' * 70}\nDone.\n{'=' * 70}")


def write_summary_report(
    output_dir: Path,
    delong_df: pd.DataFrame,
    best_k_df: pd.DataFrame,
    bin_widths: list[int],
    outer_rs_list: list[int],
    name1: str,
    name2: str,
    classifier: str,
    scaler_type: str,
    k_mode: str = "per_bw",
) -> None:
    """Write a Markdown summary report."""
    m1_auc_col = f"{name1}_auc"
    m2_auc_col = f"{name2}_auc"
    m1_better_col = f"{name1}_better"
    m2_better_col = f"{name2}_better"

    lines: list[str] = []
    lines.append("# DeLong's Test Summary Report\n")
    lines.append(f"**Method 1:** `{name1}`\n")
    lines.append(f"**Method 2:** `{name2}`\n")
    lines.append(f"**Classifier:** `{classifier}`\n")
    lines.append(f"**Scaler:** `{scaler_type}`\n")
    lines.append(f"**K selection mode:** `{k_mode}`\n")
    lines.append(f"**Test:** DeLong's test (via MLstatkit package)\n")
    lines.append(f"**Significance level:** α = 0.05 (two-sided)\n")
    lines.append(
        f"**Direction:** Method 1 is 'significantly better' when "
        f"p < 0.05 **and** AUC₁ > AUC₂; Method 2 is 'significantly better' "
        f"when p < 0.05 **and** AUC₂ > AUC₁\n"
    )

    # ── Best K per BW ──────────────────────────────────────────────────────
    k_section_title = "Best K per Bin Width" if k_mode == "per_bw" else "Unified Best K"
    lines.append(f"\n## {k_section_title}\n")
    lines.append("| BW | Method 1 K | Method 2 K |")
    lines.append("|---:|---:|---:|")
    for _, row in best_k_df.iterrows():
        lines.append(
            f"| {int(row['bw'])} | {int(row[f'{name1}_k'])} | "
            f"{int(row[f'{name2}_k'])} |"
        )

    # ── Per (BW, RS) results ───────────────────────────────────────────────
    lines.append("\n## DeLong's Test Results per (BW, RS)\n")
    lines.append(
        "| BW | RS | AUC₁ | AUC₂ | z-stat | p-value | "
        f"Method 1 better? | Method 2 better? | n_cases |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|")
    for _, row in delong_df.iterrows():
        sig1 = "✅ YES" if row[m1_better_col] else "❌ no"
        sig2 = "✅ YES" if row[m2_better_col] else "❌ no"
        lines.append(
            f"| {int(row['bw'])} | {int(row['outer_rs'])} | "
            f"{row[m1_auc_col]:.4f} | {row[m2_auc_col]:.4f} | "
            f"{row['z_stat']:.4f} | {row['p_value']:.6f} | "
            f"{sig1} | {sig2} | {int(row['n_cases'])} |"
        )

    # ── Per BW summary (averaged across RS) ────────────────────────────────
    lines.append("\n## Per-BW Summary (averaged across RS)\n")
    lines.append(
        "| BW | Mean AUC₁ | Mean AUC₂ | Mean p-value | "
        f"# RS sig. (M1>M2) | RS sig. (M1<M2) | # RS total |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for bw in bin_widths:
        sub = delong_df[delong_df["bw"] == bw]
        if sub.empty:
            continue
        mean_auc1 = sub[m1_auc_col].mean()
        mean_auc2 = sub[m2_auc_col].mean()
        mean_p = sub["p_value"].mean()
        n_sig_m1 = int(sub[m1_better_col].sum())
        n_sig_m2 = int(sub[m2_better_col].sum())
        n_total = len(sub)
        lines.append(
            f"| {bw} | {mean_auc1:.4f} | {mean_auc2:.4f} | "
            f"{mean_p:.6f} | {n_sig_m1} | {n_sig_m2} | {n_total} |"
        )

    # ── Per RS summary (averaged across BW) ────────────────────────────────
    lines.append("\n## Per-RS Summary (averaged across BW)\n")
    lines.append(
        "| RS | Mean AUC₁ | Mean AUC₂ | Mean p-value | "
        f"# BW sig. (M1>M2) | # BW sig. (M1<M2) | # BW total |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for rs in outer_rs_list:
        sub = delong_df[delong_df["outer_rs"] == rs]
        if sub.empty:
            continue
        mean_auc1 = sub[m1_auc_col].mean()
        mean_auc2 = sub[m2_auc_col].mean()
        mean_p = sub["p_value"].mean()
        n_sig_m1 = int(sub[m1_better_col].sum())
        n_sig_m2 = int(sub[m2_better_col].sum())
        n_total = len(sub)
        lines.append(
            f"| {rs} | {mean_auc1:.4f} | {mean_auc2:.4f} | "
            f"{mean_p:.6f} | {n_sig_m1} | {n_sig_m2} | {n_total} |"
        )

    # ── Overall summary ──────────────────────────────────────────────────────
    lines.append("\n## Overall Summary\n")
    overall_auc1 = delong_df[m1_auc_col].mean()
    overall_auc2 = delong_df[m2_auc_col].mean()
    overall_p = delong_df["p_value"].mean()
    overall_sig_m1 = int(delong_df[m1_better_col].sum())
    overall_sig_m2 = int(delong_df[m2_better_col].sum())
    overall_total = len(delong_df)
    lines.append(f"- **Overall mean AUC₁ ({name1}):** {overall_auc1:.4f}")
    lines.append(f"- **Overall mean AUC₂ ({name2}):** {overall_auc2:.4f}")
    lines.append(f"- **Overall mean p-value:** {overall_p:.6f}")
    lines.append(
        f"- **Method 1 significantly better in "
        f"{overall_sig_m1}/{overall_total} (BW, RS) comparisons**"
    )
    lines.append(
        f"- **Method 2 significantly better in "
        f"{overall_sig_m2}/{overall_total} (BW, RS) comparisons**"
    )

    if overall_auc1 > overall_auc2:
        lines.append(
            f"\n**Conclusion:** Method 1 ({name1}) has a higher mean AUC "
            f"({overall_auc1:.4f} vs {overall_auc2:.4f}), and is significantly "
            f"better in {overall_sig_m1}/{overall_total} comparisons. "
            f"Method 2 is significantly better in "
            f"{overall_sig_m2}/{overall_total} comparisons."
        )
    else:
        lines.append(
            f"\n**Conclusion:** Method 2 ({name2}) has a higher mean AUC "
            f"({overall_auc2:.4f} vs {overall_auc1:.4f}), and is significantly "
            f"better in {overall_sig_m2}/{overall_total} comparisons. "
            f"Method 1 is significantly better in "
            f"{overall_sig_m1}/{overall_total} comparisons."
        )

    lines.append("\n---")
    lines.append(
        f"\n*Generated by `delong_comparison.py` using MLstatkit's "
        f"`Delong_test` function.*\n"
    )

    (output_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
