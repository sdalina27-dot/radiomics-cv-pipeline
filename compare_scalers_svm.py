#!/usr/bin/env python3
"""
compare_scalers_svm.py
======================
Read every feature_frequencies.csv under
  workspace/original_only_r0.7_QuantileTransformer/artifacts/
and re-evaluate top-K SVM (SVC rbf, C=1, probability=True, random_state)
using three different scalers:
  - QuantileTransformer
  - StandardScaler
  - RobustScaler

Output is a long-format CSV identical in schema to pipeline_results.csv:
  bw, outer_rs, fold_idx, classifier, k, auc, timestamp
with classifier labeled
  svm_QuantileTransformer / svm_StandardScaler / svm_RobustScaler
so heatmap_auc.py can read it and produce one heatmap per scaler
(CLASSIFIER = 'svm_<scaler>'), enabling side-by-side comparison of the
same feature-frequency ranking under different scalers.

Usage:
    conda activate gpu-radiomics
    python compare_scalers_svm.py

Reuses definitions from cv_pipeline_mutual_info.py (importing it is safe:
all execution is guarded by `if __name__ == "__main__"`).
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# Reuse shared helpers (import has no side effects — pipeline entry is __main__-guarded)
import cv_pipeline_mutual_info as mi

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ARTIFACTS_DIR = mi.WORKSPACE_DIR.parent / "mi_simple_r0.7" / "artifacts"
SOURCE_WORKSPACE = ARTIFACTS_DIR.parent
RESULTS_FILE = SOURCE_WORKSPACE / "pipeline_results_svm_scalercomparison.csv"

BIN_WIDTHS = mi.BIN_WIDTHS
# All outer RS available in the source experiment's artifacts dir.
# A subset can be selected at runtime via `--rs 123 456` (or by editing
# OUTER_RS_FILTER below). Set to None to use all available RS.
OUTER_RS_LIST = mi.OUTER_RS_LIST
OUTER_RS_FILTER = None
N_OUTER_FOLDS = mi.N_OUTER_FOLDS
K_GRID = mi.K_GRID
ORIGINAL_ONLY = True  # source experiment used ORIGINAL_ONLY=True (see config.txt)

SCALERS = ["QuantileTransformer", "StandardScaler", "RobustScaler"]
CLASSIFIER_RANDOM_STATE = 42  # matches get_holdout_classifier default (RANDOM_SEED)


# -----------------------------------------------------------------------------
# Scaler factory (extends mi.make_scaler with RobustScaler)
# -----------------------------------------------------------------------------
def make_scaler(scaler_type: str):
    """Create a scaler. Extends mi.make_scaler with RobustScaler."""
    if scaler_type == "StandardScaler":
        return StandardScaler()
    if scaler_type == "QuantileTransformer":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    if scaler_type == "RobustScaler":
        return RobustScaler()
    raise ValueError(f"Unknown scaler type: {scaler_type}")


# -----------------------------------------------------------------------------
# Single-fold evaluation (mirrors evaluate_holdout_k_grid but with 3 scalers)
# -----------------------------------------------------------------------------
def evaluate_one_fold_for_scalers(X_train, y_train, X_test, y_test,
                                  feature_frequency_df, scalers, k_grid,
                                  random_state=CLASSIFIER_RANDOM_STATE):
    """
    For one fold's pair of (X_train, y_train) / (X_test, y_test) and a
    pre-computed feature_frequency_df, run top-K SVM with every supplied
    scaler. Returns a list of dicts with keys: scaler, k, auc.
    """
    # Reuse pipeline helper to honor existing rank-based top-K selection
    select_top_k = mi.select_top_k_features

    rows = []
    for k in k_grid:
        selected_features = select_top_k(feature_frequency_df, k)
        X_train_k = X_train[selected_features].values.astype(float)
        X_test_k = X_test[selected_features].values.astype(float)
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)

        for scaler_type in scalers:
            scaler = make_scaler(scaler_type)
            X_train_scaled = scaler.fit_transform(X_train_k)
            X_test_scaled = scaler.transform(X_test_k)

            clf = SVC(C=1.0, kernel="rbf", probability=True,
                      random_state=random_state)
            clf.fit(X_train_scaled, y_train_arr)
            y_score = clf.predict_proba(X_test_scaled)[:, 1]

            try:
                auc = float(roc_auc_score(y_test_arr, y_score))
            except ValueError:
                auc = 0.5
            rows.append({"scaler": scaler_type, "k": k, "auc": auc})
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments. No args => run all RS in OUTER_RS_LIST."""
    ap = argparse.ArgumentParser(
        description=("Evaluate top-K SVM AUC with different scalers using "
                     "pre-computed feature_frequencies.csv artifacts.\n"
                     "Optionally restrict to a subset of outer random states."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--rs", type=int, nargs="+", default=None,
        metavar="RS", dest="rs",
        help=("Subset of outer_rs to evaluate (e.g. --rs 123 456). "
              "If omitted, the module-level OUTER_RS_FILTER is used; "
              "if both are None, all RS in OUTER_RS_LIST are evaluated. "
              f"Available: {mi.OUTER_RS_LIST}"),
    )
    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    cli = parse_args()
    rs_filter = cli.rs if cli.rs else OUTER_RS_FILTER
    if rs_filter is None:
        rs_list = list(OUTER_RS_LIST)
        results_file = RESULTS_FILE
        tag = "all"
    else:
        rs_list = [r for r in OUTER_RS_LIST if r in set(rs_filter)]
        if not rs_list:
            print(f"Error: none of the requested RS {rs_filter} are available "
                  f"in OUTER_RS_LIST={OUTER_RS_LIST}")
            return
        # Suffix the results file so filtered runs never overwrite the full run.
        rs_suffix = "_".join(str(r) for r in rs_list)
        results_file = (SOURCE_WORKSPACE
                        / f"pipeline_results_svm_scalercomparison_RS{rs_suffix}.csv")
        tag = f"RS={rs_list}"

    print(f"Artifacts dir: {ARTIFACTS_DIR}")
    print(f"Results file:  {results_file}")
    print(f"BWs:    {BIN_WIDTHS}")
    print(f"RS:     {rs_list} (selected: {tag})")
    print(f"Folds:  {N_OUTER_FOLDS}")
    print(f"K_GRID: {K_GRID}")
    print(f"Scalers: {SCALERS}")
    print(f"Original_only filter (matches source experiment): {ORIGINAL_ONLY}")
    print()

    # Override N_OUTER_FOLDS / ORIGINAL_ONLY on the imported module so
    # `load_splits_in_memory` reproduces the exact outer splits that the
    # source experiment used when producing the artifacts.
    mi.N_OUTER_FOLDS = N_OUTER_FOLDS
    mi.ORIGINAL_ONLY = ORIGINAL_ONLY

    n_expected = (len(BIN_WIDTHS) * len(rs_list) * N_OUTER_FOLDS
                  * len(SCALERS) * len(K_GRID))
    print(f"Expected rows: {n_expected}")

    records = []
    processed = 0
    missing = 0
    for bw in BIN_WIDTHS:
        for outer_rs in rs_list:
            try:
                splits = mi.OuterCVGenerator.load_splits_in_memory(bw, outer_rs)
            except FileNotFoundError:
                print(f"[MISSING CSV] BW={bw} RS={outer_rs} -> skip")
                continue

            for fold_idx in range(N_OUTER_FOLDS):
                art_path = (ARTIFACTS_DIR / f"BW_{bw}" / f"outer_rs_{outer_rs}"
                            / f"fold_{fold_idx}" / "feature_frequencies.csv")
                if not art_path.exists():
                    print(f"[MISSING ART] {art_path.name} -> skip fold")
                    missing += 1
                    continue

                feature_frequency_df = pd.read_csv(art_path)
                X_train, y_train, X_test, y_test, test_ids, train_ids = splits[fold_idx]

                rows = evaluate_one_fold_for_scalers(
                    X_train, y_train, X_test, y_test,
                    feature_frequency_df, SCALERS, K_GRID,
                )

                for row in rows:
                    records.append({
                        "bw": bw,
                        "outer_rs": outer_rs,
                        "fold_idx": fold_idx,
                        "classifier": f"svm_{row['scaler']}",
                        "k": row["k"],
                        "auc": row["auc"],
                        "timestamp": datetime.now().isoformat(),
                    })
                processed += 1
                if processed % 20 == 0:
                    print(f"[progress] processed {processed} folds, "
                          f"rows so far={len(records)}")

    print()
    print(f"Processed folds: {processed}, missing: {missing}")
    print(f"Total rows: {len(records)} (expected {n_expected})")

    df = pd.DataFrame(records, columns=[
        "bw", "outer_rs", "fold_idx", "classifier", "k", "auc", "timestamp"])
    df.to_csv(results_file, index=False)
    print(f"Wrote: {results_file}")

    # Quick sanity check: print mean AUC per (scaler, k) aggregated across folds/RS/BW
    print(f"\nMean AUC per (scaler, k) across all BW/folds [{tag}]:")
    summary = (df.groupby(["classifier", "k"])["auc"]
               .agg(["mean", "std"])
               .reset_index())
    summary_pivot = summary.pivot(index="k", columns="classifier", values="mean")
    summary_pivot = summary_pivot.round(4)
    print(summary_pivot.to_string())


if __name__ == "__main__":
    main()
