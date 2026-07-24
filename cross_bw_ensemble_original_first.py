#!/usr/bin/env python3
"""
cross_bw_ensemble_original_first.py
====================================
Cross-Bin-Width Soft-voting Ensemble with Original-First Feature Selection.

When ORIGINAL_ONLY=True: filters wavelet features FIRST, then selects top k.
This ensures exactly k original features are used (vs current behavior where
top k may include wavelet features that get filtered out, resulting in <k features).

Other logic identical to cross_bw_ensemble.py.

Usage:
    conda activate gpu-radiomics
    python cross_bw_ensemble_original_first.py
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── User-configurable workspace root ─────────────────────────────────────────
WORKSPACE_DIR: Path = Path(__file__).parent / "workspace" / "test1_r0.7"

# ── Load only definitions from cv_pipeline.ipynb (skip execution cells) ─────
_CV_PIPELINE_NB: Path = Path(__file__).parent / "cv_pipeline.ipynb"

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

# ── Extract symbols ─────────────────────────────────────────────────────────
ARTIFACTS_DIR: Path = WORKSPACE_DIR / "artifacts"
CLASSIFIERS: list = _cv_ns["CLASSIFIERS"]
K_GRID: list = _cv_ns["K_GRID"]
N_OUTER_FOLDS: int = _cv_ns["N_OUTER_FOLDS"]
OUTER_RS_LIST: list = _cv_ns["OUTER_RS_LIST"]
SCALER_TYPE_OUTER: str = _cv_ns["SCALER_TYPE_OUTER"]
ORIGINAL_ONLY: bool = _cv_ns["ORIGINAL_ONLY"]
OuterCVGenerator = _cv_ns["OuterCVGenerator"]
get_holdout_classifier = _cv_ns["get_holdout_classifier"]
make_scaler = _cv_ns["make_scaler"]
select_top_k_features = _cv_ns["select_top_k_features"]
filter_original_only = _cv_ns["filter_original_only"]
del _cv_ns, _definitions_source, _nb_data, _code_cells

# ── Configuration ───────────────────────────────────────────────────────────
TARGET_BWS: list[int] = [20, 35, 40] # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] 
RESULTS_OUTPUT: Path = WORKSPACE_DIR / "cross_bw_ensemble_original_first_results.csv"


def filter_features_original_first(freq_df: pd.DataFrame, k: int) -> list[str]:
    """
    Modified feature selection for ORIGINAL_ONLY=True case.
    
    When ORIGINAL_ONLY=True: filter out wavelet features FIRST, then select top k.
    When ORIGINAL_ONLY=False: use original select_top_k_features logic.
    
    This ensures that when ORIGINAL_ONLY=True, we get exactly k original features,
    not k features that may include wavelet features that get filtered out later.
    """
    if ORIGINAL_ONLY:
        # Filter to original features only (same logic as filter_original_only)
        feature_cols = [c for c in freq_df['feature'] if c.lower() not in ['label', 'casenumber', 'caseid']]
        original_cols = [c for c in feature_cols 
                        if 'original' in c.lower() and 'wavelet' not in c.lower()]
        
        # Filter frequency DataFrame to only original features
        original_freq_df = freq_df[freq_df['feature'].isin(original_cols)].copy()
        
        # Sort by rank and take top k
        sorted_df = original_freq_df.sort_values('rank')
        top_k_df = sorted_df.head(k)
        return top_k_df['feature'].tolist()
    else:
        # Use original logic for non-ORIGINAL_ONLY case
        return select_top_k_features(freq_df, k)


def load_feature_frequencies(bw: int, outer_rs: int, fold_idx: int) -> Optional[pd.DataFrame]:
    freq_path = ARTIFACTS_DIR / f"BW_{bw}" / f"outer_rs_{outer_rs}" / f"fold_{fold_idx}" / "feature_frequencies.csv"
    if not freq_path.exists():
        print(f"  [WARNING] Missing artifact: {freq_path}")
        return None
    return pd.read_csv(freq_path)


def predict_single_bw(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    freq_df: pd.DataFrame, clf_type: str, k: int,
) -> Optional[np.ndarray]:
    try:
        # Use our modified feature selection
        selected_features = filter_features_original_first(freq_df, k)
    except Exception as exc:
        print(f"  [WARNING] filter_features_original_first failed (k={k}): {exc}")
        return None
    available = [f for f in selected_features if f in X_train.columns and f in X_test.columns]
    if len(available) < 2:
        print(f"  [WARNING] < 2 overlapping features for k={k}, skipping.")
        return None
    scaler = make_scaler(SCALER_TYPE_OUTER)
    X_tr_s = scaler.fit_transform(X_train[available])
    X_te_s = scaler.transform(X_test[available])
    clf = get_holdout_classifier(clf_type)
    clf.fit(X_tr_s, y_train)
    try:
        return clf.predict_proba(X_te_s)[:, 1]
    except Exception as exc:
        print(f"  [WARNING] predict_proba failed: {exc}")
        return None


def compute_ensemble_auc(
    y_test: pd.Series, clf_type: str, k: int,
    target_bws: list[int], outer_rs: int, fold_idx: int,
    splits_by_bw: dict[int, dict],
) -> tuple[float, int, Optional[np.ndarray]]:
    """Each BW uses its OWN X_train/X_test from splits_by_bw."""
    probas_list: list[np.ndarray] = []
    for bw in target_bws:
        freq_df = load_feature_frequencies(bw, outer_rs, fold_idx)
        if freq_df is None or freq_df.empty:
            continue
        if bw not in splits_by_bw or fold_idx not in splits_by_bw[bw]:
            continue
        X_tr, y_tr, X_te, _, _, _ = splits_by_bw[bw][fold_idx]
        y_score = predict_single_bw(X_tr, y_tr, X_te, freq_df, clf_type, k)
        if y_score is not None:
            probas_list.append(y_score)
    if not probas_list:
        return 0.5, 0, None
    y_score_ensemble = np.mean(probas_list, axis=0)
    try:
        auc = roc_auc_score(y_test, y_score_ensemble)
    except ValueError:
        auc = 0.5
    return auc, len(probas_list), y_score_ensemble


def assert_y_test_aligned(splits_by_bw: dict[int, dict], fold_idx: int) -> bool:
    y_ref = None
    for bw, splits in splits_by_bw.items():
        if fold_idx not in splits:
            continue
        _, _, _, y_test_bw, _, _ = splits[fold_idx]
        if y_ref is None:
            y_ref = y_test_bw
        else:
            assert len(y_test_bw) == len(y_ref), f"y_test len mismatch BW={bw}"
            assert (y_test_bw.values == y_ref.values).all(), f"y_test content mismatch BW={bw}"
    return y_ref is not None


def run_cross_bw_ensemble(
    target_bws: list[int] = TARGET_BWS,
    outer_rs_list: list[int] = OUTER_RS_LIST,
    n_outer_folds: int = N_OUTER_FOLDS,
    classifiers: list[str] = CLASSIFIERS,
    k_grid: list[int] = K_GRID,
) -> pd.DataFrame:
    results: list[dict] = []
    # Collect ensemble prediction vectors for second-layer softvote
    ensemble_scores: dict = {}  # key: (outer_rs, fold_idx, clf, k) → (y_score_array, y_true_array)

    for outer_rs in outer_rs_list:
        print(f"\n{'='*60}\nouter_rs = {outer_rs}\n{'='*60}")
        splits_by_bw: dict[int, dict] = {}
        for bw in target_bws:
            try:
                splits_by_bw[bw] = OuterCVGenerator.load_splits_in_memory(bw, outer_rs)
                print(f"  BW={bw}: loaded {len(splits_by_bw[bw])} folds")
            except Exception as exc:
                print(f"  [WARNING] BW={bw}: load failed — {exc}")
                splits_by_bw[bw] = {}
        for fold_idx in range(n_outer_folds):
            print(f"\n  --- Fold {fold_idx} ---")
            if not assert_y_test_aligned(splits_by_bw, fold_idx):
                print(f"  [WARNING] No valid BW data, skipping.")
                continue
            # y_test from first available BW (all identical after alignment check)
            for bw, splits in splits_by_bw.items():
                if fold_idx in splits:
                    _, _, _, y_test, _, _ = splits[fold_idx]
                    break
            else:
                continue
            for clf in classifiers:
                for k in k_grid:
                    auc_ens, n_bws, y_score_ens = compute_ensemble_auc(
                        y_test, clf, k, target_bws, outer_rs, fold_idx, splits_by_bw)
                    results.append({
                        "outer_rs": outer_rs, "fold_idx": fold_idx,
                        "classifier": clf, "k": k,
                        "auc_ensemble": auc_ens,
                        "n_bws_aggregated": n_bws,
                        "n_clfs_aggregated": 0,
                    })
                    print(f"    clf={clf:<10s} k={k:>3d}  auc={auc_ens:.4f}  (n_bws={n_bws})")
                    # Store prediction vector for second-layer softvote
                    if y_score_ens is not None:
                        ensemble_scores[(outer_rs, fold_idx, clf, k)] = (
                            y_score_ens, y_test.values)

    # Second-layer softvote: average across classifiers for same (outer_rs, fold_idx, k)
    for outer_rs in outer_rs_list:
        for fold_idx in range(n_outer_folds):
            for k in k_grid:
                scores_list, y_true = [], None
                for clf in classifiers:
                    key = (outer_rs, fold_idx, clf, k)
                    if key in ensemble_scores:
                        y_sc, y_t = ensemble_scores[key]
                        scores_list.append(y_sc)
                        if y_true is None:
                            y_true = y_t
                if len(scores_list) < 2 or y_true is None:
                    continue
                avg_score = np.mean(scores_list, axis=0)
                try:
                    sv_auc = roc_auc_score(y_true, avg_score)
                except ValueError:
                    sv_auc = 0.5
                results.append({
                    "outer_rs": outer_rs, "fold_idx": fold_idx,
                    "classifier": "softvote-3",
                    "k": k,
                    "auc_ensemble": sv_auc,
                    "n_bws_aggregated": 0,
                    "n_clfs_aggregated": len(scores_list),
                })

    return pd.DataFrame(results)


def print_ensemble_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}\n  Cross-BW Ensemble Summary (Original-First)\n  Workspace: {WORKSPACE_DIR}\n{'='*70}")
    summary = df.groupby(["classifier", "k"])["auc_ensemble"].agg(["mean", "std", "count"]).reset_index()
    n_rs, n_folds = df["outer_rs"].nunique(), df["fold_idx"].nunique()
    print(f"  {n_rs} outer RS x {n_folds} folds per RS\n")
    clf_order = ["softvote-3"] + [c for c in summary["classifier"].unique() if c != "softvote-3"]
    for clf in clf_order:
        sub = summary[summary["classifier"] == clf].sort_values("k")
        if sub.empty:
            continue
        parts = [f"k={int(row['k'])}: {row['mean']:.4f}+/-{row['std']:.4f}" for _, row in sub.iterrows()]
        print(f"  {clf:<12s}: {' | '.join(parts)}")
    print(f"\n  Overall mean AUC: {df['auc_ensemble'].mean():.4f}\n{'='*70}")


if __name__ == "__main__":
    print("="*70)
    print("  Cross-BW Soft-voting Ensemble (Original-First Feature Selection)")
    print(f"  WORKSPACE_DIR  = {WORKSPACE_DIR}")
    print(f"  ARTIFACTS_DIR  = {ARTIFACTS_DIR}")
    print(f"  TARGET_BWS     = {TARGET_BWS}")
    print(f"  ORIGINAL_ONLY  = {ORIGINAL_ONLY}")
    print("="*70)
    results_df = run_cross_bw_ensemble()
    results_df.to_csv(RESULTS_OUTPUT, index=False)
    print(f"\nResults saved -> {RESULTS_OUTPUT}")
    if not results_df.empty:
        print_ensemble_summary(results_df)