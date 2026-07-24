"""
validate_fast_ga_N8.py
=======================
Clean single-variable test: keep ALL tuned GA settings (reshuffle=True, GA_INNER_FOLDS=2,
GA_N_FOLDS=4, PATIENCE=10), but set N_OUTER_FOLDS=8 (was 5) to isolate the outer-fold /
training-set-size axis. At N_OUTER_FOLDS=8 the outer splits are IDENTICAL to original_only
(StratifiedKFold n_splits=8, random_state=outer_rs), so the vs-original_only comparison
becomes fold-aligned.

Non-destructive: output goes to workspace/fast_ga_tuned_N8_validation.
"""
import sys
from pathlib import Path

import cv_pipeline_fast_ga as fg

# ---- reduced, non-destructive scope ----
VAL_BW = [35, 40]
VAL_RS = [42, 123, 456]
N_OUTER = 8
WS = Path(__file__).parent / "workspace" / "fast_ga_tuned_N8_validation"

fg.BIN_WIDTHS = VAL_BW
fg.OUTER_RS_LIST = VAL_RS
fg.N_OUTER_FOLDS = N_OUTER
fg.WORKSPACE_DIR = WS
fg.RESULTS_FILE = WS / "pipeline_results.csv"
fg.ARTIFACTS_DIR = WS / "artifacts"

# run_pipeline / load_splits_in_memory read these from the exec namespace (_ns)
fg._ns['BIN_WIDTHS'] = VAL_BW
fg._ns['OUTER_RS_LIST'] = VAL_RS
fg._ns['N_OUTER_FOLDS'] = N_OUTER
fg._ns['WORKSPACE_DIR'] = WS
fg._ns['RESULTS_FILE'] = fg.RESULTS_FILE
fg._ns['ARTIFACTS_DIR'] = fg.ARTIFACTS_DIR

if __name__ == "__main__":
    import time
    from datetime import datetime

    print("=" * 70)
    print("TUNED fast_ga validation — N_OUTER_FOLDS=8 single-variable test")
    print(f"  BIN_WIDTHS      = {fg.BIN_WIDTHS}")
    print(f"  OUTER_RS_LIST   = {fg.OUTER_RS_LIST}")
    print(f"  N_OUTER_FOLDS   = {fg._ns['N_OUTER_FOLDS']}  (was 5 in prior test)")
    print(f"  GA_INNER_FOLDS  = {fg._ns['GA_INNER_FOLDS']}")
    print(f"  GA_N_FOLDS      = {fg._ns['GA_N_FOLDS']}")
    print(f"  USE_PERIODIC_RESHUFFLE = {fg._ns['USE_PERIODIC_RESHUFFLE']}")
    print(f"  INNER_CV_RESHUFFLE_INTERVAL = {fg._ns['INNER_CV_RESHUFFLE_INTERVAL']}")
    print(f"  PATIENCE        = {fg._ns['PATIENCE']}")
    print(f"  WORKSPACE_DIR   = {fg.WORKSPACE_DIR}")
    print("=" * 70)

    WS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = fg.run_pipeline(
        force_recompute=True,
        bin_widths=fg.BIN_WIDTHS,
        outer_rs_list=fg.OUTER_RS_LIST,
        n_outer_folds=fg.N_OUTER_FOLDS,
        max_k=fg.MAX_K,
        n_generations=fg.N_GENERATIONS,
        population_size=fg.POP_SIZE,
        n_rs=fg.GA_N_RS,
        n_folds=fg.GA_N_FOLDS,
        random_seed=fg.RANDOM_SEED,
        k_grid=fg.K_GRID,
        collect_preds=True,
    )
    dt = time.time() - t0
    print(f"\n[Tuned N8 validation] total time {dt:.1f}s, rows={len(df)}")
    if hasattr(fg, "print_summary"):
        fg.print_summary(df)
    else:
        print(df.groupby(['bw', 'classifier'])['auc'].mean())
