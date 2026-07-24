"""
validate_fast_ga_tuned.py
=========================
Non-destructive validation of the TUNED fast_ga config (periodic reshuffle re-enabled,
GA_INNER_FOLDS=2, GA_N_FOLDS=4, PATIENCE=10).

It imports cv_pipeline_fast_ga (which now holds the tuned config + exec-loaded notebook
definitions) and re-runs run_pipeline on a REDUCED scope, redirecting all output to a
separate workspace so the baseline fast_ga_r0.7 results/artifacts are untouched.

Run (background, conda env):
  /home/ser/miniconda3/envs/gpu-radiomics/bin/python3 validate_fast_ga_tuned.py
"""
import sys
from pathlib import Path

# Importing runs the module's exec-load + tuned config overrides, but NOT __main__.
import cv_pipeline_fast_ga as fg

# ---- reduced, non-destructive scope ----
VAL_BW = [35, 40]
VAL_RS = [42, 123, 456]
WS = Path(__file__).parent / "workspace" / "fast_ga_tuned_validation"

# module-level globals used by run_pipeline's call site
fg.BIN_WIDTHS = VAL_BW
fg.OUTER_RS_LIST = VAL_RS
fg.WORKSPACE_DIR = WS
fg.RESULTS_FILE = WS / "pipeline_results.csv"
fg.ARTIFACTS_DIR = WS / "artifacts"

# run_pipeline / run_single_fold read these from the exec namespace (_ns)
fg._ns['BIN_WIDTHS'] = VAL_BW
fg._ns['OUTER_RS_LIST'] = VAL_RS
fg._ns['WORKSPACE_DIR'] = WS
fg._ns['RESULTS_FILE'] = fg.RESULTS_FILE
fg._ns['ARTIFACTS_DIR'] = fg.ARTIFACTS_DIR

if __name__ == "__main__":
    import time
    from datetime import datetime

    print("=" * 70)
    print("TUNED fast_ga validation")
    print(f"  BIN_WIDTHS      = {fg.BIN_WIDTHS}")
    print(f"  OUTER_RS_LIST   = {fg.OUTER_RS_LIST}")
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
    print(f"\n[Tuned validation] total time {dt:.1f}s, rows={len(df)}")
    if hasattr(fg, "print_summary"):
        fg.print_summary(df)
    else:
        print(df.groupby(['bw', 'classifier'])['auc'].mean())
