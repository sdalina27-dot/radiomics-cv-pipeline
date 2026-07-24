"""
rerun_fast_ga_full.py
=====================
Full rerun of the TUNED fast_ga config over the COMPLETE scope
(all 10 BWs x 6 outer RS x 8 outer folds). force_recompute=True backs up the
old 5-fold pipeline_results.csv and writes a clean 8-fold results set to
workspace/fast_ga_r0.7 (overwriting the baseline that the N_OUTER_FOLDS=5
artifact had produced).
"""
import time
import cv_pipeline_fast_ga as fg

print("=" * 70)
print("FULL RERUN — fast_ga tuned config (N_OUTER_FOLDS=8)")
print(f"  BIN_WIDTHS      = {fg.BIN_WIDTHS}")
print(f"  OUTER_RS_LIST   = {fg.OUTER_RS_LIST}")
print(f"  N_OUTER_FOLDS   = {fg._ns['N_OUTER_FOLDS']}")
print(f"  GA_INNER_FOLDS  = {fg._ns['GA_INNER_FOLDS']}")
print(f"  GA_N_FOLDS      = {fg._ns['GA_N_FOLDS']}")
print(f"  USE_PERIODIC_RESHUFFLE = {fg._ns['USE_PERIODIC_RESHUFFLE']}")
print(f"  PATIENCE        = {fg._ns['PATIENCE']}")
print(f"  WORKSPACE_DIR   = {fg.WORKSPACE_DIR}")
print("=" * 70)

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
print(f"\n[FULL RERUN] total time {time.time()-t0:.1f}s, rows={len(df)}")
if hasattr(fg, "print_summary"):
    fg.print_summary(df)
