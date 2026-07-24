#!/usr/bin/env python3
"""
cv_pipeline_fast_ga.py
======================
Fast variant of cv_pipeline.ipynb with ~50× reduced GA fitness cost.

Methodological diff vs cv_pipeline.ipynb (tuned variant):
  - N_OUTER_FOLDS: 8 (restored — validation proved outer-fold count / train-set size is the dominant AUC driver; cheap GA params are negligible once folds=8)
  - GA_N_RS: 16 → 8
  - GA_N_FOLDS: 5 → 4  (4 GA runs per RS, each with fresh val_idx)
  - GA_INNER_FOLDS: 3 → 2  (2-fold StratifiedKFold inner CV; was 1)
  - USE_PERIODIC_RESHUFFLE: True → True  (re-enabled; reshuffle every INNER_CV_RESHUFFLE_INTERVAL gens)
  - PATIENCE: 10 → 10  (restored)
  - POP_SIZE × N_GENERATIONS: 30 × 30 (unchanged)

Held-out AUC weighting for feature frequency: unchanged mechanism —
val_idx (1/3 of outer training data) is never seen by GA fitness,
used after GA completes to compute held-out AUC for weighting.

Override strategy: same-name redefinition of compute_fitness_components
and _ga_fitness_components_wrapper. Other functions (run_ga_on_fold, evolve,
run_ga_single_rs_fold, _run_ga_single_rs, run_pipeline) are NOT overridden —
they receive config via parameters and pick up overrides via module-global
namespace automatically.
"""

import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit

warnings.filterwarnings("ignore")

# ── Load definitions from cv_pipeline.ipynb ─────────────────────────────────
_CV_PIPELINE_NB = Path(__file__).parent / "cv_pipeline.ipynb"
_nb_data = json.loads(_CV_PIPELINE_NB.read_text(encoding="utf-8"))
_code_cells = []
for _cell in _nb_data["cells"]:
    if _cell.get("cell_type") != "code":
        continue
    _src = "".join(_cell["source"])
    if "FULL PIPELINE EXECUTION" in _src or "PART 1 INTEGRATION TEST" in _src:
        break
    _code_cells.append(_src)

_ns = {"__name__": "__cv_pipeline_fast_ga__", "__file__": str(_CV_PIPELINE_NB)}
exec(compile("\n\n".join(_code_cells), str(_CV_PIPELINE_NB), "exec"), _ns)

# ── Pickle compatibility: alias __cv_pipeline_fast_ga__ → this module ──
# Notebook functions defined via exec() get __module__ = '__cv_pipeline_fast_ga__'
# (from _ns["__name__"]).  ProcessPoolExecutor pickles _run_ga_single_rs and
# co.; pickle tries ``import __cv_pipeline_fast_ga__``, which would fail
# without this alias.
sys.modules["__cv_pipeline_fast_ga__"] = sys.modules[__name__]

# Import everything into global namespace; keep _ns alive for __globals__ patching
globals().update({k: v for k, v in _ns.items() if not k.startswith("__")})
del _nb_data, _code_cells

# ── CONFIG OVERRIDE ──────────────────────────────────────────────────────────
N_OUTER_FOLDS = 8
GA_N_RS = 8
GA_N_FOLDS = 4
GA_INNER_FOLDS = 2
POP_SIZE = 30
N_GENERATIONS = 30
LAMBDA_WARMUP_GENERATIONS = 10
USE_PERIODIC_RESHUFFLE = True
INNER_CV_RESHUFFLE_INTERVAL = 6
PATIENCE = 10
FULL_RS = list(range(GA_N_RS))

WORKSPACE_DIR = Path(__file__).parent / "workspace" / "fast_ga_r0.7_rf"
RESULTS_FILE = WORKSPACE_DIR / "pipeline_results.csv"
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"

# Alias to guard against notebook re-deriving LAMBDA_PENALTY_MAX from LAMBDA_PENALTY
LAMBDA_PENALTY_MAX = LAMBDA_PENALTY

print(f"[fast_ga] N_OUTER_FOLDS={N_OUTER_FOLDS}, GA_N_RS={GA_N_RS}, GA_N_FOLDS={GA_N_FOLDS}")
print(f"[fast_ga] GA_INNER_FOLDS={GA_INNER_FOLDS}, POP_SIZE={POP_SIZE}, N_GENERATIONS={N_GENERATIONS}")
print(f"[fast_ga] WORKSPACE_DIR={WORKSPACE_DIR}")
print(f"[fast_ga] OUTER_RS_LIST={OUTER_RS_LIST}, PEARSON_THRESHOLD={PEARSON_THRESHOLD}")
print(f"[fast_ga] Estimated speedup vs 50×50 baseline: ~{(80*2950*3*8) / (24*930*1*5):.0f}×")


# ── OVERRIDE: compute_fitness_components ────────────────────────────────────
def compute_fitness_components(chrom, X_train, y_train, classifier_type='rf',
                                inner_folds=GA_INNER_FOLDS, rng_seed=None):
    """Fast variant: supports inner_folds=1 via StratifiedShuffleSplit."""
    n_sel = len(chrom)
    if n_sel < 2:
        return 0.0, 0.0, n_sel

    chrom_features = [X_train.columns[i] for i in chrom]
    X_subset = X_train[chrom_features].values
    y_array = np.asarray(y_train)
    seed = rng_seed or RANDOM_SEED

    y_series = pd.Series(y_array)
    class_counts = y_series.value_counts()

    if inner_folds == 1:
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=seed)
            splits = list(sss.split(X_subset, y_array))
        except ValueError:
            ss = ShuffleSplit(n_splits=1, test_size=0.33, random_state=seed)
            splits = list(ss.split(X_subset))
    else:
        if len(class_counts) < 2 or (class_counts < inner_folds).any():
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
            splits = list(cv.split(X_subset))
        else:
            cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
            splits = list(cv.split(X_subset, y_array))

    scores = []
    for train_idx, val_idx in splits:
        clf = get_classifier(classifier_type, random_state=seed, inner=True)
        clf.fit(X_subset[train_idx], y_array[train_idx])
        try:
            auc = roc_auc_score(y_array[val_idx], clf.predict_proba(X_subset[val_idx])[:, 1])
        except ValueError:
            auc = 0.5
        scores.append(auc)

    mean_auc = np.mean(scores)
    scaled_auc = max(0.0, (mean_auc - 0.5) * 2.0)

    if n_sel > 1:
        corr_sel = np.abs(np.corrcoef(X_subset.T))
        np.fill_diagonal(corr_sel, 0)
        avg_red = corr_sel.sum() / (n_sel * (n_sel - 1))
    else:
        avg_red = 0.0

    return scaled_auc, avg_red, n_sel


# ── Re-wire wrapper to use overridden function ───────────────────────────────
def _ga_fitness_components_wrapper(chrom, X_scaled, y_train, clf_type, inner_folds, rng_seed_val):
    """Picklable wrapper pointing to overridden compute_fitness_components."""
    return compute_fitness_components(chrom, X_scaled, y_train, clf_type, inner_folds, rng_seed_val)


# ── OVERRIDE: _collect_inner_preds ───────────────────────────────────────────
def _collect_inner_preds(chrom, X_scaled, y_train, clf_type, inner_folds, rng_seed_val):
    """Fast variant: handles inner_folds=1 (single split) without StratifiedKFold crash."""
    chrom_features = [X_scaled.columns[i] for i in chrom]
    X_subset = X_scaled[chrom_features].values
    y_array = np.asarray(y_train)
    index = X_scaled.index.tolist()

    if inner_folds <= 1:
        # Single train/test split — StratifiedKFold(n_splits=1) is invalid
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            np.arange(len(y_array)), test_size=0.2, random_state=rng_seed_val,
            stratify=y_array
        )
        splits = [(train_idx, val_idx)]
    else:
        from sklearn.model_selection import KFold
        y_series = pd.Series(y_array)
        class_counts = y_series.value_counts()
        if len(class_counts) < 2 or (class_counts < inner_folds).any():
            cv = KFold(n_splits=inner_folds, shuffle=True, random_state=rng_seed_val)
            splits = list(cv.split(X_subset))
        else:
            cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=rng_seed_val)
            splits = list(cv.split(X_subset, y_array))

    records = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_it = X_subset[train_idx]
        y_it = y_array[train_idx]
        X_iv = X_subset[val_idx]
        y_iv = y_array[val_idx]

        clf = get_classifier(clf_type, random_state=rng_seed_val, inner=True)
        clf.fit(X_it, y_it)

        try:
            y_score = clf.predict_proba(X_iv)[:, 1]
        except Exception:
            y_score = np.full(len(y_iv), 0.5)

        for i, vi in enumerate(val_idx):
            records.append({
                'case_index': index[vi],
                'y_true': y_iv[i],
                'y_score': y_score[i],
                'inner_fold': fold_idx
            })

    return pd.DataFrame(records)


# ── Patch _ns so notebook functions (__globals__ = _ns) see overrides ──────
_ns.update({
    'N_OUTER_FOLDS': N_OUTER_FOLDS,
    'GA_N_RS': GA_N_RS,
    'GA_N_FOLDS': GA_N_FOLDS,
    'GA_INNER_FOLDS': GA_INNER_FOLDS,
    'POP_SIZE': POP_SIZE,
    'N_GENERATIONS': N_GENERATIONS,
    'LAMBDA_WARMUP_GENERATIONS': LAMBDA_WARMUP_GENERATIONS,
    'USE_PERIODIC_RESHUFFLE': USE_PERIODIC_RESHUFFLE,
    'INNER_CV_RESHUFFLE_INTERVAL': INNER_CV_RESHUFFLE_INTERVAL,
    'PATIENCE': PATIENCE,
    'FULL_RS': FULL_RS,
    'WORKSPACE_DIR': WORKSPACE_DIR,
    'RESULTS_FILE': RESULTS_FILE,
    'ARTIFACTS_DIR': ARTIFACTS_DIR,
    'LAMBDA_PENALTY_MAX': LAMBDA_PENALTY_MAX,
    'compute_fitness_components': compute_fitness_components,
    '_ga_fitness_components_wrapper': _ga_fitness_components_wrapper,
    '_collect_inner_preds': _collect_inner_preds,
})
# _ns kept alive for the lifetime of notebook function __globals__ references


# ── Main execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from datetime import datetime
    print("=" * 70)
    print("cv_pipeline_fast_ga.py — Fast GA Variant")
    print("=" * 70)

    # Save config snapshot to WORKSPACE_DIR
    config_path = WORKSPACE_DIR / "config.txt"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_lines = [
        "# cv_pipeline_fast_ga.py Configuration Snapshot\n",
        f"# Timestamp: {datetime.now().isoformat()}\n",
        f"# Workspace: {WORKSPACE_DIR}\n",
        "",
        f"N_OUTER_FOLDS={N_OUTER_FOLDS}",
        f"GA_N_RS={GA_N_RS}",
        f"GA_N_FOLDS={GA_N_FOLDS}",
        f"GA_INNER_FOLDS={GA_INNER_FOLDS}",
        f"POP_SIZE={POP_SIZE}",
        f"N_GENERATIONS={N_GENERATIONS}",
        f"PATIENCE={PATIENCE}",
        f"USE_PERIODIC_RESHUFFLE={USE_PERIODIC_RESHUFFLE}",
        f"INNER_CV_RESHUFFLE_INTERVAL={INNER_CV_RESHUFFLE_INTERVAL}",
        f"USE_DYNAMIC_PENALTY={USE_DYNAMIC_PENALTY}",
        f"LAMBDA_WARMUP_GENERATIONS={LAMBDA_WARMUP_GENERATIONS}",
        f"LAMBDA_PENALTY_MAX={LAMBDA_PENALTY_MAX}",
        f"PEARSON_THRESHOLD={PEARSON_THRESHOLD}",
        f"CLASSIFIER_TYPE={CLASSIFIER_TYPE}",
        f"CLASSIFIERS={CLASSIFIERS}",
        f"OUTER_RS_LIST={OUTER_RS_LIST}",
        f"BIN_WIDTHS={BIN_WIDTHS}",
        f"MAX_K={MAX_K}",
        f"LAMBDA_PENALTY={LAMBDA_PENALTY}",
        f"CROSSOVER_RATE={CROSSOVER_RATE}",
        f"MUTATION_RATE={MUTATION_RATE}",
        f"K_GRID={K_GRID}",
        f"RANDOM_SEED={RANDOM_SEED}",
        f"SCALER_TYPE_OUTER={SCALER_TYPE_OUTER}",
        f"SCALER_TYPE_INNER={SCALER_TYPE_INNER}",
        f"SAVE_GA_ARTIFACTS={SAVE_GA_ARTIFACTS}",
        f"ORIGINAL_ONLY={ORIGINAL_ONLY}",
    ]
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(config_lines))
    print(f"[Config saved] → {config_path}")

    t_start = time.time()
    results_df = run_pipeline(
        force_recompute=False,
        bin_widths=BIN_WIDTHS,
        outer_rs_list=OUTER_RS_LIST,
        n_outer_folds=N_OUTER_FOLDS,
        max_k=MAX_K,
        n_generations=N_GENERATIONS,
        population_size=POP_SIZE,
        n_rs=GA_N_RS,
        n_folds=GA_N_FOLDS,
        random_seed=RANDOM_SEED,
        k_grid=K_GRID,
        collect_preds=True
    )
    t_end = time.time()

    print(f"\nTotal time: {t_end - t_start:.1f}s")
    print(f"Total records: {len(results_df)}")
    print_summary(results_df)
