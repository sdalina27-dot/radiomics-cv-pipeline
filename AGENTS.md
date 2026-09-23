# AGENTS.md

## What This Is

Radiomics outer-CV pipeline for GA / MI feature selection + holdout evaluation (80 cases, 10 bin widths, 8-fold outer CV; LOO variants exist). Not a standard Python package — no `pyproject.toml`, no automated tests, conda-managed deps. The notebooks are the source of truth; `.py` scripts load definitions from them via `exec` (definition-only loading, never `import`).

## Environment

```bash
conda activate gpu-radiomics
python = /home/ser/miniconda3/envs/gpu-radiomics/bin/python3
```

LOO scripts lock BLAS threads (`OMP_NUM_THREADS=1` etc.) **before** the numpy import — keep that ordering in any new script.

## Critical Files

| File | Role |
|------|------|
| `cv_pipeline.ipynb` | **GA source of truth** (20 cells). Cells 0–14 = definitions (the exec-load boundary), cell 15 = FULL PIPELINE EXECUTION, 16/18 = integration tests, 17 = commented-out, 19 = Pearson threshold scan. Cell 14 also defines LOO helpers (`load_predictions_df`, `append_predictions`, `is_fold_done_loo`, `compute_pooled_loo_auc`, `bootstrap_ci_svm_k4`). |
| `cv_pipeline_mutual_info.ipynb` | MI variant (13 cells): `MI_N_RS=16` × `MI_N_FOLDS=5`, **raw** MI scores (not normalized), per-fold prefilter+scaler fit (each inner fold's 4/5 train fits its own), `ORIGINAL_ONLY=True`, PowerTransformer. Workspace `mi_multirs_r0.7_PowerTransformer`. |
| `cv_pipeline_LOO.py` | Leave-One-Out variant of the GA notebook (80 folds, 1 test case each). Has `--no-mi-init` CLI flag + `USE_MI_INITIALIZATION` in `LOO_OVERRIDES` to flip pure-random GA init (GA↔MI independence ablation). |
| `cv_pipeline_MI_LOO.py` | Same LOO pattern for the MI notebook (`MI_N_RS=8`). |
| `cv_pipeline_loo_svm_from_claude_web.py` | **Current GA LOO pipeline** (single file, no notebook exec-load). SVM-only (RBF), `USE_MI_INITIALIZATION=False`, `GA_N_RS=4`, `GA_N_FOLDS=5`, `GA_INNER_FOLDS=1` (single stratified holdout fitness, not a 3-fold average). CLI `--mode synth/bench/run/ci`, `--inner-folds N` (results auto-suffixed `_inner{N}`). Underlies `loo_r0.7_{Scaler}_maxk4_noMI_inner1`. |
| `cv_pipeline_mi_loo_svm_from_claude_web.py` | **Current MI LOO pipeline** (single file). `MI_N_RS=4`, `MI_N_FOLDS=5`, SVM-only; results suffixed `_inner1` for naming parity. Underlies `mi_loo_r0.7_{Scaler}_inner1`. |
| `cv_pipeline_fast_ga.py` | Fast GA variant, currently **RF-targeted**: N8, GA 8/4/2, pop/gen 30/30, workspace `fast_ga_r0.7_rf`. |
| `cv_pipeline_v2.ipynb` | Legacy wavelet-era notebook (RobustScaler, `MAX_K=20`, `K_GRID=5–50`). Don't edit for current work. |
| `compare_scalers_svm.py` | Re-evaluates existing `feature_frequencies.csv` artifacts under 3 scalers; labels classifiers `svm_<Scaler>` so `heatmap_auc.py` can read them. |
| `run_k4_evaluation.py` | Driver: runs `compare_scalers_svm.py` across 9 experiments × 3 scalers, K=4 focus. |
| `delong_comparison.py` / `delong_loo_vs_8fold.py` / `delong_loo_vs_loo.py` | DeLong's tests via `MLstatkit.Delong_test`, per (BW, RS) pooled groups. LOO inputs need per-case `pipeline_predictions.csv`; 8-fold inputs are re-scored per-case from artifacts. `delong_loo_vs_loo.py` takes 2 optional CLI args (experiment dir names). |
| `make_loo_comparison_table.py` | MI vs GA LOO table: rows BW, sections per scaler (MI/GA columns), pooled AUC (lo, hi) from `pipeline_svm_k4_ci.csv`, stars from DeLong p (1–3) on the better cell. |
| `feature_stability_rbo.py` | RBO/Jaccard stability across 24 folds (3 RS × 8 folds), methods mi_simple / original_only / original_only_RF, BW 10/20/30/40 → `workspace/stability/`. |
| `heatmap_auc.py` | AUC heatmaps from long-format CSVs. |
| `data/` | `Cine_output_20260606_binWidth_{bw}.csv`, 80 cases each. |

**Deleted — do not reference**: `cross_bw_ensemble*.py`, `sample_cases.py`, `test_10case_auc.py`, `ga-test.ipynb`, `cv_pipeline_mutual_info_auc_weighted.py`, `cv_pipeline_mutual_info.py`.

## Current Config (cv_pipeline.ipynb cell 1 — verify before editing)

```python
BIN_WIDTHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CLASSIFIER_TYPE = 'svm'                  # GA-internal classifier (single string!)
CLASSIFIERS = ['svm']                    # GA evolution only; holdout eval hardcodes ['svm','rf','xgboost'] + softvote-3
N_OUTER_FOLDS = 8
OUTER_RS_LIST = [42, 123, 456]           # outer CV seeds; FULL_RS = list(range(16)) is the GA-seed pool
SCALER_TYPE_OUTER = 'RobustScaler'       # holdout scaler (flips per experiment; LOO runs re-override via LOO_OVERRIDES)
SCALER_TYPE_INNER = 'RobustScaler'       # GA-internal scaler
PEARSON_THRESHOLD = 0.7
ORIGINAL_ONLY = True
USE_MI_INITIALIZATION = True             # seed GA init with top-30 MI features (warm start); False = pure random init (GA/MI-independence ablation)
MAX_K = 4
K_GRID = [1, 2, 3, 4]                    # target k = 4 (curse of dimensionality)
POP_SIZE, N_GENERATIONS = 30, 30
PATIENCE = 10                            # defined but NOT connected to evolve()
GA_N_RS, GA_N_FOLDS, GA_INNER_FOLDS = 16, 5, 3
TOP_N_FREQUENCY = 5                      # top-N final-population chromosomes vote, AUC-weighted
USE_DYNAMIC_PENALTY = True               # sigmoid, 10-gen warmup, LAMBDA_PENALTY_MAX=0.005
USE_PERIODIC_RESHUFFLE = True            # inner-CV reseed every INNER_CV_RESHUFFLE_INTERVAL=5 gens
WORKSPACE_DIR = Path('/home/ser/pipeline/workspace/OO_r0.7_RobustScaler_MaxK=4')
```

Values differ per experiment — `workspace/{exp}/config.txt` snapshots what was actually run. LOO dirs have **no** config.txt.

## Loop Structure

- **K-fold**: `for bw (10)` → `for outer_rs (3)` → `for fold (8)` → [`run_ga_on_fold`: `for rs (16, parallel via ProcessPoolExecutor)` → `for ga_fold (5)` → `evolve()` over 30 gens, each fitness = 3-fold inner CV] → `evaluate_holdout_k_grid` over `K_GRID` × 3 classifiers + softvote → AUC rows.
- **LOO**: `for bw` → `LeaveOneOut()` = 80 folds (test = 1 case), **parallel across 8 workers (`fork`)** → same GA with `n_workers=1` (sequential, `GA_N_RS=8`) → per-case `(case_id, y_true, y_score)` rows → pooled AUC + bootstrap CI (SVM @ k=4) computed after the loop.

GA prefilter + inner scaler are re-fit per (outer fold × GA_RS × GA inner fold) on that inner fold's 4/5 train only: LOO = 80×8×5 = 3200 fits on ~63 of 79 cases each; K-fold = 3×8×16×5 fits on 56 of 70. Val/test data only ever gets `.transform`.

## LOO Variants (cv_pipeline_LOO.py / cv_pipeline_MI_LOO.py)

- Exec-load notebook cells 0–14 (break on `'FULL PIPELINE EXECUTION'` or `'PART 1 INTEGRATION TEST'`), then apply `LOO_OVERRIDES` to **both** `_ns` and module globals (double-write — exec-loaded functions read bare globals via `__globals__` → `_ns`), plus a `sys.modules['__cv_pipeline_loo__']` pickle alias.
- Overrides: `USE_LOOCV=True` (LOO-only flag, unused in the notebook), `K_GRID=[4]`, `MAX_K=4`, pop/gen 30/30, `GA_N_RS=8`, `GA_N_FOLDS=5`, `GA_INNER_FOLDS=3`, `USE_MI_INITIALIZATION=True`, `SCALER_TYPE_OUTER`/`SCALER_TYPE_INNER`. The scaler values + `WORKSPACE_DIR` at the top of the file are edited per experiment (current dir suffixes like `_noMIinit` reflect the MI-independence ablation runs).
- Parallelism at outer-fold level: `ProcessPoolExecutor(max_workers=N_OUTER_WORKERS, mp_context='fork')` (`N_OUTER_WORKERS` is 8 or 16, currently 16); `run_ga_on_fold` is overridden with a sequential `n_workers<=1` branch to avoid nested pools.
- CLI: `--mode synth` (≤10 min synthetic validation, agent-executable — use this to verify edits), `--mode run` (5+ h, user only), `--mode ci` (bootstrap CI from existing predictions), `--no-mi-init` (disable MI-seeded GA init for the ablation).
- Outputs: `pipeline_predictions.csv` (per-case) + `pipeline_svm_k4_ci.csv` (1000-iter percentile CI, SVM @ k=4). Artifacts: `artifacts/BW_{bw}/outer_rs_loo/fold_{i}/feature_frequencies.csv`.
- Existing runs: current = `loo_r0.7_{Scaler}_maxk4_noMI_inner1` (GA) / `mi_loo_r0.7_{Scaler}_inner1` (MI); previous = `loo_r0.7_{PowerTransformer,RobustScaler,StandardScaler}` + their `_noMIinit` counterparts, `mi_loo_r0.7_*`.

## Fast GA (cv_pipeline_fast_ga.py)

Exec-loads cells 0–14, overrides `compute_fitness_components`, `_ga_fitness_components_wrapper`, `_collect_inner_preds`, `run_ga_on_fold` + `sys.modules` pickle alias + `_ns.update()` for `__globals__` propagation. Current config: N8, GA 8/4/2, pop/gen 30/30, PATIENCE 10, reshuffle interval 6, `CLASSIFIER_TYPE='rf'`, workspace `fast_ga_r0.7_rf`. **`workspace/fast_ga_r0.7/config.txt` is stale** (shows the old 5-fold 16/3/1 run) and its artifacts hold a mix of outer-RS content — filter to `outer_rs ∈ {42,123,456}`, `fold ∈ 0..7` before comparing.

## Workspace Layout

| Family | Contents |
|---|---|
| `OO_*` | GA original-only runs; notebook default = `OO_r0.7_RobustScaler_MaxK=4` (3 RS × 8 folds × K_GRID 1–4) |
| `mi_multirs_r0.7_*` | MI notebook runs (PowerTransformer / RobustScaler / StandardScaler) |
| `loo_r0.7_*` / `mi_loo_r0.7_*` | LOO runs (3 scaler variants). **Current (20260923):** `loo_r0.7_{Scaler}_maxk4_noMI_inner1` (GA 4 RS / 1 inner) and `mi_loo_r0.7_{Scaler}_inner1` (MI 4 RS). Previous: `loo_r0.7_*_noMIinit` (GA 8 RS / 3 inner, `USE_MI_INITIALIZATION=False`) and `mi_loo_r0.7_*` (MI 8 RS). Legacy MI-seeded GA `loo_r0.7_*` also present but superseded. |
| `fast_ga_r0.7*` | fast GA runs (`fast_ga_r0.7_rf` is current/complete) |
| `original_only_r0.7` | **6 outer RS** `[42,123,456,67,2005,2026]` — restrict to `{42,123,456}` for paired comparisons |
| `test1_r0.7`…`test4_r0.7`, `MI_topK`, `data_splits`, `r_*`, `stability/`, `delong test*` | legacy / analysis outputs |

Each experiment dir contains: `pipeline_results.csv` (long format `bw, outer_rs, fold_idx, classifier, k, auc, timestamp`) **or** `pipeline_predictions.csv` (LOO per-case), `artifacts/`, and (except LOO) `config.txt`.

## Critical Rules

- **NEVER** edit `.ipynb` JSON directly. Use Jupyter-MCP tools (`insert_cell`, `edit_cell_source`, `execute_cell`). If the MCP tools report `"Notebook has 0 cells"` (RTC/YDoc empty because no Jupyter session is open — `GET /api/sessions` returns `[]`), fall back to programmatic `nbformat` editing (read → edit cell `source` → `nbformat.write`; preserves outputs/metadata) instead of hand-crafting JSON. Verify no live session exists first so autosave won't clobber the on-disk edit.
- **NEVER** fit scalers/filters on test data — the notebook enforces fit-on-train-only.
- **NEVER** `import` the notebooks; use the `exec` definition-only loading pattern (breaks at `'FULL PIPELINE EXECUTION'` / `'PART 1 INTEGRATION TEST'`).
- **NEVER** change `CLASSIFIER_TYPE` to a list — it's a single string controlling GA evolution.
- `cv_pipeline_fast_ga.py` and `cv_pipeline.ipynb` are independent files sharing exec-loaded definitions — editing one does not edit the other.
- Analysis scripts (`compare_scalers_svm.py`, `delong_*`, `feature_stability_rbo.py`, `heatmap_auc.py`) configure WORKSPACE_DIR / scalers / classifier at the top of the file — edit there, never hardcode elsewhere.

## Analysis & Comparison Gotchas

- Rank `feature_frequencies.csv` by **`raw_count`**, not `frequency` — `frequency` scales with `GA_N_RS × GA_N_FOLDS` per fold and is not comparable across runs; `raw_count` (total selections) is.
- Report AUC at the run's **target k** (current runs: k=4; older runs: k=5), not mean-over-k — averaging over the k-grid dilutes the peak.
- `original_only_r0.7` has 6 outer RS vs 3 `[42,123,456]` elsewhere — restrict OO for aligned/paired comparisons.
- When in doubt about experiment provenance, check `workspace/{exp}/config.txt` (absent for LOO dirs).
- LOO keeps per-case scores and pools them per (BW, classifier, k); 8-fold runs store per-fold AUC only — DeLong-vs-8fold must re-score per-case from artifacts first.

## Manuscript (JMRI paper — FD vs HCM, GA vs MI radiomics)

**Working document:** `manuscript/manuscript 20260915.docx` — the latest-dated `.docx` in `manuscript/` is always the current draft; superseded versions (`manuscript 20260901.docx`, `my manuscript*.docx`) must not be edited. Edited **in-place** with python-docx (env `gpu-radiomics` has `python-docx` 1.2.0). Backup before editing: `cp "manuscript/manuscript 20260915.docx" /tmp/opencode/`. Never edit `.ipynb` JSON for the paper; the `.docx` is the authoritative draft. **As of 2026-09-15 all GA results use `USE_MI_INITIALIZATION=False` (no-MI-init); the MI-seeded GA runs are superseded.** Submission target: **Journal of Magnetic Resonance Imaging (JMRI)** — no 4-table limit; tables are numbered 1–6 (Table 1 demographics, Table 2 pooled AUC all BW×scaler, Table 3 MI features at the MI-optimal config, Table 4 GA features at the GA-optimal config, Table 5 diagnostic performance, Table 6 reference models [fixed shape triad / univariate MWU filter vs GA/MI, incl. PowerTransformer]) + Tables S1–S2; Figures 1 (flowchart), 2 (pooled-AUC heatmap), 3 (ROC), 4 (calibration/DCA).

**Current headline config (20260923 regeneration, per method at its own peak AUC):** **GA @ RobustScaler BW10 = 0.843 (0.746–0.925)**, **MI @ StandardScaler BW20 = 0.778 (0.668–0.872)**. The old numbers (GA 0.831 @RS10, MI 0.779 @SS25) are superseded. **Note:** with the current runs MI's peak moved SS BW25→BW20, and at the MI-optimal SS BW20 the two methods are statistically indistinguishable (GA 0.757 vs MI 0.778, DeLong p=0.586). Regenerated tables/figures (do not edit the `.docx`): `manuscript/20260923_newGA_MI/`. Supporting files:

| File / folder | Purpose |
|---|---|
| `scroll down` | see below — the source-of-truth experiment programs whose LOO predictions underpin all manuscript numbers |
| `cv_pipeline.ipynb` | **GA source of truth** (cells 0–14 = definitions; cell 15 = full pipeline). LOO helpers (cell 14) generate per-case scores → `pipeline_predictions.csv` |
| `cv_pipeline_LOO.py` | LOO GA variant (80 folds, parallel). Runs `--mode synth` (verify edits) / `--mode run` / `--mode ci`. Filter: svm, k=4 |
| `cv_pipeline_mutual_info.ipynb` | MI variant source of truth (raw MI scoring; `MI_N_RS=16`×5 folds for K-fold; per-fold prefilter+scaler) |
| `cv_pipeline_MI_LOO.py` | LOO MI variant (`MI_N_RS=8`, legacy). Same modes as `cv_pipeline_LOO.py`; underlies older `mi_loo_r0.7_*` predictions. Current MI = `cv_pipeline_mi_loo_svm_from_claude_web.py` |
| `cv_pipeline_loo_svm_from_claude_web.py` / `cv_pipeline_mi_loo_svm_from_claude_web.py` | **Current GA/MI LOO pipelines** (single-file, SVM-only). GA = 4 RS / 1 inner `_maxk4_noMI_inner1`; MI = 4 RS `_inner1`. Underpin all 20260923 manuscript numbers |
| `manuscript/manuscript 20260915.docx` | Authoritative JMRI draft (title, abstract, intro, methods, results, discussion, limitations, conclusions, references). All GA = no-MI-init |
| `manuscript/manuscript_draft.md` | Earlier markdown draft (mostly superseded by the .docx) |
| `manuscript/roc_ga_mi_comparison.png` | 4-line ROC figure (GA/MI × RS-BW10 / SS-BW25), generated by `plot_roc_comparison.py` |
| `manuscript/fig_ga_mi_loomap.png` | Heatmap figure visualizing the same data as Table 2, `plot_mi_ga_heatmap.py` |
| `manuscript/performance_summary_table.md` | AUC/Sensitivity/Specificity/Accuracy for the 4 configs (0.5 & Youden cutoffs) |
| `plot_roc_comparison.py` | Builds the 4-line ROC from 4 `pipeline_predictions.csv`; AUCs + 95% CIs from `pipeline_svm_k4_ci.csv` |
| `plot_mi_ga_heatmap.py` | Builds `fig_ga_mi_loomap.png` (10 BW × 6 [SS/RS/PT]×[MI/GA] heatmap, AUC 0.5–0.9, black text, Arial) |
| `make_loo_comparison_table.py` | Regenerates `delong test loo k=4/MI_vs_GA_LOO_comparison.{md,csv}` |
| `make_supplementary_tables.py` → `manuscript/supplementary_tables/` | Supplementary Tables S1–S6 (DeLong p-values, MI selection behavior, MWU top-4 lists, fold-level sensitivity, GA triad stability, GA RS-vs-PT frequencies) + `supplementary.docx`; asserts 15 manuscript-claim checks at build time — rerun after any artifact change |
| `baseline_models.py` → `workspace/baselines/` | P1.1 reference models under the LOO protocol: fixed shape triad (± MajorAxisLength) + univariate MWU top-4 filter; underpins Table 6 (incl. PT column: fixed triad4 0.798 > GA 0.718 @BW10, DeLong p=0.002) |
| `sensitivity_fold_level.py` → `sensitivity fold-level k=5/` | P1.2 fold-level paired Wilcoxon (8-fold × 3-seed runs, 24 fold AUCs/BW, k=5): GA>MI @RS 8/10 survives; SS/PT counts attenuate — cite as variance-aware sensitivity |
| `delong test loo k=4/MI_vs_GA_LOO_comparison.md` | MI vs GA pooled AUC table; stars `* ** ***` = better method within scaler; `†`/`‡` = scaling vs PowerTransformer |
| `loo_top4_tables/*.md` (+ `loo_top4_feature_table.py`) | Top-4 feature tables (FD vs HCM mean±std, Mann–Whitney p, sel. prob.) |
| `manuscript/20260906/noMIinit_ablation/` | GA↔MI independence ablation outputs (no-MI-init GA vs MI across 3 scalers): `tables_noMIinit.md` (revised Tables 2–5), `fig_ga_mi_loomap_noMIinit.png` + caption, per-scaler AUC/DeLong/feature-overlap, top-4 tables |
| `manuscript/20260923_newGA_MI/` | **Current manuscript tables/figures** rebuilt from the 20260923 runs (GA 4 RS/1 inner, MI 4 RS). Scripts: `ga_mi_config.py` (shared config; `SS_BW` controls the MI-optimal BW), `verify_old_vs_new.py` → `verification_old_vs_new.md`, `make_tables.py` → Tables 2,3,4,5,6,S1,S2, `plot_fig2_heatmap.py` / `plot_fig3_roc.py` / `plot_fig4_calibration_dca.py`. GA peak RS10=0.843; MI peak SS20=0.778 (old MI peak SS25 superseded). `README.md` documents provenance and the GA<MI flip at SS BW20 |
| `delong test loo k=4/<m1> vs <m2>/delong_results.csv` | Per-BW DeLong p/z for any LOO pair (m1, m2 ∈ GA/MI × SS/RS/PT) |
| `data/Cine_output_20260606_binWidth_{bw}.csv` | Pooled feature matrices (1702 feature cols = 214 original/case; 80 cases) |
| `workspace/{loo,mi_loo}_r0.7_{Scaler}/pipeline_predictions.csv` | Per-case SVM scores (filters: classifier==svm, k==4, bw) → ROC, pooled AUC |
| `workspace/{loo,mi_loo}_r0.7_{Scaler}/pipeline_svm_k4_ci.csv` | 1000-iter bootstrap CI for pooled AUC per BW |
| `workspace/loo_r0.7_{...}/artifacts/BW_{bw}/outer_rs_loo/fold_{i}/feature_frequencies.csv` | Per-fold frequency ranking (rank by **raw_count** for feature-selection tables) |

**Significance conventions in the manuscript tables/figures:**
- Stars mark the better **method** (GA vs MI) within a scaler: `*` p<0.05, `**` p<0.01, `***` p<0.001 (DeLong, two-sided).
- `†` marks an SS/RS cell significantly better than the **same method** under PowerTransformer; `‡` (double) marks a PowerTransformer cell better than the same method under the corresponding SS/RS cell (p-levels by symbol count). GA beats its PowerTransformer at BW {10,15,25,35} (RS) / {5,10,15,25,35} (SS); the only PT-wins case is MI@BW5 (‡‡).
- Scaler comparison (GA): SS vs RS statistically equivalent (no BW significant, all p>0.14); PowerTransformer significantly worse than RS/SS in ~half of BWs.
- Mann–Whitney U for feature discriminative ability; selection stability = folds a feature ranks top-4 / 80.

**Key measured numbers to quote (current 20260923 runs, see `manuscript/20260923_newGA_MI/`):** GA@RS10 AUC 0.843 (CI 0.746–0.925), Sens/Spec/Acc (0.5 thr) 0.750/0.825/0.787; MI@SS20 0.778 (0.668–0.872), 0.750/0.700/0.725. GA is significantly > MI in 6/10 BWs under RS, 5/10 under SS, 2/10 under PT; no BW significantly favours MI. Top-4 GA@RS10 = Elongation/Sphericity/Flatness (80/80 folds) + RunLengthNonUniformity (52/80), MajorAxisLength (18/80). PT@BW10 displaces Elongation from the GA top-4 (80→63/80). Table 6: GA@RS10 (0.843) ≈ fixed tetrad (0.833, p=0.68) and > fixed triad/MWU (p=0.042/0.029); at SS BW20 the fixed tetrad (0.829) > GA (0.757, p=0.008). *Superseded 20260915 numbers (GA@RS10 0.831, MI@SS25 0.779, PT-penalty decomposition, MAL 41→60) are no longer current.*

**GA↔MI independence ablation (2026-09):** rerun the GA LOO pipeline with `USE_MI_INITIALIZATION=False` (pure-random init) for all 3 scalers. Result — **no material effect**, so GA's advantage is *not* confounded by the MI seed: 0/10 BWs significant (DeLong, paired 80 cases; min p 0.078/0.089/0.195 for RS/SS/PT); mean ΔAUC (no-init − init) +0.006/−0.007/−0.004; Spearman ρ of full feature rankings ≈ 0.99; identical shape-triad top-4 (Elongation/Sphericity/Flatness 80/80, MajorAxisLength 37 vs 41/80 @RS-BW10). The MI top-30 seed is a warm-start heuristic only. Analysis lives in `manuscript/20260906/noMIinit_ablation/` (`tables_noMIinit.md`, `fig_ga_mi_loomap_noMIinit.png` + caption, per-scaler `auc_comparison_{RS,SS,PT}.md`, top-4 tables, scripts `analyze_noMIinit*.py` / `plot_noMIinit_heatmap.py` / `make_tables*.py`). Note: the SS/PT `_noMIinit` runs' `pipeline_svm_k4_ci.csv` were written empty (bootstrap step failed); the analysis scripts recompute CI inline from `pipeline_predictions.csv`.

**Current LOO runs (2026-09-23) — all manuscript numbers now use these:** the GA pipeline (`cv_pipeline_loo_svm_from_claude_web.py`) was rerun as `loo_r0.7_{Scaler}_maxk4_noMI_inner1` (5 GA folds × **4** random seeds, `GA_INNER_FOLDS=1` = single stratified-holdout fitness instead of a 3-fold average; `USE_MI_INITIALIZATION=False`), and the MI pipeline (`cv_pipeline_mi_loo_svm_from_claude_web.py`) as `mi_loo_r0.7_{Scaler}_inner1` (5 folds × **4** random seeds). Both are SVM-only (RBF, k=4). Verified against the previous 8-seed runs (see `manuscript/20260923_newGA_MI/verification_old_vs_new.md`): MI feature rankings ρ≈0.99 (top-4 Jaccard 0.78–1.00, ΔAUC within ±0.015); GA rankings ρ 0.96–0.99 with mean ΔAUC +0.006..+0.015 (slightly higher). The 1-inner-fold GA is therefore a faithful, ~2× faster substitute. Note MI's peak moved SS BW25→BW20 (0.779→0.778, while SS BW25 = 0.756), so the MI-optimal StandardScaler bin width is now 20; set `SS_BW=25` in `manuscript/20260923_newGA_MI/ga_mi_config.py` to reproduce the old matched-config framing.

## Verified Findings (2026-07 — don't re-litigate by re-tuning GA params)

- **`N_OUTER_FOLDS` (train-set size) is the dominant AUC driver, not GA sampling.** At 80 cases, 8-fold (70 train) vs 5-fold (64 train); with folds=8, light GA sampling is a second-order effect.
- For SVM @ k=5: "A" (N8 + GA 16/4/2) ≈ `original_only` (p=0.435, ns); "B" max-speed (8/3/1) significantly worse (vs OO p=0.045, vs A p=0.017). (fast_ga.py has since moved to 8/4/2 + rf.)
- Feature selection is extremely stable across GA intensities (Spearman 0.96–0.99; top-3 features identical everywhere). Residual AUC gaps come from marginal top-k differences.
- The "SVM double-exposure" hypothesis (SVM at GA fitness + held-out weighting) was a fold=5 artifact — disappears at folds=8.

## Running Long Jobs

Full pipeline runs take ~1.5–5 h. The bash tool's ~30 s timeout kills background jobs even with `nohup ... & disown` (SIGTERM hits the process group). Launch fully detached with `setsid`:

```bash
setsid /home/ser/miniconda3/envs/gpu-radiomics/bin/python3 your_script.py > /tmp/opencode/run.log 2>&1 < /dev/null &
```

Then poll the log separately. GA/MI spawn ProcessPoolExecutor worker pools — many python3 processes are expected and normal.

## Jupyter-MCP Workflow

```python
Jupyter-MCP_connect_to_jupyter(jupyter_url="http://localhost:8888", jupyter_token="MY_TOKEN")
Jupyter-MCP_use_notebook(notebook_name="cv_pipeline", notebook_path="cv_pipeline.ipynb", mode="connect")
Jupyter-MCP_execute_cell(cell_index=N)
Jupyter-MCP_unuse_notebook(notebook_name="cv_pipeline")
```

**Gotcha**: If Jupyter-MCP is connected, VSCode cannot control the kernel. Always disconnect after work. Known quirk: `use_notebook`/`read_notebook` may report `"Notebook has 0 cells"` when the notebook has no open Jupyter session (RTC/YDoc collaborative doc is empty) — see Critical Rules for the `nbformat` fallback.

## Verification

No automated tests. Verification is agent-executed: notebook integration tests are cells 16/18 (via Jupyter-MCP); LOO scripts self-verify via `--mode synth`; analysis scripts print sanity checks. Each run has built-in sanity checks in the output.

## Uncertainty

When unsure about experiment provenance, artifact contents, or user intent — **ask the user**. Do not assume.
