# AGENTS.md

## What This Is

Radiomics 8-fold outer CV pipeline for GA feature selection + holdout evaluation. Not a standard Python package — no `pyproject.toml`, no tests, conda-managed deps. The notebook is the source of truth; exported `.py` scripts load definitions from it via `exec`.

## Environment

```bash
conda activate gpu-radiomics
python = /home/ser/miniconda3/envs/gpu-radiomics/bin/python3
```

## Critical Files

| File | Role |
|------|------|
| `cv_pipeline.ipynb` | **Source of truth** (18 cells). All pipeline logic. |
| `cv_pipeline_mutual_info.py` | Standalone MI-based variant — multi-RS×fold（16 RS × 5 folds，**normalized** MI scores，prefilter+scaler fit once on full train set）。 |
| `cv_pipeline_mutual_info.ipynb` | MI multi-RS×fold 版 — 同為 16 RS × 5 folds，但用 **raw** MI scores（不歸一化），**per-fold** preprocessing（每個 inner fold 各自 fit prefilter+scaler）。`ORIGINAL_ONLY=True`，scaler 用 PowerTransformer。 |
| `cv_pipeline_mutual_info_auc_weighted.py` | MI variant with AUC-weighted frequency (matches GA weighting). |
| `cv_pipeline_fast_ga.py` | Fast GA variant. exec-loads cv_pipeline.ipynb (cells 0–11), overrides compute_fitness_components + _ga_fitness_components_wrapper + _collect_inner_preds + a `sys.modules` pickle alias for ProcessPoolExecutor. **Current default ("A" run): N_OUTER_FOLDS=8, GA_N_RS=16, GA_N_FOLDS=4, GA_INNER_FOLDS=2, USE_PERIODIC_RESHUFFLE=True.** |
| `cross_bw_ensemble.py` | Cross-BW soft-voting ensemble (loads from notebook via `exec`). |
| `cross_bw_ensemble_original_first.py` | Ensemble with original-first feature filtering. |
| `cross_bw_ensemble_for_MI_topK.py` | Ensemble for MI topK experiments (loads from `cv_pipeline_mutual_info_auc_weighted.py`). |
| `ga-test.ipynb` | Reference GA implementation — source of truth for GA logic. |
| `test_10case_auc.py` | Reference holdout evaluation on unseen 10-case set. |
| `sample_cases.py` | CLI utility to split CSV into balanced sample + remaining. |
| `data/` | Input CSVs (`Cine_output_20260606_binWidth_{bw}.csv`), 80 cases each. |

## Project Layout

```
workspace/
├── test1_r0.7/         # original+wavelet (PEARSON=0.7)
├── test2_r0.9/         # original+wavelet (PEARSON=0.9)
├── test3_r0.7/         # experiment with pop=30, gen=30, 3 outer RS
├── test4_r0.7/         # experiment with pop=50, gen=50, 10 outer RS
├── original_only_r0.7/ # 8-fold reference (NO acceleration): full GA 16/5/3, 6 RS [42,123,456,67,2005,2026]
├── fast_ga_r0.7/       # fast GA, CURRENT "A" config: N_OUTER_FOLDS=8, GA 16/4/2, reshuffle (3 RS, 8 folds) — matches OO
├── fast_ga_r0.7_N8_test/ # "B" MAX-SPEED GA (8/3/1, N_OUTER_FOLDS=8) — NOT equivalent to A for SVM
├── original_only/      # original_ features only (legacy)
├── mi_simple_r0.7/     # MI simple single-pass (ORIGINAL_ONLY=True, no inner CV)
├── MI_topK/            # MI-based experiments (topK_20_r0.7/, topK_30_r0.7/)
├── r_0.7_rs42_svm/     # legacy experiments
├── r_0.7_rs42_svm_v2_dynamic_lambda/
├── r_0.9_rs42_svm/
└── data_splits/        # shared split CSVs (legacy, not used by current in-memory flow)
```

Each experiment dir contains:
- `pipeline_results.csv` — long-format results (bw, outer_rs, fold_idx, classifier, k, auc)
- `artifacts/` — per-BW/RS/fold GA artifacts (feature_frequencies.csv)
- `config.txt` — snapshot of Block 1 config used to run

## Critical Rules

- **NEVER** edit `.ipynb` JSON directly. Use Jupyter-MCP tools (`insert_cell`, `edit_cell_source`, `execute_cell`).
- **NEVER** fit scalers/filters on test data. The notebook enforces fit-on-train-only.
- **NEVER** `import cv_pipeline.ipynb` as a module. The notebook contains top-level execution calls (`run_pipeline()`) that re-run the full GA. Use the `exec`-based definition-only loading pattern from `cross_bw_ensemble.py` instead.
- **NEVER** change `CLASSIFIER_TYPE` to a list — it's a single string controlling GA evolution.
- `run_test_10_case` is **deprecated**. It must NOT run GA — it loads pre-computed `feature_frequencies.csv`.
- `run_test_10_case` evaluates 4 classifiers: `['softvote-3', 'svm', 'rf', 'xgboost']`.
- Each ensemble script has a top-level `WORKSPACE_DIR` (or `TRIPLETS_TO_TEST` for MI variant). Edit there to redirect output — never hardcode paths.
- MI-based ensemble (`cross_bw_ensemble_for_MI_topK.py`) loads from `.py`, not from the notebook. Others load from the notebook.
- `cv_pipeline_mutual_info.ipynb` (multi-RS, per-fold preprocessing) vs `cv_pipeline_mutual_info.py` (multi-RS, single-pass preprocessing): the notebook does per-fold prefilter+scaler fit then **raw** MI accumulation（每個 inner fold 的 4/5 train 各自獨立 fit prefilter+scaler，MI 不歸一化）; the .py does prefilter+scaler fit **once** on full train then inner CV on the already-scaled space, with **normalized** MI scores（per-fold sum-to-1）。Different preprocessing strategies → different MI rankings. Different workspace dirs, different scalers (PowerTransformer vs StandardScaler/QuantileTransformer).
- **NEVER** modify `cv_pipeline_fast_ga.py` when editing `cv_pipeline.ipynb`, and vice versa — they are independent files with shared exec-loaded definitions.

## Key Config (Cell 1 — always verify current values before editing)

```python
BIN_WIDTHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CLASSIFIER_TYPE = 'svm'              # GA internal classifier (single string)
CLASSIFIERS = ['svm', 'rf', 'xgboost']
N_OUTER_FOLDS = 8
SCALER_TYPE_OUTER = 'QuantileTransformer'
SCALER_TYPE_INNER = 'StandardScaler'
PEARSON_THRESHOLD = 0.7
ORIGINAL_ONLY = False
MAX_K = 20
LAMBDA_PENALTY = 0.005
POP_SIZE, N_GENERATIONS = 50, 50    # vary per experiment (30/30 in test3)
PATIENCE, CROSSOVER_RATE, MUTATION_RATE = 10, 0.8, 0.05
K_GRID = [5, 10, 15, 20, 30, 40, 50]
OUTER_RS_LIST = [42, 123, 456]       # test4 uses 10 RS values
GA_N_RS, GA_N_FOLDS, GA_INNER_FOLDS = 16, 5, 3
USE_DYNAMIC_PENALTY = True           # sigmoid schedule with 20-gen warmup
USE_PERIODIC_RESHUFFLE = True        # inner-CV reshuffle every 6 gens
SAVE_GA_ARTIFACTS = True             # saves feature_frequencies.csv per fold
DATA_DIR = Path('/home/ser/pipeline/data')
WORKSPACE_DIR = Path('/home/ser/pipeline/workspace/test4_r0.7')
RESULTS_FILE = WORKSPACE_DIR / 'pipeline_results.csv'
ARTIFACTS_DIR = WORKSPACE_DIR / 'artifacts'
```

Values differ per experiment (check `workspace/{exp}/config.txt` for what was actually used).

## MI Variant 差異

`cv_pipeline_mutual_info.ipynb` 與 `cv_pipeline_mutual_info.py` 雖然都是 MI-based 特徵選取，但**方法論完全不同**，產生的 feature ranking 不可互換。

| 面向 | `.ipynb` (notebook) | `.py` |
|------|---------------------|-------|
| **Preprocessing** | 每個 inner fold 的 4/5 train 各自 fit prefilter+scaler（16 RS × 5 folds = 80 次獨立 fit） | 完整 outer training set 上 fit prefilter+scaler **一次**，inner CV 在已縮放的空間上切 split |
| **MI 權重** | **raw MI scores**（不歸一化）— 低鑑別力的 fold 貢獻 ~0 | **normalized MI scores**（per-fold sum-to-1）— 每個 fold 權重總和固定為 1，noise fold 也貢獻 1/N |
| **SCALER_TYPE_INNER** | `PowerTransformer` | `StandardScaler` |
| **SCALER_TYPE_OUTER** | `PowerTransformer` | `QuantileTransformer` |
| **ORIGINAL_ONLY** | `True` | `False` |
| **工作目錄** | `mi_multirs_r0.7_PowerTransformer` | `test3_r0.7_mi` |
| **結果檔名** | `pipeline_results.csv` | `pipeline_results_mi.csv` |
| **執行方式** | Jupyter notebook（cell-by-cell 或 exec-load） | `python cv_pipeline_mutual_info.py` 直接跑 |
| **相容碼** | 乾淨，無 deprecated 包袱 | 保留 `generate_splits()`、`load_fold()`、`resolve_workspace_dir()` 等舊版函式 |

**結論**：兩者 MI 排名會不同，因為 preprocessing leakage 與 MI weighting 策略都不同。用 `.py` 的結果去跟 notebook 的結果比較是 apples-to-oranges。

## Fast GA Variant

`cv_pipeline_fast_ga.py` is a standalone `.py` that exec-loads notebook cells 0–11, then overrides 3 functions (`compute_fitness_components`, `_ga_fitness_components_wrapper`, `_collect_inner_preds`) plus a `sys.modules` pickle alias for `ProcessPoolExecutor` compatibility. Run it with:

```bash
/home/ser/miniconda3/envs/gpu-radiomics/bin/python3 cv_pipeline_fast_ga.py
```

**Speedup levers (vs the un-accelerated notebook run):** reduce `GA_N_RS` (16→8), `GA_N_FOLDS` (5→3), `GA_INNER_FOLDS` (3→1), turn off periodic reshuffle, and reduce `N_OUTER_FOLDS` (8→5). The "B"/max-speed config (`fast_ga_r0.7_N8_test`) uses the full lever set (8/3/1, reshuffle on). **Do NOT assume max-speed ≈ reference** — see Verified Findings.

**Current default ("A") config: `N_OUTER_FOLDS=8, GA_N_RS=16, GA_N_FOLDS=4, GA_INNER_FOLDS=2, USE_PERIODIC_RESHUFFLE=True, PATIENCE=10`.** This was set after the 2026-07 investigation: `N_OUTER_FOLDS` (training-set size) is the dominant AUC driver; the lighter GA sampling in "B" measurably hurts SVM.

Bug fixes baked in: `_collect_inner_preds` override (StratifiedKFold crash with 1 split), `sys.modules` pickle alias (ProcessPoolExecutor), `_ns.update()` for `__globals__` propagation.

## Data Flow

1. CSV → `OuterCVGenerator.load_splits_in_memory(bw, outer_rs)` → in-memory (X_train, y_train, X_test, y_test, ...)
2. Train split → `run_ga_on_fold()` (GA) or `run_mi_on_fold()` (multi-RS MI) or `run_mi_simple()` (single-pass MI) → `workspace/{exp}/artifacts/BW_{bw}/outer_rs_{rs}/fold_{i}/feature_frequencies.csv`
3. Feature frequencies + holdout data → `evaluate_holdout_k_grid()` → AUC per (classifier, k)
4. Results → long-format `pipeline_results.csv`: `bw, outer_rs, fold_idx, classifier, k, auc, timestamp`

## Ensemble Scripts

All ensemble scripts share the same pattern: load definitions (skipping execution cells), then soft-vote across BWs. Each BW uses its own X_train/X_test splits (different bin widths → different feature values), so diversity comes from both feature selection AND feature values.

Configuration is at the top of each file:
- `cross_bw_ensemble.py` / `cross_bw_ensemble_original_first.py`: `TARGET_BWS` list + `WORKSPACE_DIR`
- `cross_bw_ensemble_for_MI_topK.py`: `TRIPLETS_TO_TEST` (groups of (bw1, bw2, bw3)) + `WORKSPACE_DIR`

Missing BW artifacts are skipped with warning (not crash).

## Jupyter-MCP Workflow

```python
Jupyter-MCP_connect_to_jupyter(jupyter_url="http://localhost:8888", jupyter_token="MY_TOKEN")
Jupyter-MCP_use_notebook(notebook_name="cv_pipeline", notebook_path="cv_pipeline.ipynb", mode="connect")
Jupyter-MCP_execute_cell(cell_index=N)
Jupyter-MCP_unuse_notebook(notebook_name="cv_pipeline")
```

**Gotcha**: If Jupyter-MCP is connected, VSCode cannot control the kernel. Always disconnect after work.

## Verification

No automated tests. Verification is agent-executed via Jupyter-MCP cell execution. Each block has built-in sanity checks printed in output. Integration tests are in cells 14–16.

## Verified Findings (2026-07 investigation)

These took several full pipeline runs to establish — do not re-litigate them by re-tuning GA params:

- **`N_OUTER_FOLDS` (training-set size) is the dominant AUC driver, not GA sampling.** At 80 cases, 5-fold = 64 train vs 8-fold = 70 train. With `ORIGINAL_ONLY=True` and the same 160-feature pool, fixing folds=8 makes fast_ga match `original_only` on AUC (paired Wilcoxon ns). Lighter GA sampling (e.g. "B") is a second-order effect once folds=8.
- **For SVM @ k=5: "A" (N8 + GA 16/4/2) ≈ `original_only` (p=0.435, ns); "B" max-speed (8/3/1) is significantly worse than both** (vs OO p=0.045, vs A p=0.017, ~1.4–1.9% AUC lower). **Use "A", not "B", if matching the reference matters.**
- **Feature selection is extremely stable across GA intensities** (Spearman 0.96–0.99 between OO/A/B; top-3 features identical everywhere). Residual AUC gaps come from marginal top-k differences, not ranking collapse.
- **The "SVM double-exposure" hypothesis (SVM used at both GA fitness and held-out weighting) is a fold=5 artifact** — it disappears at folds=8, so don't chase it.

## Analysis & Comparison Gotchas

- **Report SVM AUC at k=5 (or best-k), NOT mean-over-k.** Averaging over the k-grid (5–50) dilutes the peak and hides "B"'s deficit (made B look equivalent when it isn't). The pipeline's own `print_summary` leads with k=5.
- **`feature_frequencies.csv` ranking: rank by `raw_count`, not the `frequency` column.** The `frequency` value scales with how many GA runs were aggregated per fold, so it is NOT comparable across runs with different `GA_N_RS`/`GA_N_FOLDS`. `raw_count` (total selections) is comparable.
- **`fast_ga_r0.7/artifacts/` is contaminated with ~150 STALE files** from the original 5-fold/6-RS run; its `config.txt` is also stale (shows `N_OUTER_FOLDS=5`). When comparing feature frequencies, filter to `outer_rs ∈ {42,123,456}` and `fold ∈ 0..7` so you only read the 8-fold run's 240 files.
- **`original_only_r0.7` has 6 outer RS** `[42,123,456,67,2005,2026]`; the fast_ga runs use only 3 RS `[42,123,456]`. Restrict OO to those 3 RS for any aligned/paired comparison.
- When in doubt about experiment provenance, check `workspace/{exp}/config.txt` for the exact parameters used.

## Running Long Jobs

Full pipeline runs take **~1.5–4 h**. The bash tool's ~30 s timeout kills background jobs even with `nohup ... & disown` (SIGTERM hits the process group). Launch fully detached with `setsid`:

```bash
setsid /home/ser/miniconda3/envs/gpu-radiomics/bin/python3 your_script.py > /tmp/opencode/run.log 2>&1 < /dev/null &
```

Then poll the log file separately. The GA spawns a `ProcessPoolExecutor` worker pool — many `python3` processes are expected and normal.

## Uncertainty

When unsure about experiment provenance, artifact contents, or user intent — **ask the user**. Do not assume. Check `workspace/{exp}/config.txt` for the exact parameters used.
