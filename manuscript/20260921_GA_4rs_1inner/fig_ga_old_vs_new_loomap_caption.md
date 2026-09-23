# Figure legend — `fig_ga_old_vs_new_loomap.png`

GA old vs new LOO configuration, pooled LOO AUC (SVM, k=4), 10 bin widths.

- **old** = `loo_r0.7_{Scaler}_noMIinit` — 5 GA folds × 8 random seeds, inner fitness = mean AUC over a 3-fold inner CV.
- **new** = `loo_r0.7_{Scaler}_maxk4_noMI_inner1` — 5 GA folds × 4 random seeds, inner fitness = single stratified holdout (1 AUC per individual per generation).
- Both use `USE_MI_INITIALIZATION=False`, so the only changes are the GA random-seed count (8 → 4) and the inner split (3-fold average → 1 holdout).

Columns (left→right): old SS · new SS · old RS · new RS · old PT · new PT.

Each cell annotation: `AUC` on the first line, significance marks on the second.

## Marks

| Mark | Meaning | Test |
|---|---|---|
| `*` / `**` / `***` | This **GA configuration** (old or new) is significantly better than the other configuration **within the same scaler** | DeLong, two-sided, p<0.05 / 0.01 / 0.001 |

## Reading guide

- A **star on the new cell** means the 4-seed / 1-inner GA is significantly better than the 8-seed / 3-inner GA at that BW/scaler.
- No mark on either cell = the two GA configurations are not significantly different at that BW/scaler.

## Marks present at each BW

| BW | old SS | new SS | old RS | new RS | old PT | new PT |
|---|---|---|---|---|---|---|
| 5 | – | – | – | – | – | – |
| 10 | – | – | – | – | – | * |
| 15 | – | – | – | – | – | – |
| 20 | – | – | – | – | – | – |
| 25 | – | – | – | – | – | – |
| 30 | – | – | – | – | – | – |
| 35 | – | – | – | – | – | * |
| 40 | – | – | – | – | – | – |
| 45 | – | ** | – | – | – | – |
| 50 | – | – | – | – | – | – |

*Generated alongside `plot_old_vs_new_heatmap.py`.*
