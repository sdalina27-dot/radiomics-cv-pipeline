#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cv_pipeline_loo_svm.py
======================
GA (genetic algorithm) 特徵選擇 + Leave-One-Out CV 的完整單檔 pipeline。
由 cv_pipeline.ipynb + cv_pipeline_LOO.py 合併、精簡而成。

與前版的差異
------------
1. **單一檔案**：不再 exec-load notebook，沒有 `_ns` / globals 鏡射那一層，
   所有設定與函式都在這裡，常數改了就生效。
2. **全部統一 RBF-SVM**：GA 內層 fitness、held-out AUC 加權、最終 holdout 評估
   一律使用同一組 SVC(kernel='rbf')。已移除 rf / xgboost / softvote-3。
3. **移除冗餘程式**：K-fold OuterCVGenerator、CSV 切分檔、run_test_10_case、
   K-fold pooled AUC、integration test cell、inner-pred 收集、視覺化 import 等。
4. **GA inner folds 可設為 1**：見下方 GA_INNER_FOLDS。

GA inner folds（本次重點）
--------------------------
`GA_INNER_FOLDS` 控制 fitness 的內層切分：

  * `GA_INNER_FOLDS >= 2` → StratifiedKFold(n_splits=GA_INNER_FOLDS)，
    對每一折訓練一次 SVM，取平均 AUC（目前 baseline = 3）。
  * `GA_INNER_FOLDS == 1` → StratifiedShuffleSplit(n_splits=1,
    test_size=GA_INNER_VAL_FRACTION)，只切一次 train/val、只訓練一次 SVM。
    預設 val 比例 = 1/3，與 3-fold 的單一折幾何相同，因此每次 fitness 評估的
    計算量約降為 1/3（GA 是整條 pipeline 的計算瓶頸，因此整體也接近 1/3）。

    代價：fitness 變成單一切分的估計值，雜訊較大、GA 較容易過擬合該切分。
    本 pipeline 預設開啟 USE_PERIODIC_RESHUFFLE（每 5 代重抽 inner-CV 種子），
    可部分抵銷此風險；建議用 `--mode bench` 先比較兩種設定。
    合成資料實測：inner=3 → 101s、inner=1 → 47s（2.13×）。未達理論 3× 是因為
    prefilter / scaler / MI / held-out AUC 等固定成本不隨 inner folds 縮放。

其他可降低計算量的開關（預設皆不改變 baseline 行為）
---------------------------------------------------
  * `USE_FITNESS_CACHE=True`（預設開）：同一次 GA run 內，(chromosome, cv_seed)
    相同就重用 fitness components。因為 fitness 對 (chrom, seed, data) 完全確定，
    這是**無損**加速；族群收斂後重複個體很多。實測（合成資料，pop=20/gen=15）
    評估次數 2160 → 1511（省 30%），且輸出的 feature_frequency_df 完全一致。
  * `USE_EARLY_STOPPING=False`（預設關）：開啟後，連續 PATIENCE 代最佳 fitness
    沒有改善就提前結束演化。會改變結果，僅在想再壓計算量時使用。

用法
----
  # 合成資料快速驗證（<2 分鐘）
  python cv_pipeline_loo_svm.py --mode synth

  # 比較 inner_folds=3 vs 1 的時間與 AUC（合成資料，數分鐘）
  python cv_pipeline_loo_svm.py --mode bench

  # 正式 LOO 跑（耗時數小時）
  python cv_pipeline_loo_svm.py --mode run                  # inner folds = 3
  python cv_pipeline_loo_svm.py --mode run --inner-folds 1  # inner folds = 1

  # 只從既有 predictions 重算 pooled AUC / bootstrap CI
  python cv_pipeline_loo_svm.py --mode ci

注意：`--inner-folds` 預設會把結果寫到帶有 `_inner{N}` 後綴的 workspace，
避免 inner=1 與 inner=3 的 predictions 混進同一個 CSV。
"""

# ── BLAS thread locking（必須在 import numpy 之前）────────────────────────────
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import json
import time
import shutil
import tempfile
import warnings
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import (StandardScaler, RobustScaler,
                                   QuantileTransformer, PowerTransformer)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import (LeaveOneOut, KFold, StratifiedKFold,
                                     ShuffleSplit, StratifiedShuffleSplit)
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

warnings.filterwarnings('ignore')


# =============================================================================
# 1. 全域設定
# =============================================================================

# ── 資料 ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/home/ser/pipeline/data')
CSV_TEMPLATE = 'Cine_output_20260606_binWidth_{bw}.csv'
BIN_WIDTHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ORIGINAL_ONLY = True          # 只用 original_ 特徵（排除 wavelet）

# ── 前處理 ────────────────────────────────────────────────────────────────────
PEARSON_THRESHOLD = 0.7       # 訓練集內的 Pearson 相關性過濾門檻
SCALER_TYPE_INNER = 'PowerTransformer'   # GA 內部（每個 GA train fold 各 fit 一次）
SCALER_TYPE_OUTER = 'PowerTransformer'   # 最終 holdout（每個 LOO fold fit 一次）

# ── 分類器（全 pipeline 唯一模型：RBF-SVM）──────────────────────────────────
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
RANDOM_SEED = 42

# ── GA 超參數 ────────────────────────────────────────────────────────────────
MAX_K = 4                     # soft penalty 的目標特徵數上限
POP_SIZE = 30
N_GENERATIONS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
LAMBDA_PENALTY = 0.005        # 同時作為動態懲罰的上限 lambda_max

# init_population 的 k 抽樣範圍：k ~ U[max(2, MAX_K+INIT_K_LO_OFFSET),
#                                      min(MAX_K+INIT_K_HI_OFFSET, n_features+1))
# 預設區間遠大於 MAX_K，讓早期探索不受限，再由 soft penalty 收斂到 <= MAX_K。
INIT_K_LO_OFFSET = -10
INIT_K_HI_OFFSET = 15
N_MI_SEEDED = 5               # 用 MI 種子化的初始個體數
N_MI_TOP = 30                 # MI 排名前 N 名做為種子個體的取樣池
USE_MI_INITIALIZATION = False  # False = 純隨機初始化（GA↔MI 獨立性消融）

# GA 外層重複：GA_N_RS 個 random seed × GA_N_FOLDS 個 StratifiedKFold 折
GA_N_RS = 4
GA_N_FOLDS = 5

# ── 【本次重點】GA 內層 fitness 切分 ────────────────────────────────────────
# >=2 → StratifiedKFold(GA_INNER_FOLDS)，取各折 AUC 平均（baseline = 3）
#  =1 → 單次 StratifiedShuffleSplit(test_size=GA_INNER_VAL_FRACTION)，只 fit 1 次
GA_INNER_FOLDS = 3
GA_INNER_VAL_FRACTION = 1.0 / 3.0   # 只在 GA_INNER_FOLDS == 1 時使用

# ── Fitness 組成 ─────────────────────────────────────────────────────────────
# fitness = scaled_auc^2 / (1 + avg_redundancy) - lambda_p * max(0, n_sel - MAX_K)
# scaled_auc = max(0, (mean_auc - 0.5) * 2)
USE_REDUNDANCY_IN_FITNESS = False   # False → avg_redundancy 強制為 0

# 動態懲罰係數：前 LAMBDA_WARMUP_GENERATIONS 代 lambda=0，之後遞增到 LAMBDA_PENALTY
USE_DYNAMIC_PENALTY = True
LAMBDA_WARMUP_GENERATIONS = 10
LAMBDA_SCHEDULE_TYPE = 'sigmoid'    # 'linear' 或 'sigmoid'

# 定期重抽 inner-CV 種子，避免 GA 過擬合固定切分
USE_PERIODIC_RESHUFFLE = True
INNER_CV_RESHUFFLE_INTERVAL = 5

# ── 計算量開關 ───────────────────────────────────────────────────────────────
# 無損：同一次 GA run 內快取 (chromosome, cv_seed) → fitness components
USE_FITNESS_CACHE = True
# 有損（會改變結果）：連續 PATIENCE 代無改善就停止演化
USE_EARLY_STOPPING = False
PATIENCE = 10

# ── 特徵頻率累計 ─────────────────────────────────────────────────────────────
# 每個 GA run 結束後，取最終族群 fitness 前 TOP_N_FREQUENCY 名個體，
# 各自在該折的 val_idx 上算 held-out AUC，以 max(0,(auc-0.5)*2) 當權重累計頻率。
TOP_N_FREQUENCY = 5

# ── Holdout 評估 ─────────────────────────────────────────────────────────────
K_GRID = [4]                  # 由頻率排名取 Top-K 特徵；LOO 下建議 <= MAX_K

# ── 輸出 ─────────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path('/home/ser/pipeline/workspace')
SAVE_GA_ARTIFACTS = True

# ── 平行化 ───────────────────────────────────────────────────────────────────
N_OUTER_WORKERS = 16          # outer LOO fold 層級的 process 數
MP_START_METHOD = 'fork'      # Linux / WSL2

# ── 衍生路徑（由 apply_config() 計算，CLI 覆寫後會重算）─────────────────────
RUN_NAME = None
WORKSPACE_DIR = None
ARTIFACTS_DIR = None
PREDICTIONS_FILE = None
CI_FILE = None
CONFIG_FILE = None

PREDICTION_COLUMNS = ['bw', 'outer_rs', 'fold_idx', 'classifier', 'k',
                      'case_id', 'y_true', 'y_score', 'timestamp']


def build_run_name() -> str:
    """由目前設定推導 workspace 子目錄名，讓不同設定的結果不會混在一起。"""
    mi = 'MI' if USE_MI_INITIALIZATION else 'noMI'
    return (f'loo_r{PEARSON_THRESHOLD}_{SCALER_TYPE_OUTER}'
            f'_maxk{MAX_K}_{mi}_inner{GA_INNER_FOLDS}')


def apply_config(workspace_dir=None) -> None:
    """重算所有衍生路徑。CLI 覆寫常數之後必須呼叫一次。"""
    global RUN_NAME, WORKSPACE_DIR, ARTIFACTS_DIR
    global PREDICTIONS_FILE, CI_FILE, CONFIG_FILE

    RUN_NAME = build_run_name()
    WORKSPACE_DIR = Path(workspace_dir) if workspace_dir else WORKSPACE_ROOT / RUN_NAME
    ARTIFACTS_DIR = WORKSPACE_DIR / 'artifacts'
    PREDICTIONS_FILE = WORKSPACE_DIR / 'pipeline_predictions.csv'
    CI_FILE = WORKSPACE_DIR / 'pipeline_svm_k{}_ci.csv'.format(K_GRID[-1] if K_GRID else 'NA')
    CONFIG_FILE = WORKSPACE_DIR / 'run_config.json'


CONFIG_KEYS = [
    'BIN_WIDTHS', 'ORIGINAL_ONLY', 'PEARSON_THRESHOLD',
    'SCALER_TYPE_INNER', 'SCALER_TYPE_OUTER',
    'SVM_KERNEL', 'SVM_C', 'SVM_GAMMA', 'RANDOM_SEED',
    'MAX_K', 'POP_SIZE', 'N_GENERATIONS', 'CROSSOVER_RATE', 'MUTATION_RATE',
    'LAMBDA_PENALTY', 'USE_MI_INITIALIZATION',
    'GA_N_RS', 'GA_N_FOLDS', 'GA_INNER_FOLDS', 'GA_INNER_VAL_FRACTION',
    'USE_REDUNDANCY_IN_FITNESS', 'USE_DYNAMIC_PENALTY',
    'LAMBDA_WARMUP_GENERATIONS', 'LAMBDA_SCHEDULE_TYPE',
    'USE_PERIODIC_RESHUFFLE', 'INNER_CV_RESHUFFLE_INTERVAL',
    'USE_FITNESS_CACHE', 'USE_EARLY_STOPPING', 'PATIENCE',
    'TOP_N_FREQUENCY', 'K_GRID', 'N_OUTER_WORKERS',
]


def config_dict() -> dict:
    g = globals()
    cfg = {k: g[k] for k in CONFIG_KEYS}
    cfg['RUN_NAME'] = RUN_NAME
    cfg['WORKSPACE_DIR'] = str(WORKSPACE_DIR)
    return cfg


def print_config() -> None:
    print('=' * 70)
    print('Configuration')
    print('=' * 70)
    for k, v in config_dict().items():
        print(f'  {k:<28s}: {v}')
    inner_note = ('單次 stratified holdout (val={:.3f})'.format(GA_INNER_VAL_FRACTION)
                  if GA_INNER_FOLDS <= 1 else f'StratifiedKFold({GA_INNER_FOLDS}) 取平均')
    print(f'  → GA inner fitness 切分: {inner_note}')
    print('=' * 70)


# =============================================================================
# 2. 資料載入
# =============================================================================

def resolve_binwidth_csv(bw) -> Path:
    return DATA_DIR / CSV_TEMPLATE.format(bw=bw)


def bw_csv_exists(bw) -> bool:
    return resolve_binwidth_csv(bw).exists()


def find_case_id_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in ('casenumber', 'caseid'):
            return col
    raise ValueError('No CaseNumber / CaseID column found.')


def find_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() == 'label':
            return col
    raise ValueError('No Label column found.')


def filter_original_only(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 original_（且非 wavelet）特徵，並保留 CaseNumber / Label。"""
    non_feature = [c for c in df.columns
                   if c.lower() in ('label', 'casenumber', 'caseid')]
    feature = [c for c in df.columns
               if c.lower() not in ('label', 'casenumber', 'caseid')]
    original = [c for c in feature
                if 'original' in c.lower() and 'wavelet' not in c.lower()]
    return df[non_feature + original]


def safe_save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_loo_splits(bw) -> dict:
    """讀取某個 bin width 的 CSV，回傳 {fold_idx: (X_tr, y_tr, X_te, y_te, te_ids)}。

    LeaveOneOut：N 個樣本 → N 個 fold，每個 test 只有 1 筆。
    """
    df = pd.read_csv(resolve_binwidth_csv(bw))
    if ORIGINAL_ONLY:
        df = filter_original_only(df)

    case_id_col = find_case_id_column(df)
    label_col = find_label_column(df)

    splits = {}
    for fold_idx, (train_idx, test_idx) in enumerate(LeaveOneOut().split(df)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        splits[fold_idx] = (
            train_df.drop(columns=[case_id_col, label_col]),
            train_df[label_col].reset_index(drop=True),
            test_df.drop(columns=[case_id_col, label_col]),
            test_df[label_col].reset_index(drop=True),
            test_df[case_id_col].reset_index(drop=True),
        )
    return splits


# =============================================================================
# 3. 前處理與分類器
# =============================================================================

def make_scaler(scaler_type: str):
    if scaler_type == 'StandardScaler':
        return StandardScaler()
    if scaler_type == 'RobustScaler':
        return RobustScaler()
    if scaler_type == 'QuantileTransformer':
        return QuantileTransformer(output_distribution='normal',
                                   random_state=RANDOM_SEED)
    if scaler_type == 'PowerTransformer':
        return PowerTransformer(method='yeo-johnson', standardize=True)
    raise ValueError(f'Unknown scaler type: {scaler_type}')


def make_svm(random_state: int = None) -> SVC:
    """全 pipeline 唯一的分類器：RBF-SVM。"""
    return SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA,
               probability=True,
               random_state=RANDOM_SEED if random_state is None else random_state)


def safe_auc(y_true, y_score, default=np.nan) -> float:
    """單一類別時回傳 default。

    sklearn < 1.8 會丟 ValueError，>= 1.8 改為回傳 NaN 並發出警告，
    這裡把兩種行為統一掉，避免 NaN 汙染後續統計。
    """
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        return default
    return default if auc is None or np.isnan(auc) else float(auc)


def pearson_correlation_filter(X: pd.DataFrame, threshold: float) -> list:
    """高相關特徵群中只保留一個（以欄位順序為優先）。"""
    names = X.columns.tolist()
    if threshold >= 1.0 or X.shape[1] <= 1:
        return names

    corr = np.abs(np.corrcoef(X.values, rowvar=False))
    n = X.shape[1]
    to_drop = set()
    for a in range(n):
        if a in to_drop:
            continue
        for b in range(a + 1, n):
            if b in to_drop:
                continue
            if corr[a, b] > threshold:
                to_drop.add(b)
    return [names[i] for i in range(n) if i not in to_drop]


def fit_train_only_prefilter(X_train: pd.DataFrame, threshold: float = None) -> list:
    """VarianceThreshold(0) + Pearson 過濾，只 fit 在訓練資料上。"""
    if threshold is None:
        threshold = PEARSON_THRESHOLD
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    X_vt = X_train.loc[:, vt.get_support()]
    return pearson_correlation_filter(X_vt, threshold)


# =============================================================================
# 4. GA fitness
# =============================================================================

def make_inner_splits(X_subset: np.ndarray, y_array: np.ndarray,
                      inner_folds: int, seed: int) -> list:
    """建立 GA fitness 的內層切分。

    inner_folds >= 2 → StratifiedKFold(inner_folds)，共 inner_folds 次模型訓練。
    inner_folds == 1 → StratifiedShuffleSplit(n_splits=1,
                       test_size=GA_INNER_VAL_FRACTION)，只 1 次模型訓練。

    類別數不足以分層時自動退回非分層版本（KFold / ShuffleSplit）。
    """
    class_counts = pd.Series(y_array).value_counts()
    n_samples = len(y_array)

    if inner_folds <= 1:
        test_size = GA_INNER_VAL_FRACTION
        # 每個類別至少要有 2 筆才能分層切一次 train/val
        stratifiable = len(class_counts) >= 2 and (class_counts >= 2).all()
        if stratifiable:
            n_val = max(1, int(round(n_samples * test_size)))
            # 分層時驗證集至少要能裝下每個類別各 1 筆
            if n_val >= len(class_counts) and (n_samples - n_val) >= len(class_counts):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                             random_state=seed)
                return list(sss.split(X_subset, y_array))
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return list(ss.split(X_subset))

    if len(class_counts) < 2 or (class_counts < inner_folds).any():
        cv = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        return list(cv.split(X_subset))

    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    return list(cv.split(X_subset, y_array))


def compute_fitness_components(chrom, X_train, y_train, inner_folds=None, rng_seed=None):
    """計算 fitness 中「昂貴」的部分，不含懲罰項，可被快取。

    Returns
    -------
    (scaled_auc, avg_redundancy, n_sel)
    """
    n_sel = len(chrom)
    if n_sel < 2:
        return 0.0, 0.0, n_sel

    if inner_folds is None:
        inner_folds = GA_INNER_FOLDS
    seed = RANDOM_SEED if rng_seed is None else rng_seed

    chrom_features = [X_train.columns[i] for i in chrom]
    X_subset = X_train[chrom_features].values
    y_array = np.asarray(y_train)

    scores = []
    for train_idx, val_idx in make_inner_splits(X_subset, y_array, inner_folds, seed):
        clf = make_svm(random_state=seed)
        clf.fit(X_subset[train_idx], y_array[train_idx])
        # val 只有單一類別時 AUC 無定義 → 記為 0.5（等同隨機）
        scores.append(safe_auc(y_array[val_idx],
                               clf.predict_proba(X_subset[val_idx])[:, 1],
                               default=0.5))

    mean_auc = float(np.mean(scores))
    scaled_auc = max(0.0, (mean_auc - 0.5) * 2.0)

    if USE_REDUNDANCY_IN_FITNESS and n_sel > 1:
        corr_sel = np.abs(np.corrcoef(X_subset.T))
        np.fill_diagonal(corr_sel, 0)
        avg_red = float(corr_sel.sum() / (n_sel * (n_sel - 1)))
    else:
        avg_red = 0.0

    return scaled_auc, avg_red, n_sel


def apply_penalty(scaled_auc, avg_redundancy, n_sel, lambda_p, max_k=None):
    """純算術：把 components 套上當代 lambda_p 得到最終 fitness（不訓練模型）。"""
    if max_k is None:
        max_k = MAX_K
    fitness = (scaled_auc ** 2) / (1.0 + avg_redundancy)
    fitness -= lambda_p * max(0, n_sel - max_k)
    return fitness


def lambda_schedule(gen, n_generations, warmup_generations=None,
                    lambda_max=None, schedule_type=None):
    """第 gen 代（0-indexed）應使用的 lambda_p。"""
    if warmup_generations is None:
        warmup_generations = LAMBDA_WARMUP_GENERATIONS
    if lambda_max is None:
        lambda_max = LAMBDA_PENALTY
    if schedule_type is None:
        schedule_type = LAMBDA_SCHEDULE_TYPE

    if gen < warmup_generations:
        return 0.0

    remaining = max(1, (n_generations - 1) - warmup_generations)
    progress = min(1.0, max(0.0, (gen - warmup_generations) / remaining))

    if schedule_type == 'linear':
        return lambda_max * progress
    if schedule_type == 'sigmoid':
        k, mid = 10.0, 0.5
        s = 1.0 / (1.0 + np.exp(-k * (progress - mid)))
        s_lo = 1.0 / (1.0 + np.exp(-k * (0.0 - mid)))
        s_hi = 1.0 / (1.0 + np.exp(-k * (1.0 - mid)))
        return lambda_max * (s - s_lo) / (s_hi - s_lo)
    raise ValueError(f"Unknown schedule_type: {schedule_type}")


def derive_reshuffle_seed(base_seed, gen):
    """確定性推導：同一代必得相同種子，不同代必不同。"""
    return int((base_seed * 1000 + gen) % (2 ** 31 - 1))


# =============================================================================
# 5. GA 運算子與演化迴圈
# =============================================================================

def init_population(rng, n_features, population_size, max_k, mi_scores=None):
    """染色體 = 排序後的特徵索引 tuple。前 N_MI_SEEDED 個用 MI 前段特徵種子化。"""
    k_lo = max(2, min(max_k + INIT_K_LO_OFFSET, n_features))
    k_hi = max(k_lo + 1, min(max_k + INIT_K_HI_OFFSET, n_features + 1))
    n_seeded = min(N_MI_SEEDED, population_size)

    top_mi = None
    if mi_scores is not None and len(mi_scores) == n_features:
        top_mi = np.argsort(mi_scores)[-min(N_MI_TOP, n_features):]

    population = []
    for i in range(population_size):
        k = max(2, int(rng.integers(k_lo, k_hi)))
        if i < n_seeded and top_mi is not None:
            chosen = rng.choice(top_mi, size=min(k, len(top_mi)), replace=False)
        else:
            chosen = rng.choice(n_features, size=min(k, n_features), replace=False)
        population.append(tuple(sorted(int(c) for c in chosen)))
    return population


def tournament_select(population, fitness_scores, rng):
    cand = rng.choice(len(population), size=min(3, len(population)), replace=False)
    best = cand[int(np.argmax([fitness_scores[c] for c in cand]))]
    return population[best]


def crossover(parent1, parent2, rng, crossover_rate=None):
    """Two-point crossover（不強制 max_k，由 soft penalty 約束）。"""
    if crossover_rate is None:
        crossover_rate = CROSSOVER_RATE
    if rng.random() >= crossover_rate:
        return parent1, parent2

    union = sorted(set(parent1) | set(parent2))
    if len(union) < 2:
        return parent1, parent2

    pts = sorted(rng.choice(len(union), size=2, replace=False))
    s1, s2 = set(parent1), set(parent2)
    c1_set, c2_set = set(), set()
    for idx, gene in enumerate(union):
        inside = pts[0] <= idx < pts[1]
        src1, src2 = (s2, s1) if inside else (s1, s2)
        if gene in src1:
            c1_set.add(gene)
        if gene in src2:
            c2_set.add(gene)

    c1 = tuple(sorted(c1_set)) if len(c1_set) >= 2 else parent1
    c2 = tuple(sorted(c2_set)) if len(c2_set) >= 2 else parent2
    return c1, c2


def mutate(chromosome, rng, n_features, mutation_rate=None):
    """Bit-flip mutation（不強制 max_k）。"""
    if mutation_rate is None:
        mutation_rate = MUTATION_RATE
    curr = set(chromosome)
    new = set()
    for f in range(n_features):
        if f in curr:
            if rng.random() >= mutation_rate:
                new.add(f)
        elif rng.random() < mutation_rate:
            new.add(f)
    return tuple(sorted(new)) if len(new) >= 2 else chromosome


def evolve(rng, population, n_generations, max_k, X_scaled, y_train,
           inner_folds, base_cv_seed, crossover_rate=None, mutation_rate=None):
    """演化族群（steady-state：新子代取代目前最差個體）。

    - 動態懲罰：每代重算 lambda_p，對整個族群的 components 重新套用（純算術）。
    - 定期重抽：每 INNER_CV_RESHUFFLE_INTERVAL 代換 inner-CV 種子並重估族群。
    - fitness 快取：(chrom, cv_seed) → components，無損加速。
    - 提前停止：USE_EARLY_STOPPING 時，連續 PATIENCE 代無改善就結束。

    Returns
    -------
    (final_population, final_fitness_scores, reshuffle_log, n_evals)
    """
    if crossover_rate is None:
        crossover_rate = CROSSOVER_RATE
    if mutation_rate is None:
        mutation_rate = MUTATION_RATE

    n_features = X_scaled.shape[1]
    pop = list(population)
    pop_size = len(pop)
    current_cv_seed = base_cv_seed
    reshuffle_log = []

    cache = {} if USE_FITNESS_CACHE else None
    n_evals = 0

    def components_of(chrom):
        nonlocal n_evals
        if cache is not None:
            key = (chrom, current_cv_seed)
            hit = cache.get(key)
            if hit is not None:
                return hit
        out = compute_fitness_components(chrom, X_scaled, y_train,
                                         inner_folds=inner_folds,
                                         rng_seed=current_cv_seed)
        n_evals += 1
        if cache is not None:
            cache[(chrom, current_cv_seed)] = out
        return out

    components_cache = [components_of(c) for c in pop]

    def current_lambda(gen):
        if not USE_DYNAMIC_PENALTY:
            return LAMBDA_PENALTY
        return lambda_schedule(gen, n_generations,
                               LAMBDA_WARMUP_GENERATIONS, LAMBDA_PENALTY,
                               LAMBDA_SCHEDULE_TYPE)

    fitness_scores = [apply_penalty(*c, current_lambda(0), max_k=max_k)
                      for c in components_cache]

    best_so_far = max(fitness_scores)
    stale = 0

    for gen in range(n_generations):
        if (USE_PERIODIC_RESHUFFLE and gen > 0
                and gen % INNER_CV_RESHUFFLE_INTERVAL == 0):
            current_cv_seed = derive_reshuffle_seed(base_cv_seed, gen)
            reshuffle_log.append((gen, current_cv_seed))
            components_cache = [components_of(c) for c in pop]

        lam = current_lambda(gen)
        fitness_scores = [apply_penalty(*c, lam, max_k=max_k)
                          for c in components_cache]

        for _ in range(pop_size // 2):
            p1 = tournament_select(pop, fitness_scores, rng)
            p2 = tournament_select(pop, fitness_scores, rng)
            c1, c2 = crossover(p1, p2, rng, crossover_rate=crossover_rate)
            c1 = mutate(c1, rng, n_features, mutation_rate=mutation_rate)
            c2 = mutate(c2, rng, n_features, mutation_rate=mutation_rate)
            for child in (c1, c2):
                child_comp = components_of(child)
                f = apply_penalty(*child_comp, lam, max_k=max_k)
                worst = int(np.argmin(fitness_scores))
                if f > fitness_scores[worst]:
                    pop[worst] = child
                    fitness_scores[worst] = f
                    components_cache[worst] = child_comp

        if USE_EARLY_STOPPING:
            gen_best = max(fitness_scores)
            if gen_best > best_so_far + 1e-12:
                best_so_far = gen_best
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

    return list(pop), list(fitness_scores), reshuffle_log, n_evals


# =============================================================================
# 6. GA 頻率引擎
# =============================================================================

def build_feature_frequency_df(frequency_counter, raw_count_counter=None) -> pd.DataFrame:
    """frequency = held-out AUC 加權分數；raw_count = 未加權出現次數。"""
    items = sorted(frequency_counter.items(), key=lambda x: (-x[1], x[0]))
    df = pd.DataFrame([{'feature': f, 'frequency': v, 'rank': i}
                       for i, (f, v) in enumerate(items, 1)])
    if raw_count_counter is not None and not df.empty:
        df['raw_count'] = df['feature'].map(raw_count_counter).fillna(0).astype(int)
    return df


def run_ga_single_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                       rs, fold_idx, max_k, n_generations, population_size,
                       inner_folds, crossover_rate, mutation_rate, top_n):
    """在單一 (rs, GA fold) 上跑一次 GA。

    流程：
      1. 只用 train fold 做 VarianceThreshold + Pearson 過濾
      2. 只用 train fold fit scaler（GA 內部全程用縮放後資料）
      3. （可選）MI 種子化初始族群
      4. 演化
      5. 取最終族群 fitness 前 top_n 名，各自在 val fold 上算 held-out AUC

    Returns
    -------
    (results, n_evals)  results = list[(selected_feature_names, held_out_auc)]
    """
    rng = np.random.default_rng(rs * 100 + fold_idx)
    base_seed = rs * 100 + fold_idx

    selected_features = fit_train_only_prefilter(X_train_fold)
    if len(selected_features) < 2:
        selected_features = list(X_train_fold.columns[:2])
    X_filtered = X_train_fold[selected_features]

    scaler = make_scaler(SCALER_TYPE_INNER)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered),
                            columns=X_filtered.columns, index=X_filtered.index)

    mi_scores = None
    if USE_MI_INITIALIZATION:
        try:
            mi_scores = mutual_info_classif(X_scaled, y_train_fold,
                                            random_state=RANDOM_SEED)
        except Exception:
            mi_scores = None

    population = init_population(rng, len(selected_features), population_size,
                                 max_k, mi_scores=mi_scores)

    final_pop, final_fitness, _reshuffle_log, n_evals = evolve(
        rng, population, n_generations, max_k, X_scaled, y_train_fold,
        inner_folds=inner_folds, base_cv_seed=base_seed,
        crossover_rate=crossover_rate, mutation_rate=mutation_rate)

    # 取 fitness 前 top_n 名的不重複個體
    if top_n is None:
        top_n = len(final_pop)
    top_n = max(1, min(top_n, len(final_pop)))
    seen, top_chroms = set(), []
    for idx in np.argsort(final_fitness)[::-1]:
        chrom = final_pop[idx]
        if chrom in seen:
            continue
        seen.add(chrom)
        top_chroms.append(chrom)
        if len(top_chroms) >= top_n:
            break

    # 每個 top-N 個體在 val fold 上的 held-out AUC → 作為頻率權重
    results = []
    X_val_scaled_full = None
    if X_val_fold is not None and y_val_fold is not None and len(y_val_fold) > 0:
        try:
            X_val_scaled_full = pd.DataFrame(
                scaler.transform(X_val_fold[selected_features]),
                columns=selected_features, index=X_val_fold.index)
        except Exception:
            X_val_scaled_full = None

    y_val_array = np.asarray(y_val_fold) if y_val_fold is not None else None

    for rank, chrom in enumerate(top_chroms):
        selected = [X_filtered.columns[i] for i in chrom]
        held_out_auc = 0.5
        if X_val_scaled_full is not None:
            try:
                clf = make_svm(random_state=base_seed + rank)
                clf.fit(X_scaled[selected], np.asarray(y_train_fold))
                y_score = clf.predict_proba(X_val_scaled_full[selected].values)[:, 1]
                held_out_auc = safe_auc(y_val_array, y_score, default=0.5)
            except Exception as e:
                if rank == 0:
                    print(f'    [WARNING] held-out AUC failed '
                          f'(rs={rs}, fold={fold_idx}): {type(e).__name__}: {e}')
                held_out_auc = 0.5
        results.append((selected, held_out_auc))

    return results, n_evals


def run_ga_single_rs(X_train, y_train, rs, n_folds, max_k, n_generations,
                     population_size, inner_folds, crossover_rate,
                     mutation_rate, top_n):
    """對單一 random seed 跑完 n_folds 個 GA fold。可被 pickle（供平行化）。"""
    results, total_evals = [], 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_results, n_evals = run_ga_single_fold(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            X_train.iloc[val_idx], y_train.iloc[val_idx],
            rs, fold_idx, max_k, n_generations, population_size,
            inner_folds, crossover_rate, mutation_rate, top_n)
        results.extend(fold_results)
        total_evals += n_evals
    return results, total_evals


def run_ga_on_fold(X_train, y_train, max_k=None, n_generations=None,
                   population_size=None, n_rs=None, n_folds=None,
                   inner_folds=None, crossover_rate=None, mutation_rate=None,
                   top_n=None, n_workers=1, verbose=False):
    """跨 n_rs 個 random seed × n_folds 個 GA fold 累計特徵頻率。

    n_workers <= 1 時完全循序執行 —— 在 outer-fold 層已經開了 process pool 的
    情境下，巢狀 pool 只會增加 spawn/pickle 成本而無平行效益。

    Returns
    -------
    (feature_frequency_df, n_evals)
    """
    max_k = MAX_K if max_k is None else max_k
    n_generations = N_GENERATIONS if n_generations is None else n_generations
    population_size = POP_SIZE if population_size is None else population_size
    n_rs = GA_N_RS if n_rs is None else n_rs
    n_folds = GA_N_FOLDS if n_folds is None else n_folds
    inner_folds = GA_INNER_FOLDS if inner_folds is None else inner_folds
    crossover_rate = CROSSOVER_RATE if crossover_rate is None else crossover_rate
    mutation_rate = MUTATION_RATE if mutation_rate is None else mutation_rate
    top_n = TOP_N_FREQUENCY if top_n is None else top_n

    rs_list = list(range(n_rs))
    frequency_counter, raw_count_counter = {}, {}
    total_evals = 0

    def accumulate(results):
        # 每個 top-N 個體投一票，權重 = max(0, (held_out_auc - 0.5) * 2)
        for selected_features, held_out_auc in results:
            weight = max(0.0, (held_out_auc - 0.5) * 2.0)
            for feat in selected_features:
                frequency_counter[feat] = frequency_counter.get(feat, 0.0) + weight
                raw_count_counter[feat] = raw_count_counter.get(feat, 0) + 1

    args = (n_folds, max_k, n_generations, population_size, inner_folds,
            crossover_rate, mutation_rate, top_n)

    if n_workers <= 1:
        for rs in rs_list:
            try:
                results, n_evals = run_ga_single_rs(X_train, y_train, rs, *args)
                accumulate(results)
                total_evals += n_evals
            except Exception as e:
                print(f'  WARNING: RS {rs} failed: {type(e).__name__}: {e}')
    else:
        ctx = mp.get_context(MP_START_METHOD)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = {ex.submit(run_ga_single_rs, X_train, y_train, rs, *args): rs
                       for rs in rs_list}
            for fut in as_completed(futures):
                rs = futures[fut]
                try:
                    results, n_evals = fut.result()
                    accumulate(results)
                    total_evals += n_evals
                except Exception as e:
                    print(f'  WARNING: RS {rs} failed: {type(e).__name__}: {e}')

    freq_df = build_feature_frequency_df(frequency_counter, raw_count_counter)

    if verbose and not freq_df.empty:
        print('\n' + '=' * 60)
        print('GA Frequency Engine — Top-10 features')
        print('=' * 60)
        for _, row in freq_df.head(10).iterrows():
            print(f"  {int(row['rank']):2d}. {row['feature']:<34s} "
                  f"score = {row['frequency']:.4f}  (raw_count = {row['raw_count']})")
        print('=' * 60)

    return freq_df, total_evals


# =============================================================================
# 7. Holdout 評估（RBF-SVM）
# =============================================================================

def select_top_k_features(feature_frequency_df: pd.DataFrame, k: int) -> list:
    return feature_frequency_df.sort_values('rank').head(k)['feature'].tolist()


def evaluate_holdout_k_grid(X_train, y_train, X_test, y_test,
                            feature_frequency_df, k_grid=None,
                            scaler_type=None, case_ids=None) -> list:
    """對每個 k 取 Top-K 特徵，重新 fit scaler + RBF-SVM，回傳 test 的預測分數。

    LOO 下每個 test fold 只有 1 筆，per-fold AUC 無意義，因此這裡只輸出
    (y_true, y_score)，AUC 留到 compute_pooled_loo_auc() 彙總後再算。
    """
    if k_grid is None:
        k_grid = K_GRID
    if scaler_type is None:
        scaler_type = SCALER_TYPE_OUTER

    y_true = np.asarray(y_test)
    ids = ([str(c) for c in case_ids] if case_ids is not None
           else [str(i) for i in X_test.index.tolist()])

    results = []
    for k in k_grid:
        selected = select_top_k_features(feature_frequency_df, k)
        scaler = make_scaler(scaler_type)
        X_tr = scaler.fit_transform(X_train[selected])
        X_te = scaler.transform(X_test[selected])

        clf = make_svm()
        clf.fit(X_tr, np.asarray(y_train))
        y_score = clf.predict_proba(X_te)[:, 1]

        results.append({'k': k, 'selected_features': selected,
                        'y_true': y_true, 'y_score': y_score, 'case_id': ids})
    return results


# =============================================================================
# 8. Predictions I/O、checkpoint、彙總
# =============================================================================

def load_predictions_df() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    try:
        return pd.read_csv(PREDICTIONS_FILE)
    except Exception as e:
        print(f'[WARNING] 無法讀取 {PREDICTIONS_FILE}: {e}，從空結果開始。')
        return pd.DataFrame(columns=PREDICTION_COLUMNS)


def append_predictions(rows) -> None:
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PREDICTIONS_FILE, mode='a',
                              header=not PREDICTIONS_FILE.exists(), index=False)


def is_fold_done(predictions_df, bw, outer_rs, fold_idx, k_grid=None) -> bool:
    """該 LOO fold 的所有 k 是否都已算過。"""
    if k_grid is None:
        k_grid = K_GRID
    if predictions_df.empty:
        return False
    sub = predictions_df[(predictions_df['bw'] == bw)
                         & (predictions_df['outer_rs'].astype(str) == str(outer_rs))
                         & (predictions_df['fold_idx'] == fold_idx)
                         & (predictions_df['classifier'] == 'svm')]
    if sub.empty:
        return False
    return set(k_grid).issubset(set(sub['k'].unique()))


def compute_pooled_loo_auc(predictions_df, k_grid=None) -> pd.DataFrame:
    """LOO 的正確彙總：把所有 fold 的 (y_true, y_score) 併起來算一個 AUC。"""
    if k_grid is None:
        k_grid = K_GRID
    records = []
    for bw in sorted(predictions_df['bw'].unique()):
        for k in k_grid:
            sub = predictions_df[(predictions_df['bw'] == bw)
                                 & (predictions_df['classifier'] == 'svm')
                                 & (predictions_df['k'] == k)]
            if sub.empty:
                continue
            auc = safe_auc(sub['y_true'], sub['y_score'], default=np.nan)
            records.append({'bw': bw, 'classifier': 'svm', 'k': k,
                            'auc_pooled': auc, 'n_pooled': len(sub)})
    return pd.DataFrame(records)


def bootstrap_ci(predictions_df, k=None, n_iter=1000, alpha=0.05,
                 random_state=RANDOM_SEED) -> pd.DataFrame:
    """pooled AUC 的 percentile bootstrap CI（對 case 重抽樣）。"""
    if k is None:
        if not K_GRID:
            raise ValueError('K_GRID is empty; cannot infer target k.')
        k = K_GRID[-1]

    rng = np.random.default_rng(random_state)
    records = []
    for bw in sorted(predictions_df['bw'].unique()):
        sub = predictions_df[(predictions_df['bw'] == bw)
                             & (predictions_df['classifier'] == 'svm')
                             & (predictions_df['k'] == k)]
        if sub.empty:
            continue
        y_true = sub['y_true'].values
        y_score = sub['y_score'].values
        n = len(sub)

        boot = []
        for _ in range(n_iter):
            idx = rng.integers(0, n, size=n)
            auc_b = safe_auc(y_true[idx], y_score[idx], default=np.nan)
            if not np.isnan(auc_b):     # 重抽後只剩單一類別 → 跳過該次
                boot.append(auc_b)
        if not boot:
            records.append({'bw': bw, 'k': k, 'auc_pooled': np.nan,
                            'auc_lo': np.nan, 'auc_hi': np.nan,
                            'n_iter': n_iter, 'n_pooled': n})
            continue

        boot = np.asarray(boot)
        records.append({
            'bw': bw, 'k': k,
            'auc_pooled': safe_auc(y_true, y_score, default=np.nan),
            'auc_lo': float(np.percentile(boot, 100 * alpha / 2)),
            'auc_hi': float(np.percentile(boot, 100 * (1 - alpha / 2))),
            'n_iter': n_iter, 'n_pooled': n,
        })
    return pd.DataFrame(records)


# =============================================================================
# 9. 單一 LOO fold 與主流程
# =============================================================================

def compute_single_fold(bw, outer_rs, fold_idx, X_train, y_train, X_test, y_test,
                        test_case_ids, max_k, n_generations, population_size,
                        n_rs, n_folds, inner_folds, k_grid,
                        crossover_rate, mutation_rate, artifacts_dir):
    """跑完一個 LOO fold（GA 選特徵 → holdout 預測），回傳要寫進 CSV 的 rows。

    純計算 + 最多寫一份 GA artifact；供 outer ProcessPoolExecutor 呼叫。
    """
    feature_frequency_df, n_evals = run_ga_on_fold(
        X_train, y_train, max_k=max_k, n_generations=n_generations,
        population_size=population_size, n_rs=n_rs, n_folds=n_folds,
        inner_folds=inner_folds, crossover_rate=crossover_rate,
        mutation_rate=mutation_rate, n_workers=1)

    if feature_frequency_df.empty:
        raise RuntimeError(f'GA 未產生任何特徵頻率 (bw={bw}, fold={fold_idx})')

    if artifacts_dir is not None:
        art_dir = Path(artifacts_dir) / f'BW_{bw}' / f'outer_rs_{outer_rs}' / f'fold_{fold_idx}'
        safe_save_csv(feature_frequency_df, art_dir / 'feature_frequencies.csv')

    ho_results = evaluate_holdout_k_grid(
        X_train, y_train, X_test, y_test, feature_frequency_df,
        k_grid=k_grid, scaler_type=SCALER_TYPE_OUTER, case_ids=test_case_ids)

    timestamp = datetime.now().isoformat()
    rows = []
    for res in ho_results:
        for i in range(len(res['y_score'])):
            rows.append({
                'bw': bw, 'outer_rs': outer_rs, 'fold_idx': fold_idx,
                'classifier': 'svm', 'k': res['k'],
                'case_id': res['case_id'][i],
                'y_true': res['y_true'][i], 'y_score': float(res['y_score'][i]),
                'timestamp': timestamp,
            })
    return rows, n_evals


def run_pipeline_loo(force_recompute=False, bin_widths=None, k_grid=None,
                     n_outer_workers=None) -> pd.DataFrame:
    """LOO 主流程：outer fold 層級平行化，逐 fold checkpoint 到 CSV。"""
    if bin_widths is None:
        bin_widths = BIN_WIDTHS
    if k_grid is None:
        k_grid = K_GRID
    if n_outer_workers is None:
        n_outer_workers = N_OUTER_WORKERS

    outer_rs = 'loo'
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config_dict(), indent=2, default=str),
                           encoding='utf-8')

    predictions_df = load_predictions_df()
    if force_recompute and PREDICTIONS_FILE.exists():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = WORKSPACE_DIR / f'pipeline_predictions_backup_{ts}.csv'
        PREDICTIONS_FILE.rename(backup)
        print(f'[force_recompute] 已備份 {PREDICTIONS_FILE} → {backup}')
        predictions_df = pd.DataFrame(columns=PREDICTION_COLUMNS)

    ctx = mp.get_context(MP_START_METHOD)
    artifacts_dir = str(ARTIFACTS_DIR) if SAVE_GA_ARTIFACTS else None

    for bw in bin_widths:
        if not bw_csv_exists(bw):
            print(f'BW {bw} 的 CSV 不存在，略過。')
            continue

        print(f'\n{"=" * 60}')
        print(f'BW={bw}  |  LOO  |  inner_folds={GA_INNER_FOLDS}  |  '
              f'{n_outer_workers} outer workers')
        print('=' * 60)

        splits = load_loo_splits(bw)
        n_total = len(splits)
        pending = [f for f in range(n_total)
                   if not is_fold_done(predictions_df, bw, outer_rs, f, k_grid)]
        print(f'  N = {n_total} 個樣本 → {n_total} 個 outer fold；'
              f'待算 {len(pending)} 個（已完成 {n_total - len(pending)} 個）')
        if not pending:
            continue

        t_bw = time.time()
        with ProcessPoolExecutor(max_workers=n_outer_workers, mp_context=ctx) as ex:
            futures = {}
            for fold_idx in pending:
                X_train, y_train, X_test, y_test, test_case_ids = splits[fold_idx]
                futures[ex.submit(
                    compute_single_fold, bw, outer_rs, fold_idx,
                    X_train, y_train, X_test, y_test, test_case_ids,
                    MAX_K, N_GENERATIONS, POP_SIZE, GA_N_RS, GA_N_FOLDS,
                    GA_INNER_FOLDS, k_grid, CROSSOVER_RATE, MUTATION_RATE,
                    artifacts_dir)] = fold_idx

            n_done = 0
            for fut in as_completed(futures):
                fold_idx = futures[fut]
                try:
                    rows, n_evals = fut.result()
                    append_predictions(rows)
                    predictions_df = pd.concat(
                        [predictions_df, pd.DataFrame(rows)], ignore_index=True)
                    n_done += 1
                    elapsed = time.time() - t_bw
                    eta = elapsed / n_done * (len(pending) - n_done)
                    print(f'  [{n_done}/{len(pending)}] fold {fold_idx} 完成  '
                          f'({n_evals} 次 fitness 評估, 已耗時 {elapsed / 60:.1f} 分, '
                          f'ETA {eta / 60:.1f} 分)')
                except Exception as e:
                    print(f'  WARNING: fold {fold_idx} 失敗: {type(e).__name__}: {e}')

    return predictions_df


def report(predictions_df, k_grid=None, n_iter=1000, save=True) -> None:
    """印出 pooled LOO AUC 與 bootstrap CI，並存檔。"""
    if k_grid is None:
        k_grid = K_GRID
    if predictions_df.empty:
        print('沒有任何 predictions。')
        return

    pooled = compute_pooled_loo_auc(predictions_df, k_grid=k_grid)
    print('\nPooled LOO AUC:')
    print(pooled.sort_values(['bw', 'k']).to_string(index=False))

    ci_df = bootstrap_ci(predictions_df, n_iter=n_iter)
    print(f'\nBootstrap CI (SVM @ k={k_grid[-1]}, n_iter={n_iter}):')
    print(ci_df.to_string(index=False))

    if save:
        CI_FILE.parent.mkdir(parents=True, exist_ok=True)
        ci_df.to_csv(CI_FILE, index=False)
        pooled.to_csv(WORKSPACE_DIR / 'pipeline_pooled_auc.csv', index=False)
        print(f'\n→ {CI_FILE}')
        print(f'→ {WORKSPACE_DIR / "pipeline_pooled_auc.csv"}')


# =============================================================================
# 10. 合成資料驗證 / benchmark
# =============================================================================

def _make_synthetic_workspace(n_samples=24, n_features=20, n_informative=5,
                              random_state=0):
    """建立合成資料 CSV 與暫時 workspace，回傳 (tmp_dir, restore_fn)。"""
    global DATA_DIR, CSV_TEMPLATE, WORKSPACE_DIR, ARTIFACTS_DIR
    global PREDICTIONS_FILE, CI_FILE, CONFIG_FILE, ORIGINAL_ONLY

    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, random_state=random_state,
                               class_sep=1.5)
    df = pd.DataFrame(X, columns=[f'original_syn_{i}' for i in range(n_features)])
    df['Label'] = y
    df['CaseNumber'] = range(n_samples)

    tmp_dir = Path(tempfile.mkdtemp(prefix='loo_synth_'))
    df.to_csv(tmp_dir / 'Cine_output_20260606_binWidth_999.csv', index=False)

    saved = dict(DATA_DIR=DATA_DIR, WORKSPACE_DIR=WORKSPACE_DIR,
                 ARTIFACTS_DIR=ARTIFACTS_DIR, PREDICTIONS_FILE=PREDICTIONS_FILE,
                 CI_FILE=CI_FILE, CONFIG_FILE=CONFIG_FILE,
                 ORIGINAL_ONLY=ORIGINAL_ONLY)

    DATA_DIR = tmp_dir
    WORKSPACE_DIR = tmp_dir / 'workspace'
    ARTIFACTS_DIR = WORKSPACE_DIR / 'artifacts'
    PREDICTIONS_FILE = WORKSPACE_DIR / 'pipeline_predictions.csv'
    CI_FILE = WORKSPACE_DIR / 'ci.csv'
    CONFIG_FILE = WORKSPACE_DIR / 'run_config.json'

    def restore():
        g = globals()
        g.update(saved)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return tmp_dir, restore


def _pick_balanced_folds(splits, n_per_class=2) -> list:
    """挑出每個類別各 n_per_class 個 LOO fold（test 樣本的類別），
    讓合成測試的 pooled AUC 一定含兩個類別。"""
    by_class = {}
    for fold_idx, (_, _, _, y_test, _) in splits.items():
        label = int(np.asarray(y_test)[0])
        by_class.setdefault(label, []).append(fold_idx)
    picked = []
    for label in sorted(by_class):
        picked.extend(by_class[label][:n_per_class])
    return sorted(picked)


def _run_synth_folds(fold_indices, splits, pop, gen, n_rs, n_folds, inner_folds):
    rows, evals, t0 = [], 0, time.time()
    for fold_idx in fold_indices:
        X_train, y_train, X_test, y_test, test_case_ids = splits[fold_idx]
        fold_rows, n_evals = compute_single_fold(
            999, 'loo', fold_idx, X_train, y_train, X_test, y_test, test_case_ids,
            MAX_K, gen, pop, n_rs, n_folds, inner_folds, K_GRID,
            CROSSOVER_RATE, MUTATION_RATE, None)
        rows.extend(fold_rows)
        evals += n_evals
    return rows, evals, time.time() - t0


def synthetic_validation() -> bool:
    """小參數合成資料：驗證流程、row 數、pooled AUC、bootstrap CI、checkpoint。"""
    print('=' * 70)
    print('SYNTHETIC VALIDATION')
    print('=' * 70)

    tmp_dir, restore = _make_synthetic_workspace()
    try:
        splits = load_loo_splits(999)
        print(f'  合成資料 N = {len(splits)} → {len(splits)} 個 LOO fold')

        # 挑選兩類各 2 個 fold，確保 pooled AUC 可計算（單一類別會是 NaN）
        fold_indices = _pick_balanced_folds(splits, n_per_class=2)
        rows, evals, elapsed = _run_synth_folds(
            fold_indices, splits, pop=10, gen=10, n_rs=2, n_folds=2,
            inner_folds=GA_INNER_FOLDS)
        test_df = pd.DataFrame(rows)

        expected = len(fold_indices) * len(K_GRID)
        assert len(test_df) == expected, f'row 數 {len(test_df)} != {expected}'
        print(f'  ✅ row 數正確: {len(test_df)}（{evals} 次 fitness 評估, {elapsed:.1f}s）')

        assert set(test_df['classifier'].unique()) == {'svm'}
        print('  ✅ 只有 svm 一種 classifier')

        pooled = compute_pooled_loo_auc(test_df, k_grid=K_GRID)
        assert pooled['auc_pooled'].notna().all(), 'pooled AUC 有 NaN'
        assert ((pooled['auc_pooled'] >= 0) & (pooled['auc_pooled'] <= 1)).all()
        print(f'  ✅ pooled AUC 有效\n{pooled.to_string(index=False)}')

        ci_df = bootstrap_ci(test_df, n_iter=100)
        assert ci_df['auc_pooled'].notna().all(), 'bootstrap CI 有 NaN'
        assert (ci_df['auc_lo'] <= ci_df['auc_pooled']).all()
        assert (ci_df['auc_pooled'] <= ci_df['auc_hi']).all()
        print(f'  ✅ bootstrap CI 有效\n{ci_df.to_string(index=False)}')

        missing_fold = next(f for f in splits if f not in fold_indices)
        empty = pd.DataFrame(columns=PREDICTION_COLUMNS)
        assert not is_fold_done(empty, 999, 'loo', fold_indices[0], k_grid=K_GRID)
        assert is_fold_done(test_df, 999, 'loo', fold_indices[0], k_grid=K_GRID)
        assert not is_fold_done(test_df, 999, 'loo', missing_fold, k_grid=K_GRID)
        print('  ✅ checkpoint (is_fold_done) 正確')

        # inner_folds=1 也要能跑
        rows1, evals1, elapsed1 = _run_synth_folds(
            fold_indices[:1], splits, pop=10, gen=10, n_rs=2, n_folds=2,
            inner_folds=1)
        assert len(rows1) == len(K_GRID)
        print(f'  ✅ inner_folds=1 可執行（{evals1} 次評估, {elapsed1:.1f}s）')

        print('\n' + '=' * 70)
        print('SYNTHETIC VALIDATION PASSED ✅')
        print('=' * 70)
        return True
    finally:
        restore()


def benchmark_inner_folds(inner_list=(3, 1), n_per_class=2, pop=20, gen=20,
                          n_rs=3, n_folds=3) -> pd.DataFrame:
    """在合成資料上比較不同 GA_INNER_FOLDS 的耗時與 pooled AUC。

    注意：合成資料的 AUC 不能外推到真實 radiomics 資料，這裡主要看
    **時間比例** 與 **流程是否穩定**；AUC 差異僅供參考。
    """
    print('=' * 70)
    print(f'BENCHMARK: GA_INNER_FOLDS {list(inner_list)}')
    print(f'  合成資料, 每類 {n_per_class} 個 LOO fold, pop={pop}, gen={gen}, '
          f'n_rs={n_rs}, n_folds={n_folds}')
    print('=' * 70)

    tmp_dir, restore = _make_synthetic_workspace(n_samples=40, n_features=30)
    try:
        splits = load_loo_splits(999)
        fold_indices = _pick_balanced_folds(splits, n_per_class=n_per_class)
        records = []
        for inner in inner_list:
            rows, evals, elapsed = _run_synth_folds(
                fold_indices, splits, pop, gen, n_rs, n_folds, inner)
            pooled = compute_pooled_loo_auc(pd.DataFrame(rows), k_grid=K_GRID)
            auc = float(pooled['auc_pooled'].iloc[-1]) if not pooled.empty else np.nan
            records.append({'inner_folds': inner, 'seconds': round(elapsed, 1),
                            'fitness_evals': evals,
                            'svm_fits': evals * max(1, inner),
                            'pooled_auc': round(auc, 4)})
            print(f'  inner_folds={inner}: {elapsed:.1f}s, {evals} 次評估, '
                  f'pooled AUC={auc:.4f}')

        df = pd.DataFrame(records)
        if len(df) > 1:
            base = df.iloc[0]
            df['speedup_vs_first'] = (base['seconds'] / df['seconds']).round(2)
        print('\n' + df.to_string(index=False))
        print('\n注意：合成資料的 AUC 不代表真實資料表現，請以時間比例為主。')
        return df
    finally:
        restore()


# =============================================================================
# 11. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LOOCV + GA feature selection pipeline (RBF-SVM only)')
    parser.add_argument('--mode', choices=['synth', 'bench', 'run', 'ci'],
                        default='synth',
                        help='synth: 合成資料驗證; bench: 比較 inner folds; '
                             'run: 正式 LOO; ci: 只重算 pooled AUC / CI')
    parser.add_argument('--inner-folds', type=int, default=None,
                        help=f'覆寫 GA_INNER_FOLDS（預設 {GA_INNER_FOLDS}）；'
                             '1 = 單次 stratified holdout，計算量約 1/inner')
    parser.add_argument('--val-fraction', type=float, default=None,
                        help=f'inner-folds=1 時的驗證集比例（預設 {GA_INNER_VAL_FRACTION:.3f}）')
    parser.add_argument('--bin-widths', type=int, nargs='+', default=None,
                        help='覆寫 BIN_WIDTHS')
    parser.add_argument('--k-grid', type=int, nargs='+', default=None,
                        help='覆寫 K_GRID')
    parser.add_argument('--n-rs', type=int, default=None, help='覆寫 GA_N_RS')
    parser.add_argument('--ga-folds', type=int, default=None, help='覆寫 GA_N_FOLDS')
    parser.add_argument('--pop-size', type=int, default=None, help='覆寫 POP_SIZE')
    parser.add_argument('--generations', type=int, default=None,
                        help='覆寫 N_GENERATIONS')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'覆寫 N_OUTER_WORKERS（預設 {N_OUTER_WORKERS}）')
    parser.add_argument('--workspace', type=str, default=None,
                        help='覆寫輸出目錄（預設由設定自動命名，含 _inner{N} 後綴）')
    parser.add_argument('--no-mi-init', action='store_true',
                        help='關閉 MI 種子化初始族群（純隨機初始化）')
    parser.add_argument('--no-cache', action='store_true',
                        help='關閉 fitness 快取（預設開啟，為無損加速）')
    parser.add_argument('--early-stopping', action='store_true',
                        help=f'開啟提前停止（連續 {PATIENCE} 代無改善即結束演化）')
    parser.add_argument('--force', action='store_true',
                        help='忽略既有 checkpoint，備份後重算')
    parser.add_argument('--ci-iter', type=int, default=1000,
                        help='bootstrap 次數（預設 1000）')
    args = parser.parse_args()

    g = globals()
    if args.inner_folds is not None:
        g['GA_INNER_FOLDS'] = max(1, args.inner_folds)
    if args.val_fraction is not None:
        g['GA_INNER_VAL_FRACTION'] = args.val_fraction
    if args.bin_widths is not None:
        g['BIN_WIDTHS'] = args.bin_widths
    if args.k_grid is not None:
        g['K_GRID'] = args.k_grid
    if args.n_rs is not None:
        g['GA_N_RS'] = args.n_rs
    if args.ga_folds is not None:
        g['GA_N_FOLDS'] = args.ga_folds
    if args.pop_size is not None:
        g['POP_SIZE'] = args.pop_size
    if args.generations is not None:
        g['N_GENERATIONS'] = args.generations
    if args.workers is not None:
        g['N_OUTER_WORKERS'] = args.workers
    if args.no_mi_init:
        g['USE_MI_INITIALIZATION'] = False
    if args.no_cache:
        g['USE_FITNESS_CACHE'] = False
    if args.early_stopping:
        g['USE_EARLY_STOPPING'] = True

    apply_config(args.workspace)
    print_config()

    if args.mode == 'synth':
        synthetic_validation()

    elif args.mode == 'bench':
        benchmark_inner_folds()

    elif args.mode == 'run':
        t0 = time.time()
        predictions_df = run_pipeline_loo(force_recompute=args.force)
        print(f'\n總耗時: {(time.time() - t0) / 60:.1f} 分鐘')
        print(f'總 prediction 筆數: {len(predictions_df)}')
        report(predictions_df, n_iter=args.ci_iter)

    elif args.mode == 'ci':
        predictions_df = load_predictions_df()
        if predictions_df.empty:
            print('找不到 predictions，請先執行 --mode run。')
            sys.exit(1)
        report(predictions_df, n_iter=args.ci_iter)


apply_config()

if __name__ == '__main__':
    main()
