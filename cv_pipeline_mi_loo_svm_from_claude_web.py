#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cv_pipeline_mi_loo_svm.py
=========================
Mutual Information (MI) 特徵選擇 + Leave-One-Out CV 的完整單檔 pipeline。
由 cv_pipeline_mutual_info.ipynb + cv_pipeline_MI_LOO.py 合併、精簡而成
（結構與 GA 版 cv_pipeline_loo_svm.py 對齊，predictions CSV 格式相同，可直接比較）。

與前版的差異
------------
1. **單一檔案**：不再 exec-load notebook，沒有 `_ns` / globals 鏡射那一層。
2. **全部統一 RBF-SVM**：最終 holdout 評估只用 SVC(kernel='rbf')。
   已移除 rf / xgboost / softvote-3 與 xgboost 依賴。
   （MI 特徵選擇本身不需要分類器。）
3. **移除冗餘程式**：K-fold OuterCVGenerator、K-fold run_single_fold / run_pipeline、
   pipeline_results.csv 相關 I/O、softvote、multi-RS 變異分解 / summary、
   run_mi_simple alias、未使用的 Counter / RandomForest / xgboost import。
4. **數值與原版逐位元一致**：已用等價測試比對原版 run_mi_on_fold 與 SVM holdout 分數。

MI 特徵排名方式（與原版相同）
------------------------------
每個 LOO outer fold 的訓練集（N-1 筆）上：

    for rs in range(MI_N_RS):                         
        StratifiedKFold(MI_N_FOLDS, random_state=rs)  
            對每一折的 4/5 train：
              VarianceThreshold(0) + Pearson(>PEARSON_THRESHOLD) 過濾（只 fit 在 4/5）
              SCALER_TYPE_INNER fit_transform（只 fit 在 4/5）
              mutual_info_classif(random_state=rs)
              → 每個通過過濾的特徵累加其 raw MI 分數

frequency = 20 次（4 × 5）raw MI 分數總和；raw_count = 該特徵通過 prefilter 的次數。
每一折的 1/5 驗證集在 MI 版本中**不會被使用**——實際上等同於對 40 個 80% 子樣本
做 MI 並加總（類似 stability selection），這是原版的設計，此處保持不變。

接著依 frequency 排名取 Top-K（K_GRID），在完整 N-1 訓練集上 fit
SCALER_TYPE_OUTER + RBF-SVM，預測被留下的那 1 筆。

用法
----
  # 合成資料快速驗證（<1 分鐘，含整體時間估算）
  python cv_pipeline_mi_loo_svm.py --mode synth

  # 正式 LOO 跑
  python cv_pipeline_mi_loo_svm.py --mode run

  # 只從既有 predictions 重算 pooled AUC / bootstrap CI
  python cv_pipeline_mi_loo_svm.py --mode ci

預設 workspace = WORKSPACE_ROOT / 'mi_loo_r{PEARSON_THRESHOLD}_{SCALER_TYPE_OUTER}{RUN_NAME_SUFFIX}'
（例：mi_loo_r0.7_PowerTransformer_inner1）。MI 沒有 inner folds，後綴只是命名，
方便與 GA inner1 的結果對照，不影響計算。
若 workspace 內已有 run_config.json 且關鍵設定不同，程式會拒絕寫入，
避免不同設定的結果混在同一個 CSV（可用 --workspace 換目錄，或 --force 備份後重算）。
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
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
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
SCALER_CHOICES = ('StandardScaler', 'RobustScaler',
                  'QuantileTransformer', 'PowerTransformer')
SCALER_TYPE_INNER = 'RobustScaler'   # MI 計算前（每個 4/5 子樣本各 fit 一次）
SCALER_TYPE_OUTER = 'RobustScaler'   # 最終 holdout（每個 LOO fold fit 一次）

# ── 分類器（全 pipeline 唯一模型：RBF-SVM）──────────────────────────────────
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
RANDOM_SEED = 42

# ── MI 特徵排名 ──────────────────────────────────────────────────────────────
# MI_N_RS 個 random seed × MI_N_FOLDS 折 = 每個 outer fold 的累加次數（對齊GA）
MI_N_RS = 4
MI_N_FOLDS = 5

# ── Holdout 評估 ─────────────────────────────────────────────────────────────
K_GRID = [4]                  # 由 MI 排名取 Top-K 特徵；N≈80 時建議 <= 4

# ── 輸出 ─────────────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path('/home/ser/pipeline/workspace')
SAVE_ARTIFACTS = True         # 每個 fold 存一份 feature_frequencies.csv
RUN_NAME_SUFFIX = '_inner1'   # 附加在 workspace 目錄名後（純命名用）

# ── 平行化 ───────────────────────────────────────────────────────────────────
N_OUTER_WORKERS = 8           # outer LOO fold 層級的 process 數
MP_START_METHOD = 'fork'      # Linux / WSL2

# ── 衍生路徑（由 apply_config() 計算，CLI 覆寫後會重算）─────────────────────
RUN_NAME = None
WORKSPACE_DIR = None
ARTIFACTS_DIR = None
PREDICTIONS_FILE = None
CI_FILE = None
POOLED_FILE = None
CONFIG_FILE = None

PREDICTION_COLUMNS = ['bw', 'outer_rs', 'fold_idx', 'classifier', 'k',
                      'case_id', 'y_true', 'y_score', 'timestamp']

# 會影響 predictions 數值的設定；workspace 內既有 run_config.json 若與此不同，拒絕續跑
RESULT_AFFECTING_KEYS = [
    'ORIGINAL_ONLY', 'PEARSON_THRESHOLD', 'SCALER_TYPE_INNER', 'SCALER_TYPE_OUTER',
    'SVM_KERNEL', 'SVM_C', 'SVM_GAMMA', 'RANDOM_SEED', 'MI_N_RS', 'MI_N_FOLDS',
]

CONFIG_KEYS = RESULT_AFFECTING_KEYS + ['BIN_WIDTHS', 'K_GRID', 'N_OUTER_WORKERS']


def build_run_name() -> str:
    """例：mi_loo_r0.7_PowerTransformer_inner1。"""
    return f'mi_loo_r{PEARSON_THRESHOLD}_{SCALER_TYPE_OUTER}{RUN_NAME_SUFFIX}'


def apply_config(workspace_dir=None) -> None:
    """重算所有衍生路徑。CLI 覆寫常數之後必須呼叫一次。"""
    global RUN_NAME, WORKSPACE_DIR, ARTIFACTS_DIR
    global PREDICTIONS_FILE, CI_FILE, POOLED_FILE, CONFIG_FILE

    RUN_NAME = build_run_name()
    WORKSPACE_DIR = Path(workspace_dir) if workspace_dir else WORKSPACE_ROOT / RUN_NAME
    ARTIFACTS_DIR = WORKSPACE_DIR / 'artifacts'
    PREDICTIONS_FILE = WORKSPACE_DIR / 'pipeline_predictions.csv'
    CI_FILE = WORKSPACE_DIR / 'pipeline_svm_k{}_ci.csv'.format(K_GRID[-1] if K_GRID else 'NA')
    POOLED_FILE = WORKSPACE_DIR / 'pipeline_pooled_auc.csv'
    CONFIG_FILE = WORKSPACE_DIR / 'run_config.json'


def config_dict() -> dict:
    g = globals()
    cfg = {k: g[k] for k in CONFIG_KEYS}
    cfg['RUN_NAME'] = RUN_NAME
    cfg['WORKSPACE_DIR'] = str(WORKSPACE_DIR)
    return cfg


def print_config() -> None:
    print('=' * 70)
    print('Configuration (MI + LOOCV + RBF-SVM)')
    print('=' * 70)
    for k, v in config_dict().items():
        print(f'  {k:<22s}: {v}')
    print(f'  → 每個 outer fold 的 MI 累加次數: {MI_N_RS} × {MI_N_FOLDS} = {MI_N_RS * MI_N_FOLDS}')
    print('=' * 70)


def check_workspace_config(force: bool) -> None:
    """workspace 內已有 run_config.json 時，確認關鍵設定一致，避免結果混雜。"""
    if force or not CONFIG_FILE.exists():
        return
    try:
        saved = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
    except Exception:
        return
    current = config_dict()
    diffs = [(k, saved.get(k), current[k]) for k in RESULT_AFFECTING_KEYS
             if k in saved and saved.get(k) != current[k]]
    if diffs:
        print('\n[ERROR] 此 workspace 先前以不同設定執行過，繼續會把不同設定的結果混在一起：')
        for k, old, new in diffs:
            print(f'    {k}: 既有 = {old!r}  目前 = {new!r}')
        print(f'  workspace: {WORKSPACE_DIR}')
        print('  請改用 --workspace 指定新目錄，或加 --force 備份既有 predictions 後重算。')
        sys.exit(2)


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
# 4. MI 特徵排名引擎
# =============================================================================

def compute_mi_scores(X, y, random_state) -> np.ndarray:
    """mutual_info_classif；失敗時退回全 1（與原版相同）並警告。"""
    try:
        return np.asarray(mutual_info_classif(X, y, random_state=random_state),
                          dtype=float)
    except Exception as e:
        print(f'  [WARNING] MI computation failed (rs={random_state}): '
              f'{type(e).__name__}: {e}')
        return np.ones(X.shape[1], dtype=float)


def run_mi_single_rs(X_train, y_train, rs, n_folds) -> list:
    """單一 random seed：n_folds 個 4/5 子樣本各自 prefilter → scale → MI。

    Returns
    -------
    list[(feature_name, raw_mi_score)]，依 fold 順序串接
    """
    pairs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    for train_idx, _unused_val_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]

        selected = fit_train_only_prefilter(X_tr)
        if len(selected) < 2:
            selected = list(X_tr.columns[:2])
        X_filtered = X_tr[selected]

        scaler = make_scaler(SCALER_TYPE_INNER)
        X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered),
                                columns=X_filtered.columns, index=X_filtered.index)

        # raw MI（不做 per-fold 正規化）：沒有鑑別力的子樣本貢獻 ≈ 0，而非 1/N
        mi = compute_mi_scores(X_scaled, y_tr, random_state=rs)
        pairs.extend(zip(selected, mi.tolist()))
    return pairs


def build_feature_frequency_df(frequency_counter, raw_count_counter=None) -> pd.DataFrame:
    """frequency = raw MI 分數總和；raw_count = 通過 prefilter 的次數。"""
    items = sorted(frequency_counter.items(), key=lambda x: (-x[1], x[0]))
    df = pd.DataFrame([{'feature': f, 'frequency': v, 'rank': i}
                       for i, (f, v) in enumerate(items, 1)])
    if raw_count_counter is not None and not df.empty:
        df['raw_count'] = df['feature'].map(raw_count_counter).fillna(0).astype(int)
    return df


def run_mi_on_fold(X_train, y_train, n_rs=None, n_folds=None, verbose=False) -> pd.DataFrame:
    """跨 n_rs 個 random seed × n_folds 折累加 raw MI，回傳特徵排名。

    在 outer-fold 層已開 process pool 的情境下，這裡一律循序執行
    （巢狀 pool 只會增加 fork/pickle 成本，沒有平行效益）。
    """
    n_rs = MI_N_RS if n_rs is None else n_rs
    n_folds = MI_N_FOLDS if n_folds is None else n_folds

    frequency_counter, raw_count_counter = {}, {}
    for rs in range(n_rs):
        try:
            for feat, score in run_mi_single_rs(X_train, y_train, rs, n_folds):
                frequency_counter[feat] = frequency_counter.get(feat, 0) + score
                raw_count_counter[feat] = raw_count_counter.get(feat, 0) + 1
        except Exception as e:
            print(f'  WARNING: RS {rs} failed: {type(e).__name__}: {e}')

    freq_df = build_feature_frequency_df(frequency_counter, raw_count_counter)

    if verbose and not freq_df.empty:
        print('\n' + '=' * 60)
        print(f'MI Feature Ranking — {n_rs} RS × {n_folds} folds '
              f'= {n_rs * n_folds} accumulations')
        print('=' * 60)
        for _, row in freq_df.head(10).iterrows():
            print(f"  {int(row['rank']):2d}. {row['feature']:<36s} "
                  f"score = {row['frequency']:.4f}  (raw_count = {row['raw_count']})")
        print('=' * 60)

    return freq_df


# =============================================================================
# 5. Holdout 評估（RBF-SVM）
# =============================================================================

def select_top_k_features(feature_frequency_df: pd.DataFrame, k: int) -> list:
    return feature_frequency_df.sort_values('rank').head(k)['feature'].tolist()


def evaluate_holdout_k_grid(X_train, y_train, X_test, y_test,
                            feature_frequency_df, k_grid=None,
                            scaler_type=None, case_ids=None) -> list:
    """對每個 k 取 Top-K 特徵，重新 fit scaler + RBF-SVM，回傳 test 的預測分數。

    LOO 下每個 test fold 只有 1 筆，per-fold AUC 無意義，因此只輸出
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
# 6. Predictions I/O、checkpoint、彙總
# =============================================================================

def coerce_prediction_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """統一 predictions 欄位型別。

    在空 DataFrame 上 pd.concat 新 rows 會讓 y_true 變成 object dtype，
    sklearn 會把 object 陣列判為 'unknown' target → roc_auc_score 失敗 → AUC=NaN。
    （從 CSV 讀回則沒問題，所以只會在 --mode run 結束時的報表出現。）
    """
    if df.empty:
        return df
    df = df.copy()
    for col in ('bw', 'fold_idx', 'k', 'y_true'):
        df[col] = pd.to_numeric(df[col])
    df['y_score'] = pd.to_numeric(df['y_score']).astype(float)
    df['outer_rs'] = df['outer_rs'].astype(str)
    return df


def load_predictions_df() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    try:
        return coerce_prediction_dtypes(pd.read_csv(PREDICTIONS_FILE))
    except Exception as e:
        print(f'[WARNING] 無法讀取 {PREDICTIONS_FILE}: {e}，從空結果開始。')
        return pd.DataFrame(columns=PREDICTION_COLUMNS)


def append_predictions(rows) -> None:
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PREDICTIONS_FILE, mode='a',
                              header=not PREDICTIONS_FILE.exists(), index=False)


def is_fold_done(predictions_df, bw, outer_rs, fold_idx, k_grid=None) -> bool:
    """該 LOO fold 的所有 k 是否都已有 svm 預測。"""
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
    predictions_df = coerce_prediction_dtypes(predictions_df)
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

    predictions_df = coerce_prediction_dtypes(predictions_df)
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
# 7. 單一 LOO fold 與主流程
# =============================================================================

def compute_single_fold(bw, outer_rs, fold_idx, X_train, y_train, X_test, y_test,
                        test_case_ids, n_rs, n_folds, k_grid, artifacts_dir):
    """跑完一個 LOO fold（MI 排名 → holdout 預測），回傳要寫進 CSV 的 rows。"""
    feature_frequency_df = run_mi_on_fold(X_train, y_train, n_rs=n_rs, n_folds=n_folds)
    if feature_frequency_df.empty:
        raise RuntimeError(f'MI 未產生任何特徵排名 (bw={bw}, fold={fold_idx})')

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
    return rows


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
    check_workspace_config(force=force_recompute)

    predictions_df = load_predictions_df()
    if force_recompute and PREDICTIONS_FILE.exists():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = WORKSPACE_DIR / f'pipeline_predictions_backup_{ts}.csv'
        PREDICTIONS_FILE.rename(backup)
        print(f'[force_recompute] 已備份 {PREDICTIONS_FILE} → {backup}')
        predictions_df = pd.DataFrame(columns=PREDICTION_COLUMNS)

    CONFIG_FILE.write_text(json.dumps(config_dict(), indent=2, default=str),
                           encoding='utf-8')

    ctx = mp.get_context(MP_START_METHOD)
    artifacts_dir = str(ARTIFACTS_DIR) if SAVE_ARTIFACTS else None

    for bw in bin_widths:
        if not bw_csv_exists(bw):
            print(f'BW {bw} 的 CSV 不存在，略過。')
            continue

        print(f'\n{"=" * 60}')
        print(f'BW={bw}  |  MI + LOO  |  {n_outer_workers} outer workers')
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
                    MI_N_RS, MI_N_FOLDS, k_grid, artifacts_dir)] = fold_idx

            n_done = 0
            for fut in as_completed(futures):
                fold_idx = futures[fut]
                try:
                    rows = fut.result()
                    append_predictions(rows)
                    predictions_df = pd.concat(
                        [predictions_df, pd.DataFrame(rows)], ignore_index=True)
                    n_done += 1
                    elapsed = time.time() - t_bw
                    eta = elapsed / n_done * (len(pending) - n_done)
                    print(f'  [{n_done}/{len(pending)}] fold {fold_idx} 完成  '
                          f'(已耗時 {elapsed / 60:.1f} 分, ETA {eta / 60:.1f} 分)')
                except Exception as e:
                    print(f'  WARNING: fold {fold_idx} 失敗: {type(e).__name__}: {e}')

    return coerce_prediction_dtypes(predictions_df)


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
        pooled.to_csv(POOLED_FILE, index=False)
        print(f'\n→ {CI_FILE}')
        print(f'→ {POOLED_FILE}')


# =============================================================================
# 8. 合成資料驗證
# =============================================================================

def _pick_balanced_folds(splits, n_per_class=2) -> list:
    """挑出每個類別各 n_per_class 個 LOO fold，確保 pooled AUC 可計算。"""
    by_class = {}
    for fold_idx, (_, _, _, y_test, _) in splits.items():
        by_class.setdefault(int(np.asarray(y_test)[0]), []).append(fold_idx)
    picked = []
    for label in sorted(by_class):
        picked.extend(by_class[label][:n_per_class])
    return sorted(picked)


def synthetic_validation() -> bool:
    """合成資料：驗證流程、row 數、pooled AUC、bootstrap CI、checkpoint，並估算總耗時。"""
    global DATA_DIR, ORIGINAL_ONLY

    print('=' * 70)
    print('SYNTHETIC VALIDATION (MI + LOO + RBF-SVM)')
    print('=' * 70)

    n_samples, n_features = 24, 20
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=5, random_state=0, class_sep=1.5)
    df = pd.DataFrame(X, columns=[f'original_syn_{i}' for i in range(n_features)])
    df['Label'] = y
    df['CaseNumber'] = range(n_samples)

    tmp_dir = Path(tempfile.mkdtemp(prefix='mi_loo_synth_'))
    df.to_csv(tmp_dir / CSV_TEMPLATE.format(bw=999), index=False)
    saved_data_dir, saved_original_only = DATA_DIR, ORIGINAL_ONLY
    DATA_DIR = tmp_dir

    try:
        splits = load_loo_splits(999)
        print(f'  合成資料 N = {len(splits)} → {len(splits)} 個 LOO fold')
        print(f'  MI 設定: {MI_N_RS} RS × {MI_N_FOLDS} folds')

        fold_indices = _pick_balanced_folds(splits, n_per_class=2)
        all_rows, per_fold_t = [], []
        for fold_idx in fold_indices:
            X_train, y_train, X_test, y_test, test_case_ids = splits[fold_idx]
            t0 = time.time()
            all_rows.extend(compute_single_fold(
                999, 'loo', fold_idx, X_train, y_train, X_test, y_test,
                test_case_ids, MI_N_RS, MI_N_FOLDS, K_GRID, None))
            per_fold_t.append(time.time() - t0)
        test_df = pd.DataFrame(all_rows)

        expected = len(fold_indices) * len(K_GRID)
        assert len(test_df) == expected, f'row 數 {len(test_df)} != {expected}'
        assert set(test_df['classifier'].unique()) == {'svm'}
        assert list(test_df.columns) == PREDICTION_COLUMNS
        print(f'  ✅ row 數 / 欄位正確: {len(test_df)} rows，只有 svm')

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

        t_fold = float(np.mean(per_fold_t))
        est_min = t_fold * 80 / max(1, N_OUTER_WORKERS) * len(BIN_WIDTHS) / 60
        print(f'\n  [時間估算] 合成資料每 fold {t_fold:.2f}s；'
              f'N=80、{N_OUTER_WORKERS} workers、{len(BIN_WIDTHS)} 個 BW '
              f'≈ {est_min:.1f} 分鐘（真實資料特徵數較多，實際會更久）')

        print('\n' + '=' * 70)
        print('SYNTHETIC VALIDATION PASSED ✅')
        print('=' * 70)
        return True
    finally:
        DATA_DIR, ORIGINAL_ONLY = saved_data_dir, saved_original_only
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# 9. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LOOCV + Mutual-Information feature ranking pipeline (RBF-SVM only)')
    parser.add_argument('--mode', choices=['synth', 'run', 'ci'], default='synth',
                        help='synth: 合成資料驗證; run: 正式 LOO; ci: 只重算 pooled AUC / CI')
    parser.add_argument('--bin-widths', type=int, nargs='+', default=None,
                        help='覆寫 BIN_WIDTHS')
    parser.add_argument('--k-grid', type=int, nargs='+', default=None,
                        help='覆寫 K_GRID')
    parser.add_argument('--n-rs', type=int, default=None,
                        help=f'覆寫 MI_N_RS（預設 {MI_N_RS}）')
    parser.add_argument('--mi-folds', type=int, default=None,
                        help=f'覆寫 MI_N_FOLDS（預設 {MI_N_FOLDS}）')
    parser.add_argument('--scaler-inner', choices=SCALER_CHOICES, default=None,
                        help=f'覆寫 SCALER_TYPE_INNER（預設 {SCALER_TYPE_INNER}）')
    parser.add_argument('--scaler-outer', choices=SCALER_CHOICES, default=None,
                        help=f'覆寫 SCALER_TYPE_OUTER（預設 {SCALER_TYPE_OUTER}）')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'覆寫 N_OUTER_WORKERS（預設 {N_OUTER_WORKERS}）')
    parser.add_argument('--workspace', type=str, default=None,
                        help='覆寫輸出目錄（預設 WORKSPACE_ROOT/mi_loo_r{r}_{scaler}_inner1）')
    parser.add_argument('--force', action='store_true',
                        help='忽略既有 checkpoint / 設定檢查，備份 predictions 後重算')
    parser.add_argument('--ci-iter', type=int, default=1000,
                        help='bootstrap 次數（預設 1000）')
    args = parser.parse_args()

    g = globals()
    if args.bin_widths is not None:
        g['BIN_WIDTHS'] = args.bin_widths
    if args.k_grid is not None:
        g['K_GRID'] = args.k_grid
    if args.n_rs is not None:
        g['MI_N_RS'] = args.n_rs
    if args.mi_folds is not None:
        g['MI_N_FOLDS'] = args.mi_folds
    if args.scaler_inner is not None:
        g['SCALER_TYPE_INNER'] = args.scaler_inner
    if args.scaler_outer is not None:
        g['SCALER_TYPE_OUTER'] = args.scaler_outer
    if args.workers is not None:
        g['N_OUTER_WORKERS'] = args.workers

    apply_config(args.workspace)
    print_config()

    if args.mode == 'synth':
        synthetic_validation()

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
