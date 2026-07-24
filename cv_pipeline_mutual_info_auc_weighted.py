#!/usr/bin/env python3
"""
8-fold outer CV pipeline with Mutual Information (filter-based) feature selection
using AUC-weighted frequency, matching the GA version's weighting mechanism.

Key changes from original cv_pipeline:
- Replaced GA feature selection with Mutual Information (mutual_info_classif)
- Uses fixed global K value (MAX_K) for selecting TOP K MI features
- Uses 4/5 fold for feature selection, 1/5 fold test AUC as weight (same as GA)
- Each RS uses the same fixed K value
- AUC-weighted frequency accumulation (same as GA version)
"""

# =============================================================================
# Block 1: Global Configuration
# =============================================================================

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
BIN_WIDTHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CLASSIFIER_TYPE = 'svm'  # Internal classifier for MI-based selection
CLASSIFIERS = ['svm', 'rf', 'xgboost']  # Classifiers for holdout evaluation

# -----------------------------------------------------------------------------
# Cross-Validation Configuration
# -----------------------------------------------------------------------------
N_OUTER_FOLDS = 8
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Scaling Configuration
# -----------------------------------------------------------------------------
SCALER_TYPE = 'QuantileTransformer'
SCALER_TYPE_OUTER = 'QuantileTransformer'
SCALER_TYPE_INNER = 'StandardScaler' 

# -----------------------------------------------------------------------------
# Feature Selection Configuration (Mutual Information with AUC weighting)
# -----------------------------------------------------------------------------
PEARSON_THRESHOLD = 0.7
ORIGINAL_ONLY = True

# MI Configuration
MAX_K = 20  # Global K value for selecting TOP K MI features
MI_N_RS = 16  # Number of random seeds for MI computation
MI_N_FOLDS = 5  # Number of inner folds for MI computation

# Feature Selection Grid for holdout evaluation
K_GRID = [5, 10, 15, 20, 30, 40, 50]

# -----------------------------------------------------------------------------
# Outer CV Multi-RS Configuration
# -----------------------------------------------------------------------------
OUTER_RS_LIST = [42, 123, 456]
RESULTS_FILE = None  # Will be set dynamically
SAVE_ARTIFACTS = True
ARTIFACTS_DIR = None  # Will be set dynamically

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
DATA_DIR = Path('/home/ser/pipeline/data')
WORKSPACE_DIR = Path('/home/ser/pipeline/workspace/MI_topK/topK_20_r0.7')  # Change this path as needed
DATA_SPLITS_DIR = WORKSPACE_DIR / 'data_splits'
EXPERIMENTS_DIR = WORKSPACE_DIR / f'experiments_{SCALER_TYPE_OUTER}_{SCALER_TYPE_INNER}'

# Set RESULTS_FILE and ARTIFACTS_DIR
RESULTS_FILE = WORKSPACE_DIR / f'pipeline_results_mi_auc.csv'
ARTIFACTS_DIR = WORKSPACE_DIR / 'artifacts'

# =============================================================================
# Block 1: Imports
# =============================================================================

import sys
import json
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Numerical and data science libraries
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# XGBoost import (optional - will check availability)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. xgboost classifier will be disabled.")

# Visualization (optional - may not be available in all environments)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None
    warnings.warn("Visualization libraries not available.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("All imports successful!")
print(f"XGBoost available: {XGBOOST_AVAILABLE}")

# =============================================================================
# Block 2: CSV Loader and Feature/Label Extraction Helpers
# =============================================================================

def resolve_binwidth_csv(bw: int) -> Path:
    """Looks under DATA_DIR for Cine_output_20260606_binWidth_{bw}.csv and returns the path."""
    return DATA_DIR / f"Cine_output_20260606_binWidth_{bw}.csv"

def find_case_id_column(df: pd.DataFrame) -> str:
    """Detects CaseNumber or CaseID column case-insensitively."""
    for col in df.columns:
        if col.lower() in ['casenumber', 'caseid']:
            return col
    raise ValueError("No CaseNumber or CaseID column found in DataFrame.")

def find_label_column(df: pd.DataFrame) -> str:
    """Detects Label column case-insensitively."""
    for col in df.columns:
        if col.lower() == 'label':
            return col
    raise ValueError("No Label column found in DataFrame.")

def load_feature_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Reads CSV with pandas, detects CaseNumber/CaseID and Label case-insensitively,
    and returns the original DataFrame, label Series, and feature column names.
    """
    df = pd.read_csv(csv_path)
    case_id_col = find_case_id_column(df)
    label_col = find_label_column(df)
    
    labels = df[label_col]
    feature_cols = [col for col in df.columns if col not in [case_id_col, label_col]]
    
    return df, labels, feature_cols

def filter_original_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only keep original features (no wavelet)."""
    non_feature_cols = [c for c in df.columns if c.lower() in ['label', 'casenumber', 'caseid']]
    feature_cols = [c for c in df.columns if c.lower() not in ['label', 'casenumber', 'caseid']]
    original_cols = [c for c in feature_cols
                     if 'original' in c.lower() and 'wavelet' not in c.lower()]
    return df[non_feature_cols + original_cols]

def ensure_workspace_dirs() -> tuple[Path, Path]:
    """Ensures that DATA_SPLITS_DIR and EXPERIMENTS_DIR exist and returns them."""
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_SPLITS_DIR, EXPERIMENTS_DIR

def safe_save_csv(df: pd.DataFrame, path: Path) -> None:
    """Safely saves a DataFrame to the specified path, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# =============================================================================
# Block 3: Preprocessing Foundation (Fit-Train-Only Filters & Scaler Factory)
# =============================================================================

def make_scaler(scaler_type: str = SCALER_TYPE):
    """
    Factory function to create a scaler based on the configured SCALER_TYPE.
    Returns StandardScaler or QuantileTransformer(output_distribution='normal').
    """
    if scaler_type == 'StandardScaler':
        return StandardScaler()
    elif scaler_type == 'QuantileTransformer':
        return QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

def pearson_correlation_filter(X: pd.DataFrame, threshold: float) -> list[str]:
    """
    Pearson correlation filter that keeps one feature from each highly correlated group.
    """
    feature_names = X.columns.tolist()
    if threshold >= 1.0 or X.shape[1] <= 1:
        return feature_names

    corr_matrix = np.abs(np.corrcoef(X.values, rowvar=False))
    n_features = X.shape[1]
    to_drop = set()
    for a in range(n_features):
        if a in to_drop:
            continue
        for b in range(a + 1, n_features):
            if b in to_drop:
                continue
            if corr_matrix[a, b] > threshold:
                to_drop.add(b)

    keep_features = [feature_names[i] for i in range(n_features) if i not in to_drop]
    return keep_features

def fit_train_only_prefilter(X_train: pd.DataFrame, y_train: pd.Series, threshold: float = PEARSON_THRESHOLD) -> list[str]:
    """
    Applies VarianceThreshold(0) and Pearson correlation filtering fit only on X_train.
    Returns the selected feature names, never using test data.
    """
    # 1. Apply VarianceThreshold(0) to remove constant features
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    vt_mask = vt.get_support()
    
    # Filter X_train to non-constant features
    X_train_vt = X_train.loc[:, vt_mask]
    
    # 2. Apply Pearson correlation filtering
    selected_features = pearson_correlation_filter(X_train_vt, threshold)
    
    return selected_features

# =============================================================================
# Block 3: Classifier Factory
# =============================================================================

def get_classifier(clf_type: str, random_state: int = 42, inner: bool = True):
    if clf_type == 'rf':
        return RandomForestClassifier(
            n_estimators=100 if inner else 500,
            max_depth=5 if inner else None,
            random_state=random_state, n_jobs=1
        )
    elif clf_type == 'svm':
        return SVC(C=1.0, kernel='rbf', probability=True, random_state=random_state)
    elif clf_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not available.")
        return xgb.XGBClassifier(
            n_estimators=100 if inner else 200,
            max_depth=5 if inner else 6,
            random_state=random_state, eval_metric='logloss', n_jobs=1
        )
    else:
        raise ValueError(f"Unsupported classifier type: {clf_type}")

# =============================================================================
# Block 3: Mutual Information Feature Selection Engine with AUC Weighting
# =============================================================================

def run_mi_single_rs_fold(X_train_rs, y_train_rs, rs, fold_idx, classifier_type='svm',
                         max_k=MAX_K, X_val=None, y_val=None):
    """
    Runs MI feature selection on a single fold of a single random state.
    
    Key implementation:
    - Uses MI to rank all features
    - Selects TOP K features based on MI scores
    - Uses holdout AUC as weight (same as GA version)
    - Returns (selected_features, held_out_auc) for frequency accumulation
    
    Parameters:
    -----------
    X_train_rs : pd.DataFrame
        Training data for this RS and fold
    y_train_rs : pd.Series
        Training labels for this RS and fold
    rs : int
        Random state
    fold_idx : int
        Fold index
    classifier_type : str
        Classifier type for AUC computation
    max_k : int
        Global K value for selecting TOP K features
    X_val : pd.DataFrame, optional
        Validation data for holdout AUC computation
    y_val : pd.Series, optional
        Validation labels for holdout AUC computation
        
    Returns:
    --------
    tuple
        (selected_features, held_out_auc) - selected features and their holdout AUC
    """
    rng = np.random.default_rng(rs * 100 + fold_idx)
    
    # 1. Preprocessing filters (same as GA version)
    selected_features = fit_train_only_prefilter(X_train_rs, y_train_rs, threshold=PEARSON_THRESHOLD)
    if len(selected_features) < 2:
        selected_features = list(X_train_rs.columns[:2])
    X_filtered = X_train_rs[selected_features]
    
    # 2. Fit scaler on training data only
    scaler = make_scaler(SCALER_TYPE_INNER)
    X_filtered_scaled_values = scaler.fit_transform(X_filtered)
    X_filtered_scaled = pd.DataFrame(
        X_filtered_scaled_values, 
        columns=X_filtered.columns, 
        index=X_filtered.index
    )
    
    # 3. Compute MI scores for all features
    try:
        mi_scores = mutual_info_classif(X_filtered_scaled, y_train_rs, random_state=42)
    except Exception:
        mi_scores = np.ones(len(selected_features))
    
    # 4. Select TOP K features based on MI scores
    if len(selected_features) <= max_k:
        top_k_indices = list(range(len(selected_features)))
    else:
        top_k_indices = np.argsort(mi_scores)[-max_k:]  # Get indices of top K MI features
    
    selected = [X_filtered.columns[i] for i in top_k_indices]
    
    # 5. Compute holdout AUC (same as GA version)
    held_out_auc = 0.5  # default fallback
    if X_val is not None and y_val is not None and len(y_val) > 0:
        try:
            # Transform validation data using the same scaler
            X_val_full_filtered = X_val[selected_features]
            X_val_scaled_full = scaler.transform(X_val_full_filtered)
            X_val_scaled_full_df = pd.DataFrame(
                X_val_scaled_full,
                columns=selected_features,
                index=X_val.index
            )
            
            # Select only the TOP K features
            X_val_scaled = X_val_scaled_full_df[selected].values
            
            # Train on full training data with selected features
            X_train_final = X_filtered_scaled[selected]
            y_train_final = np.asarray(y_train_rs)
            clf = get_classifier(classifier_type, random_state=rs * 100 + fold_idx, inner=True)
            clf.fit(X_train_final, y_train_final)
            
            # Predict on validation and compute AUC
            y_val_array = np.asarray(y_val)
            y_score = clf.predict_proba(X_val_scaled)[:, 1]
            held_out_auc = roc_auc_score(y_val_array, y_score)
        except Exception as e:
            print(f"    [WARNING] held-out AUC computation failed (rs={rs}, fold={fold_idx}): {type(e).__name__}: {e}")
            held_out_auc = 0.5
    
    return selected, held_out_auc

def _run_mi_single_rs(X_train, y_train, rs, n_folds, classifier_type, max_k):
    """
    Helper: runs all inner folds for a single random state.
    Returns list of (selected_features, held_out_auc) tuples.
    """
    results_with_auc = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        selected, held_out_auc = run_mi_single_rs_fold(
            X_train_fold, y_train_fold, rs, fold_idx,
            classifier_type=classifier_type, max_k=max_k,
            X_val=X_val_fold, y_val=y_val_fold
        )
        results_with_auc.append((selected, held_out_auc))
    
    return results_with_auc

def run_mi_on_fold(X_train, y_train, classifier_type='svm', max_k=MAX_K,
                   n_rs=MI_N_RS, n_folds=MI_N_FOLDS, random_seed=RANDOM_SEED):
    """
    Runs MI feature selection across multiple random states and folds.
    
    This is the MI-based replacement for run_ga_on_fold(). Key differences:
    - Uses MI to select TOP K features (instead of GA evolution)
    - Uses AUC-weighted frequency accumulation (same as GA version)
    - Each RS uses the same fixed K value
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature DataFrame.
    y_train : pd.Series
        Training label Series.
    classifier_type : str
        Classifier type for AUC computation.
    max_k : int
        Global K value for selecting TOP K MI features.
    n_rs : int
        Number of random states.
    n_folds : int
        Number of inner folds.
    random_seed : int
        Base random seed.
    
    Returns:
    --------
    feature_frequency_df : pd.DataFrame
        DataFrame with columns: feature, frequency, rank
        Sorted by frequency descending, then feature name ascending.
    """
    rs_list = list(range(n_rs))
    
    frequency_counter = {}
    raw_count_counter = {}
    
    # Use ProcessPoolExecutor for parallel execution (same as GA version)
    n_workers = min(n_rs, os.cpu_count() or 1)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_rs = {
            executor.submit(
                _run_mi_single_rs,
                X_train, y_train, rs, n_folds, classifier_type, max_k
            ): rs for rs in rs_list
        }
        
        for future in as_completed(future_to_rs):
            rs = future_to_rs[future]
            try:
                results_with_auc = future.result()
                # AUC-weighted frequency accumulation (same as GA version)
                for selected_features, held_out_auc in results_with_auc:
                    weight = max(0.0, (held_out_auc - 0.5) * 2.0)  # Same weighting as GA
                    for feat in selected_features:
                        frequency_counter[feat] = frequency_counter.get(feat, 0) + weight
                        raw_count_counter[feat] = raw_count_counter.get(feat, 0) + 1
            except Exception as e:
                print(f"  WARNING: RS {rs} failed with error: {e}")
    
    # Build feature frequency DataFrame (same format as GA version)
    feature_frequency_df = build_feature_frequency_df(frequency_counter, raw_count_counter)
    
    # Print summary (same as GA version)
    print("\n" + "=" * 60)
    print("MI Feature Selection — Feature Ranking Comparison")
    print("=" * 60)
    
    # Weighted score Top-10
    weighted_top10 = sorted(frequency_counter.items(), key=lambda x: (-x[1], x[0]))[:10]
    print("\n[Weighted Score] Top-10 features:")
    for rank, (feat, score) in enumerate(weighted_top10, 1):
        print(f"  {rank:2d}. {feat:<30s}  score = {score:.4f}")
    
    # Raw count Top-10
    raw_top10 = sorted(raw_count_counter.items(), key=lambda x: (-x[1], x[0]))[:10]
    print("\n[Raw Count] Top-10 features:")
    for rank, (feat, count) in enumerate(raw_top10, 1):
        print(f"  {rank:2d}. {feat:<30s}  count = {count}")
    print("=" * 60)
    
    return feature_frequency_df

def build_feature_frequency_df(frequency_counter, raw_count_counter=None):
    """
    Build a feature frequency DataFrame from a frequency counter.
    """
    sorted_items = sorted(frequency_counter.items(), key=lambda x: (-x[1], x[0]))
    records = [{'feature': f, 'frequency': freq, 'rank': i}
               for i, (f, freq) in enumerate(sorted_items, 1)]
    df = pd.DataFrame(records)

    if raw_count_counter is not None and not df.empty:
        df['raw_count'] = df['feature'].map(raw_count_counter).fillna(0).astype(int)

    return df

# =============================================================================
# Block 4: Holdout Evaluation Helpers
# =============================================================================

def select_top_k_features(feature_frequency_df: pd.DataFrame, k: int) -> list[str]:
    """
    Selects the top K features from the MI frequency ranking DataFrame.
    The returned list of feature names is ordered by frequency rank.
    """
    sorted_df = feature_frequency_df.sort_values('rank')
    top_k_df = sorted_df.head(k)
    return top_k_df['feature'].tolist()

def get_holdout_classifier(clf_type: str, random_state: int = RANDOM_SEED):
    """Exposes a reusable classifier factory for final holdout evaluation."""
    return get_classifier(clf_type, inner=False)

# =============================================================================
# Block 4: K_GRID Holdout Evaluation Loop
# =============================================================================

def evaluate_holdout_k_grid(X_train, y_train, X_test, feature_frequency_df, clf_type,
                             k_grid=None, scaler_type=SCALER_TYPE_OUTER, case_ids=None,
                             y_test=None):
    """
    Loops over Top-K feature counts, trains final holdout models, computes AUC,
    and returns predictions/results.
    """
    if k_grid is None:
        k_grid = K_GRID

    y_true = np.asarray(y_test) if y_test is not None else None

    results = []
    for k in k_grid:
        # Select Top-K features from feature_frequency_df
        selected_features = select_top_k_features(feature_frequency_df, k)

        # Filter both train/test to these Top-K features
        X_train_filtered = X_train[selected_features]
        X_test_filtered = X_test[selected_features]

        # Fit the configured scaler on train only and transform train/test
        scaler = make_scaler(scaler_type)

        X_train_scaled = scaler.fit_transform(X_train_filtered)
        X_test_scaled = scaler.transform(X_test_filtered)

        # Train classifier on train
        clf = get_holdout_classifier(clf_type)
        clf.fit(X_train_scaled, y_train)

        # Predict probabilities on test
        y_score = clf.predict_proba(X_test_scaled)[:, 1]

        # Compute ROC AUC with roc_auc_score
        if y_true is not None:
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = 0.5
        else:
            auc = None

        results.append({
            'k': k,
            'auc': auc,
            'selected_features': selected_features,
            'y_true': y_true,
            'y_score': y_score,
            'case_index': X_test.index.tolist(),
            'case_id': [str(cid) for cid in case_ids] if case_ids is not None else X_test.index.tolist()
        })

    return results

# =============================================================================
# Block 5: Directory/Checkpoint Helpers and Missing-BW Skip Logic
# =============================================================================

def get_data_splits_dir(bw: int) -> Path:
    """Returns the path to the data splits directory for a given bin width."""
    return DATA_SPLITS_DIR / f"BW_{bw}"

def get_experiment_dir(bw: int, clf_type: str, fold_idx: int) -> Path:
    """Returns the path to the experiment directory for a given bin width, classifier type, and fold index."""
    return EXPERIMENTS_DIR / f"BW_{bw}" / clf_type / f"fold_{fold_idx}"

def bw_csv_exists(bw: int) -> bool:
    """Checks if the binwidth CSV file exists."""
    return resolve_binwidth_csv(bw).exists()

def save_json(data: dict, path: Path) -> None:
    """Saves a dictionary as a JSON file, ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

# -----------------------------------------------------------------------------
# Result file I/O functions for long-format checkpoint
# -----------------------------------------------------------------------------
def load_results_df() -> pd.DataFrame:
    """Load existing pipeline_results_mi_auc.csv or return empty DataFrame."""
    cols = ['bw', 'outer_rs', 'fold_idx', 'classifier', 'k', 'auc', 'timestamp']
    if not RESULTS_FILE.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(RESULTS_FILE, dtype={'bw': int, 'outer_rs': int, 'fold_idx': int, 'k': int, 'auc': float})
        return df
    except Exception as e:
        print(f"[WARNING] Cannot read {RESULTS_FILE}: {e}, starting from empty results.")
        return pd.DataFrame(columns=cols)

def is_fold_done(results_df, bw, outer_rs, fold_idx, k_grid=None, classifiers=None):
    """Check if all (classifier, k) combinations for a fold are already computed."""
    if k_grid is None: k_grid = K_GRID
    if classifiers is None: classifiers = ['svm', 'rf', 'xgboost', 'softvote-3']
    if results_df.empty: return False
    subset = results_df[(results_df['bw'] == bw) & (results_df['outer_rs'] == outer_rs) & (results_df['fold_idx'] == fold_idx)]
    if subset.empty: return False
    found_clfs = set(subset['classifier'].unique())
    found_ks = set(subset['k'].unique())
    return set(classifiers).issubset(found_clfs) and set(k_grid).issubset(found_ks)

def append_fold_results(rows):
    """Append rows to pipeline_results_mi_auc.csv, overwriting duplicates on (bw, outer_rs, fold_idx, classifier, k)."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    
    if RESULTS_FILE.exists():
        try:
            df_existing = pd.read_csv(RESULTS_FILE)
            # Drop existing rows that conflict with new rows on the unique key
            key_cols = ['bw', 'outer_rs', 'fold_idx', 'classifier', 'k']
            df_new_keys = set(zip(df_new['bw'], df_new['outer_rs'], df_new['fold_idx'], df_new['classifier'], df_new['k']))
            df_filtered = df_existing[
                ~df_existing.apply(lambda r: (r['bw'], r['outer_rs'], r['fold_idx'], r['classifier'], r['k']) in df_new_keys, axis=1)
            ]
            df_combined = pd.concat([df_filtered, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new
    
    df_combined.to_csv(RESULTS_FILE, index=False)

# -----------------------------------------------------------------------------
# In-memory softvote computation
# -----------------------------------------------------------------------------
def compute_softvote_in_memory(predictions_by_clf, k_grid=None):
    """Compute softvote-3 AUC from in-memory predictions (no disk I/O)."""
    if k_grid is None: k_grid = K_GRID
    softvote_aucs = {}
    for k in k_grid:
        scores_list = []
        y_true = None
        for clf_name in ['svm', 'rf', 'xgboost']:
            if clf_name not in predictions_by_clf: continue
            matching = [r for r in predictions_by_clf[clf_name] if r['k'] == k]
            if not matching: continue
            res = matching[0]
            scores_list.append(res['y_score'])
            if y_true is None: y_true = res['y_true']
        if not scores_list or y_true is None:
            softvote_aucs[k] = 0.5
            continue
        avg_score = np.mean(scores_list, axis=0)
        try:
            softvote_aucs[k] = roc_auc_score(y_true, avg_score)
        except ValueError:
            softvote_aucs[k] = 0.5
    return softvote_aucs

# -----------------------------------------------------------------------------
# Summary functions for multi-RS results
# -----------------------------------------------------------------------------
def compute_pipeline_summary(results_df):
    """Compute mean/std AUC grouped by (bw, classifier, k)."""
    summary = results_df.groupby(['bw', 'classifier', 'k'])['auc'].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ['bw', 'classifier', 'k', 'mean_auc', 'std_auc', 'n_folds']
    return summary

def print_summary(results_df, classifiers=None, top_k_values=None):
    """Pretty-print pipeline summary table."""
    if classifiers is None: classifiers = ['softvote-3', 'svm', 'rf', 'xgboost']
    if top_k_values is None: top_k_values = sorted(K_GRID)[:6]
    summary = compute_pipeline_summary(results_df)
    n_rs = results_df['outer_rs'].nunique()
    n_folds = results_df['fold_idx'].nunique()
    print(f"\n{'='*70}")
    print(f"Pipeline Summary  ({n_rs} outer RS × {n_folds} folds per RS)")
    print(f"{'='*70}")
    for bw in sorted(results_df['bw'].unique()):
        print(f"\nBW = {bw}:")
        sub = summary[(summary['bw'] == bw) & (summary['k'].isin(top_k_values))]
        for clf in classifiers:
            clf_sub = sub[sub['classifier'] == clf].sort_values('k')
            if clf_sub.empty: continue
            parts = [f"k={row['k']}: {row['mean_auc']:.4f}±{row['std_auc']:.4f}" for _, row in clf_sub.iterrows()]
            print(f"  {clf:<12s}: {' | '.join(parts)}")

# =============================================================================
# Heatmap Visualization (same pattern as cv_pipeline.ipynb)
# =============================================================================

def generate_auc_heatmaps(results_df: pd.DataFrame, bw: int, show_per_rs: bool = False):
    """
    Generate AUC heatmaps from results_df:
    1. Per-classifier fold AUC heatmaps (Row: K, Col: Fold, Cell: AUC)
    2. Summary heatmap (Row: K, Col: Classifier, Cell: Mean AUC ± SD)

    Parameters:
    -----------
    results_df : pd.DataFrame
        Long-format results with columns [bw, classifier, k, auc, fold_idx, outer_rs]
    bw : int
        Bin width to plot
    show_per_rs : bool
        If True, also plot per-RS heatmaps
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping heatmap.")
        return

    sub = results_df[results_df['bw'] == bw].copy()
    if sub.empty:
        print(f"No AUC records found for BW {bw} to plot heatmaps.")
        return

    all_classifiers = ['softvote-3', 'svm', 'rf', 'xgboost']

    # --- Per-classifier Fold AUC Heatmaps ---
    for clf in all_classifiers:
        df_clf = sub[sub['classifier'] == clf]
        if df_clf.empty:
            continue

        if show_per_rs and 'outer_rs' in df_clf.columns:
            # Plot per-RS
            for rs_val in sorted(df_clf['outer_rs'].unique()):
                df_rs = df_clf[df_clf['outer_rs'] == rs_val]
                pivot_auc = df_rs.pivot_table(index='k', columns='fold_idx', values='auc', aggfunc='first')
                _plot_single_heatmap(pivot_auc, f'{clf} — RS {rs_val}', bw, 'Fold Index')
        else:
            # Aggregate across all RS
            pivot_auc = df_clf.pivot_table(index='k', columns='fold_idx', values='auc', aggfunc='mean')
            _plot_single_heatmap(pivot_auc, clf, bw, 'Fold Index')

    # --- Summary Heatmap (Mean ± SD across RS) ---
    # Step 1: per-RS mean AUC (collapse folds within each RS)
    rs_mean = sub.groupby(['k', 'classifier', 'outer_rs'])['auc'].mean().reset_index()
    # Step 2: mean and std across RSs (reflects between-RS stability)
    summary_stats = rs_mean.groupby(['k', 'classifier'])['auc'].agg(['mean', 'std']).reset_index()

    pivot_mean = summary_stats.pivot(index='k', columns='classifier', values='mean')
    pivot_std = summary_stats.pivot(index='k', columns='classifier', values='std')

    ordered_clfs = [c for c in all_classifiers if c in pivot_mean.columns]
    pivot_mean = pivot_mean[ordered_clfs]
    pivot_std = pivot_std[ordered_clfs]

    annot_matrix = np.empty(pivot_mean.shape, dtype=object)
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            mean_val = pivot_mean.iloc[i, j]
            std_val = pivot_std.iloc[i, j]
            if pd.isna(mean_val):
                annot_matrix[i, j] = "NaN"
            else:
                annot_matrix[i, j] = f"{mean_val:.4f}\n±{std_val:.4f}"

    fig_sum, ax_sum = plt.subplots(figsize=(10, max(6, len(pivot_mean)*0.8)))
    sns.heatmap(pivot_mean, annot=annot_matrix, fmt="", cmap="RdYlBu_r", vmin=0.5, vmax=1.0,
                linewidths=0.5, cbar_kws={'label': 'Mean AUC'}, ax=ax_sum)
    ax_sum.set_title(f'Summary: Mean AUC ± SD (BW {bw})', fontsize=14, fontweight='bold')
    ax_sum.set_xlabel('Classifier', fontsize=12)
    ax_sum.set_ylabel('Top K Features', fontsize=12)
    plt.tight_layout()

    if SAVE_ARTIFACTS:
        save_path = ARTIFACTS_DIR / f'BW_{bw}' / 'heatmaps' / 'heatmap_summary_mean_std.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_sum.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig_sum)
        print(f"Saved Summary Heatmap: {save_path}")
    else:
        plt.show()
        plt.close(fig_sum)

    # Also save per-classifier fold heatmaps after the summary
    for clf in all_classifiers:
        df_clf = sub[sub['classifier'] == clf]
        if df_clf.empty:
            continue
        pivot_auc = df_clf.pivot_table(index='k', columns='fold_idx', values='auc', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_auc, annot=True, fmt=".4f", cmap="RdYlBu_r", vmin=0.5, vmax=1.0,
                    linewidths=0.5, cbar_kws={'label': 'AUC Score'}, ax=ax)
        ax.set_title(f'AUC per Fold - {clf} (BW {bw})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fold Index', fontsize=12)
        ax.set_ylabel('Top K Features', fontsize=12)
        plt.tight_layout()
        if SAVE_ARTIFACTS:
            save_path = ARTIFACTS_DIR / f'BW_{bw}' / 'heatmaps' / f'heatmap_fold_auc_{clf}.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved Fold Heatmap: {save_path.name}")
        else:
            plt.show()
            plt.close(fig)


def _plot_single_heatmap(pivot_auc, clf_name, bw, xlabel):
    """Helper to plot a single fold-AUC heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_auc, annot=True, fmt=".4f", cmap="RdYlBu_r", vmin=0.5, vmax=1.0,
                linewidths=0.5, cbar_kws={'label': 'AUC Score'}, ax=ax)
    ax.set_title(f'AUC per Fold - {clf_name} (BW {bw})', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Top K Features', fontsize=12)
    plt.tight_layout()
    if SAVE_ARTIFACTS:
        safe_name = clf_name.replace(' ', '_').replace('/', '_')
        save_path = ARTIFACTS_DIR / f'BW_{bw}' / 'heatmaps' / f'heatmap_fold_auc_{safe_name}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Fold Heatmap: {save_path.name}")
    else:
        plt.show()
        plt.close(fig)

# =============================================================================
# Block 2: Outer CV Generator
# =============================================================================

class OuterCVGenerator:
    @staticmethod
    def generate_splits(bw: int, overwrite: bool = False) -> dict:
        """Generates 8-fold outer cross-validation splits for a given bin width."""
        csv_path = resolve_binwidth_csv(bw)
        df, labels, feature_cols = load_feature_csv(csv_path)
        
        skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        splits_dir = get_data_splits_dir(bw)
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        split_paths = {}
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, labels)):
            train_path = splits_dir / f"fold_{fold_idx}_train_70.csv"
            test_path = splits_dir / f"fold_{fold_idx}_test_10.csv"
            
            split_paths[fold_idx] = {
                'train': train_path,
                'test': test_path
            }
            
            if not overwrite and train_path.exists() and test_path.exists():
                continue
                
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            safe_save_csv(train_df, train_path)
            safe_save_csv(test_df, test_path)
            
        return split_paths

    @staticmethod
    def load_fold(bw: int, fold_idx: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Loads the train and test splits for a given fold and bin width."""
        splits_dir = get_data_splits_dir(bw)
        train_path = splits_dir / f"fold_{fold_idx}_train_70.csv"
        test_path = splits_dir / f"fold_{fold_idx}_test_10.csv"
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        train_label_col = find_label_column(train_df)
        test_label_col = find_label_column(test_df)
        
        train_case_col = find_case_id_column(train_df)
        test_case_col = find_case_id_column(test_df)
        
        y_train = train_df[train_label_col]
        X_train = train_df.drop(columns=[train_case_col, train_label_col])
        
        y_test = test_df[test_label_col]
        X_test = test_df.drop(columns=[test_case_col, test_label_col])
        test_case_ids = test_df[test_case_col]
        train_case_ids = train_df[train_case_col]
        
        return X_train, y_train, X_test, y_test, test_case_ids, train_case_ids

    @staticmethod
    def load_splits_in_memory(bw: int, outer_rs: int) -> dict:
        """Load all outer fold splits for a given bin width and random state into memory."""
        csv_path = resolve_binwidth_csv(bw)
        df, labels, feature_cols = load_feature_csv(csv_path)
        if ORIGINAL_ONLY:
            df = filter_original_only(df)
            feature_cols = [c for c in df.columns if c.lower() not in ['label', 'casenumber', 'caseid']]
        skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=outer_rs)
        splits = {}
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, labels)):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            case_id_col = find_case_id_column(df)
            label_col = find_label_column(df)
            X_train = train_df.drop(columns=[case_id_col, label_col])
            y_train = train_df[label_col].reset_index(drop=True)
            train_case_ids = train_df[case_id_col].reset_index(drop=True)
            X_test = test_df.drop(columns=[case_id_col, label_col])
            y_test = test_df[label_col].reset_index(drop=True)
            test_case_ids = test_df[case_id_col].reset_index(drop=True)
            splits[fold_idx] = (X_train, y_train, X_test, y_test, test_case_ids, train_case_ids)
        return splits

# =============================================================================
# Block 5: Main Execution Loop and Pipeline Orchestrator
# =============================================================================

def run_single_fold(bw, outer_rs, fold_idx, X_train, y_train, X_test, y_test, test_case_ids, train_case_ids, results_df,
                    max_k=MAX_K, n_rs=MI_N_RS, n_folds=MI_N_FOLDS, random_seed=RANDOM_SEED, k_grid=None):
    """
    Runs the MI feature selection and holdout evaluation for a single fold.
    Uses in-memory data — no CSV I/O for splits or predictions.
    """
    if k_grid is None: k_grid = K_GRID
    
    # Checkpoint: skip if all (classifier, k) combos already done
    if is_fold_done(results_df, bw, outer_rs, fold_idx, k_grid=k_grid):
        print(f"[SKIP] BW={bw}, RS={outer_rs}, fold={fold_idx} — already done.")
        return []
    
    # Run MI on fold
    print(f"Running MI feature selection on fold {fold_idx} (n_rs={n_rs}, n_folds={n_folds}, max_k={max_k})...")
    feature_frequency_df = run_mi_on_fold(
        X_train, y_train,
        classifier_type=CLASSIFIER_TYPE,
        max_k=max_k,
        n_rs=n_rs,
        n_folds=n_folds,
        random_seed=random_seed
    )
    
    # Save MI artifacts if enabled
    if SAVE_ARTIFACTS:
        art_dir = ARTIFACTS_DIR / f"BW_{bw}" / f"outer_rs_{outer_rs}" / f"fold_{fold_idx}"
        art_dir.mkdir(parents=True, exist_ok=True)
        freq_path = art_dir / 'feature_frequencies.csv'
        safe_save_csv(feature_frequency_df, freq_path)
        print(f"  Saved feature_frequencies.csv → {art_dir}")
    
    # Evaluate holdout for each classifier in-memory
    predictions_by_clf = {}
    for clf in ['svm', 'rf', 'xgboost']:
        print(f"  Evaluating {clf}...")
        ho_results = evaluate_holdout_k_grid(
            X_train, y_train, X_test,
            feature_frequency_df, clf,
            k_grid=k_grid, scaler_type=SCALER_TYPE_OUTER,
            case_ids=test_case_ids, y_test=y_test
        )
        predictions_by_clf[clf] = ho_results
    
    # Compute softvote in-memory
    softvote_aucs = compute_softvote_in_memory(predictions_by_clf, k_grid)
    
    # Assemble result rows
    rows = []
    timestamp = datetime.now().isoformat()
    for clf in ['svm', 'rf', 'xgboost', 'softvote-3']:
        for k in k_grid:
            if clf == 'softvote-3':
                auc_val = softvote_aucs.get(k, 0.5)
            else:
                matching = [r for r in predictions_by_clf[clf] if r['k'] == k]
                auc_val = matching[0]['auc'] if matching else 0.5
            rows.append({
                'bw': bw,
                'outer_rs': outer_rs,
                'fold_idx': fold_idx,
                'classifier': clf,
                'k': k,
                'auc': auc_val,
                'timestamp': timestamp
            })
    
    # Persist to long-format checkpoint
    append_fold_results(rows)
    
    return rows

def run_pipeline(force_recompute=False, bin_widths=None, outer_rs_list=None, n_outer_folds=None,
                 max_k=MAX_K, n_rs=MI_N_RS, n_folds=MI_N_FOLDS, random_seed=RANDOM_SEED, k_grid=None,
                 collect_preds=False):
    """
    Main pipeline execution loop — multi-RS outer CV with in-memory I/O.
    """
    # Initialize configuration
    if bin_widths is None:
        bin_widths = BIN_WIDTHS
    if outer_rs_list is None:
        outer_rs_list = OUTER_RS_LIST
    if n_outer_folds is None:
        n_outer_folds = N_OUTER_FOLDS
    if k_grid is None:
        k_grid = K_GRID
    
    # Ensure workspace directories exist
    ensure_workspace_dirs()
    
    # Load existing results for checkpointing
    results_df = load_results_df()
    
    # Backup existing results if force_recompute
    if force_recompute and RESULTS_FILE.exists():
        backup_path = RESULTS_FILE.with_suffix('.backup.csv')
        import shutil
        shutil.copy(RESULTS_FILE, backup_path)
        print(f"Backed up existing results to: {backup_path}")
        results_df = pd.DataFrame(columns=['bw', 'outer_rs', 'fold_idx', 'classifier', 'k', 'auc', 'timestamp'])
    
    # Main execution loop
    total_folds = len(bin_widths) * len(outer_rs_list) * n_outer_folds
    processed_folds = 0
    
    for bw in bin_widths:
        if not bw_csv_exists(bw):
            print(f"[WARNING] BW={bw} CSV not found, skipping.")
            continue
            
        for outer_rs in outer_rs_list:
            print(f"\n{'='*70}")
            print(f"Processing BW={bw}, Outer RS={outer_rs}")
            print(f"{'='*70}")
            
            # Load all splits for this BW and RS into memory
            splits = OuterCVGenerator.load_splits_in_memory(bw, outer_rs)
            
            for fold_idx in range(n_outer_folds):
                if fold_idx not in splits:
                    print(f"[WARNING] Fold {fold_idx} not found in splits for BW={bw}, RS={outer_rs}")
                    continue
                    
                X_train, y_train, X_test, y_test, test_case_ids, train_case_ids = splits[fold_idx]
                
                print(f"\n--- Fold {fold_idx} (train: {len(X_train)}, test: {len(X_test)}) ---")
                
                # Run single fold
                rows = run_single_fold(
                    bw, outer_rs, fold_idx,
                    X_train, y_train, X_test, y_test, test_case_ids, train_case_ids,
                    results_df, max_k=max_k, n_rs=n_rs, n_folds=n_folds,
                    random_seed=random_seed, k_grid=k_grid
                )
                
                if rows:
                    # Update results_df with new rows
                    new_rows_df = pd.DataFrame(rows)
                    results_df = pd.concat([results_df, new_rows_df], ignore_index=True)
                    processed_folds += 1
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"Pipeline completed!")
    print(f"Processed {processed_folds}/{total_folds} folds")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"{'='*70}")
    
    # Print summary statistics
    if not results_df.empty:
        print_summary(results_df)
        
        # Generate heatmaps per BW
        for bw in sorted(results_df['bw'].unique()):
            generate_auc_heatmaps(results_df, bw=bw, show_per_rs=True)
    
    return results_df

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Starting MI-based CV pipeline with AUC-weighted frequency...")
    print(f"Configuration:")
    print(f"  BIN_WIDTHS: {BIN_WIDTHS}")
    print(f"  OUTER_RS_LIST: {OUTER_RS_LIST}")
    print(f"  N_OUTER_FOLDS: {N_OUTER_FOLDS}")
    print(f"  MI_N_RS: {MI_N_RS}")
    print(f"  MI_N_FOLDS: {MI_N_FOLDS}")
    print(f"  MAX_K (Global): {MAX_K}")
    print(f"  K_GRID: {K_GRID}")
    print(f"  RESULTS_FILE: {RESULTS_FILE}")
    
    # Run the pipeline
    results = run_pipeline(force_recompute=False)
    
    print("\nPipeline execution completed successfully!")