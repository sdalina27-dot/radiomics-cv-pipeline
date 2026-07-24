#!/usr/bin/env python3
"""
8-fold outer CV pipeline with Mutual Information (filter-based) feature selection
instead of GA. This is a modified version of cv_pipeline.ipynb that replaces
the genetic algorithm feature selection with Mutual Information scoring.

Key changes:
- Replaced GA feature selection with Mutual Information (mutual_info_classif)
- Removed all GA-related functions (evolve, init_population, crossover, mutate, etc.)
- Added MI-based feature ranking and frequency calculation
- Kept all other pipeline components (data loading, preprocessing, holdout evaluation)
"""

# =============================================================================
# Block 1: Global Configuration
# =============================================================================
# This cell defines all configurable parameters for the 8-fold CV pipeline.
# Modify these values to adjust pipeline behavior without changing code logic.

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
# BIN_WIDTHS: List of bin widths to process. Add more BWs as needed.
# Design: List type allows easy extension for additional bin widths.
BIN_WIDTHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# CLASSIFIER_TYPE: Single classifier for internal feature ranking (not used for MI)
# Kept for compatibility with existing code structure
CLASSIFIER_TYPE = 'svm'

# CLASSIFIERS: List of classifier types for holdout evaluation.
# Options: 'svm', 'rf', 'xgboost'
CLASSIFIERS = ['svm', 'rf', 'xgboost']

# -----------------------------------------------------------------------------
# Cross-Validation Configuration
# -----------------------------------------------------------------------------
# N_OUTER_FOLDS: Number of outer CV folds (8-fold CV)
N_OUTER_FOLDS = 8

# RANDOM_SEED: Fixed seed for reproducibility of 8-fold split
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Scaling Configuration
# -----------------------------------------------------------------------------
# SCALER_TYPE: Scaler to use - 'StandardScaler' or 'QuantileTransformer'
# Note: Stored as string to allow conditional instantiation
SCALER_TYPE = 'QuantileTransformer'
SCALER_TYPE_OUTER = 'QuantileTransformer'
SCALER_TYPE_INNER = 'StandardScaler' 

# -----------------------------------------------------------------------------
# Feature Selection Configuration (Mutual Information)
# -----------------------------------------------------------------------------
# PEARSON_THRESHOLD: Correlation threshold for feature filtering (pre-MI)
PEARSON_THRESHOLD = 0.7

# ORIGINAL_ONLY: If True, only use original_ features (no wavelet)
ORIGINAL_ONLY = False

# MI Configuration
MI_N_RS = 16  # Number of random states for MI stability estimation
MI_N_FOLDS = 5  # Number of folds for MI inner CV

# Feature Selection Grid
K_GRID = [5, 10, 15, 20, 30, 40, 50]
MAX_K = 20  # Maximum number of features to select

# -----------------------------------------------------------------------------
# Outer CV Multi-RS Configuration
# -----------------------------------------------------------------------------
# Multiple RS outer CV for stability estimation
OUTER_RS_LIST = [42, 123, 456]
RESULTS_FILE = None  # Will be set dynamically
SAVE_ARTIFACTS = True
ARTIFACTS_DIR = None  # Will be set dynamically

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
# Base directories for data and experiments
import os
from pathlib import Path
DATA_DIR = Path('/home/ser/pipeline/data')
WORKSPACE_DIR = Path('/home/ser/pipeline/workspace/test3_r0.7_mi')
DATA_SPLITS_DIR = WORKSPACE_DIR / 'data_splits'
EXPERIMENTS_DIR = WORKSPACE_DIR / f'experiments_{SCALER_TYPE_OUTER}_{SCALER_TYPE_INNER}'

# Set RESULTS_FILE and ARTIFACTS_DIR
RESULTS_FILE = WORKSPACE_DIR / f'pipeline_results_mi.csv'
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
# This cell defines robust helper functions for data loading, column detection,
# and workspace directory management.

def resolve_binwidth_csv(bw: int) -> Path:
    """
    Looks under DATA_DIR for Cine_output_20260606_binWidth_{bw}.csv and returns the path.
    """
    return DATA_DIR / f"Cine_output_20260606_binWidth_{bw}.csv"

def find_case_id_column(df: pd.DataFrame) -> str:
    """
    Detects CaseNumber or CaseID column case-insensitively.
    """
    for col in df.columns:
        if col.lower() in ['casenumber', 'caseid']:
            return col
    raise ValueError("No CaseNumber or CaseID column found in DataFrame.")

def find_label_column(df: pd.DataFrame) -> str:
    """
    Detects Label column case-insensitively.
    """
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
    """
    Filter DataFrame to only keep original features (no wavelet).
    Original features have 'original' in their name but NOT 'wavelet'.
    Preserves non-feature columns (CaseNumber/CaseID, Label).
    """
    non_feature_cols = [c for c in df.columns if c.lower() in ['label', 'casenumber', 'caseid']]
    feature_cols = [c for c in df.columns if c.lower() not in ['label', 'casenumber', 'caseid']]
    # Original features: contain 'original' but NOT 'wavelet'
    original_cols = [c for c in feature_cols
                     if 'original' in c.lower() and 'wavelet' not in c.lower()]
    return df[non_feature_cols + original_cols]

def ensure_workspace_dirs() -> tuple[Path, Path]:
    """
    Ensures that DATA_SPLITS_DIR and EXPERIMENTS_DIR exist and returns them.
    """
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_SPLITS_DIR, EXPERIMENTS_DIR

def safe_save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Safely saves a DataFrame to the specified path, creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# =============================================================================
# Block 3: Preprocessing Foundation (Fit-Train-Only Filters & Scaler Factory)
# =============================================================================
# This cell defines the preprocessing filters and scaler factory that will be
# used by the MI feature selection and holdout evaluation. All operations are strictly
# fit on training data only to prevent data leakage.

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
    Matches the Kaggle notebook's logic but operates on and returns pandas DataFrames/lists.
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
# Block 3: Mutual Information Feature Selection Engine
# =============================================================================
# This replaces the GA feature selection with Mutual Information (filter-based) approach

from collections import Counter

def compute_mutual_info_scores(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> np.ndarray:
    """
    Compute Mutual Information scores for all features against the target.
    Returns array of MI scores, one per feature in X_train.columns order.
    """
    try:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
        return mi_scores
    except Exception as e:
        print(f"Warning: MI computation failed: {e}")
        # Fallback: return uniform scores
        return np.ones(X_train.shape[1])

def run_mi_on_fold(X_train: pd.DataFrame, y_train: pd.Series, 
                   classifier_type: str = CLASSIFIER_TYPE,
                   n_rs: int = MI_N_RS, n_folds: int = MI_N_FOLDS,
                   random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Runs Mutual Information feature selection across multiple random states and folds.
    
    This is the MI-based replacement for run_ga_on_fold(). Instead of evolving
    a population of feature subsets, we compute MI scores for each feature and
    aggregate across multiple RS×fold combinations for stability.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature DataFrame.
    y_train : pd.Series
        Training label Series.
    classifier_type : str
        Classifier type (kept for compatibility, not used in MI).
    n_rs : int
        Number of random states for MI computation.
    n_folds : int
        Number of inner folds for MI stability estimation.
    random_seed : int
        Base random seed.
    
    Returns:
    --------
    feature_frequency_df : pd.DataFrame
        DataFrame with columns: feature, frequency, rank
        Sorted by frequency descending, then feature name ascending.
    """
    # Apply preprocessing filters (same as GA version)
    selected_features = fit_train_only_prefilter(X_train, y_train, threshold=PEARSON_THRESHOLD)
    if len(selected_features) < 2:
        selected_features = list(X_train.columns[:2])
    
    X_filtered = X_train[selected_features]
    
    # Fit scaler on training data only
    scaler = make_scaler(SCALER_TYPE_INNER)
    X_filtered_scaled_values = scaler.fit_transform(X_filtered)
    X_filtered_scaled = pd.DataFrame(
        X_filtered_scaled_values, 
        columns=X_filtered.columns, 
        index=X_filtered.index
    )
    
    # Compute MI scores across multiple random states for stability
    frequency_counter = Counter()
    raw_count_counter = Counter()
    
    rs_list = list(range(n_rs))
    
    for rs in rs_list:
        # Create inner CV splits for this random state
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_filtered_scaled, y_train)):
            X_train_fold = X_filtered_scaled.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            
            # Compute MI scores on this fold
            mi_scores = compute_mutual_info_scores(X_train_fold, y_train_fold, random_state=rs)
            
            # Get feature indices sorted by MI score (descending)
            feature_indices = np.argsort(mi_scores)[::-1]  # descending order
            
            # Assign weights based on MI scores (normalized to sum to 1 per fold)
            # Higher MI score = higher weight
            if np.sum(mi_scores) > 0:
                normalized_scores = mi_scores / np.sum(mi_scores)
            else:
                normalized_scores = np.ones_like(mi_scores) / len(mi_scores)
            
            # Accumulate scores for each feature
            for feat_idx, score in zip(feature_indices, normalized_scores):
                feat_name = X_filtered.columns[feat_idx]
                frequency_counter[feat_name] += score
                raw_count_counter[feat_name] += 1
    
    # Build feature frequency DataFrame (same format as GA version)
    feature_frequency_df = build_feature_frequency_df(frequency_counter, raw_count_counter)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MI Feature Selection — Feature Ranking Summary")
    print("=" * 60)
    
    # MI-based Top-10
    mi_top10 = sorted(frequency_counter.items(), key=lambda x: (-x[1], x[0]))[:10]
    print("\n[MI Score] Top-10 features:")
    for rank, (feat, score) in enumerate(mi_top10, 1):
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
    
    Parameters:
    -----------
    frequency_counter : dict
        Dict mapping feature names to MI-weighted scores (float).
        Higher score = higher rank.
    raw_count_counter : dict, optional
        Dict mapping feature names to raw integer counts (unweighted).
        If provided, an additional 'raw_count' column is added for comparison.
    
    Returns:
    --------
    pd.DataFrame
        Columns: feature, frequency, rank (, raw_count if raw_count_counter provided).
        Sorted by frequency descending, then feature name ascending.
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
# This cell defines the holdout evaluation helper functions that select Top-K
# features from the MI frequency ranking and expose a reusable classifier factory
# for final holdout evaluation.

def select_top_k_features(feature_frequency_df: pd.DataFrame, k: int) -> list[str]:
    """
    Selects the top K features from the MI frequency ranking DataFrame.
    The returned list of feature names is ordered by frequency rank.
    """
    # Sort by rank to ensure correct order, then take the first k rows
    sorted_df = feature_frequency_df.sort_values('rank')
    top_k_df = sorted_df.head(k)
    return top_k_df['feature'].tolist()

def get_holdout_classifier(clf_type: str, random_state: int = RANDOM_SEED):
    """
    Exposes a reusable classifier factory for final holdout evaluation.
    Supports 'svm', 'rf', and 'xgboost'.
    """
    return get_classifier(clf_type, inner=False)

# =============================================================================
# Block 4: K_GRID Holdout Evaluation Loop
# =============================================================================
# This cell defines the holdout evaluation loop over the Top-K feature counts
# grid (K_GRID) and a helper to build a tidy predictions DataFrame.

def evaluate_holdout_k_grid(X_train, y_train, X_test, feature_frequency_df, clf_type,
                             k_grid=None, scaler_type=SCALER_TYPE_OUTER, case_ids=None):
    """
    Loops over Top-K feature counts, trains final holdout models, computes AUC,
    and returns predictions/results.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature DataFrame.
    y_train : pd.Series or np.ndarray
        Training label Series or array.
    X_test : pd.DataFrame
        Test feature DataFrame (may contain Label column).
    feature_frequency_df : pd.DataFrame
        Ranked feature frequency DataFrame from MI.
    clf_type : str
        Classifier type ('svm', 'rf', 'xgboost').
    k_grid : list of int, optional
        Grid of Top-K feature counts to evaluate. Defaults to K_GRID.
    scaler_type : str, default=SCALER_TYPE_OUTER
        Scaler type to use.
    case_ids : pd.Series or list, optional
        Original CaseID values for test samples.

    Returns:
    --------
    list of dict
        List of result dictionaries containing k, auc, selected_features, y_true, y_score, case_index, case_id.
    """
    if k_grid is None:
        k_grid = K_GRID

    # Extract y_true from X_test if it contains the label column
    try:
        label_col = find_label_column(X_test)
        y_true = X_test[label_col].values
    except Exception:
        y_true = None

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

        # Train get_holdout_classifier(clf_type) on train
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

def build_holdout_predictions_df(results: list[dict]) -> pd.DataFrame:
    """
    Creates a tidy DataFrame with columns: k, auc, case_index, case_id, y_true, y_score, selected_features.
    """
    rows = []
    for res in results:
        k = res['k']
        auc = res['auc']
        selected_features = res['selected_features']
        y_true = res['y_true']
        y_score = res['y_score']

        # Determine case indices
        if 'case_index' in res:
            case_indices = res['case_index']
        elif hasattr(y_true, 'index'):
            case_indices = y_true.index.tolist()
        else:
            case_indices = list(range(len(y_true))) if y_true is not None else list(range(len(y_score)))

        # Determine case IDs
        if 'case_id' in res:
            case_ids = res['case_id']
        else:
            case_ids = case_indices

        for idx, (yt, ys) in enumerate(zip(y_true if y_true is not None else [None]*len(y_score), y_score)):
            rows.append({
                'k': k,
                'auc': auc,
                'case_index': case_indices[idx],
                'case_id': str(case_ids[idx]),
                'y_true': yt,
                'y_score': ys,
                'selected_features': selected_features
            })

    return pd.DataFrame(rows)

# =============================================================================
# Block 5: Directory/Checkpoint Helpers and Missing-BW Skip Logic
# =============================================================================
# This cell defines directory resolution, checkpointing, and missing-BW skip logic.
# It ensures that the pipeline can gracefully handle non-writable environments
# and skip already completed folds or missing input files.

REQUIRED_WORKSPACE_DIR = os.path.join(os.getcwd(), 'workspace')

def resolve_workspace_dir(required_dir: Path = Path('/home/ser/pipeline/workspace'), fallback_dir: Path = DATA_DIR / 'workspace') -> Path:
    """
    Resolves the workspace directory.
    Prefer required_dir when it exists and is writable.
    If required_dir cannot be created/written in this environment, fall back to fallback_dir and print a clear warning.
    """
    try:
        # Try to create the directory if it doesn't exist
        required_dir.mkdir(parents=True, exist_ok=True)
        # Try to write a temporary file to verify writability
        test_file = required_dir / '.write_test'
        test_file.touch()
        test_file.unlink()
        return required_dir
    except Exception as e:
        print(f"WARNING: Required workspace directory {required_dir} is not writable or cannot be created.")
        print(f"Error: {e}")
        print(f"Falling back to: {fallback_dir}")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir

def get_data_splits_dir(bw: int) -> Path:
    """
    Returns the path to the data splits directory for a given bin width.
    """
    # Use the global DATA_SPLITS_DIR
    return DATA_SPLITS_DIR / f"BW_{bw}"

def get_experiment_dir(bw: int, clf_type: str, fold_idx: int) -> Path:
    """
    Returns the path to the experiment directory for a given bin width, classifier type, and fold index.
    """
    # Use the global EXPERIMENTS_DIR
    return EXPERIMENTS_DIR / f"BW_{bw}" / clf_type / f"fold_{fold_idx}"

def get_fold_paths(bw: int, clf_type: str, fold_idx: int) -> dict:
    """
    Returns a dictionary of paths for a given fold.
    """
    exp_dir = get_experiment_dir(bw, clf_type, fold_idx)
    return {
        'experiment_dir': exp_dir,
        'mi_results': exp_dir / 'mi_results.json',
        'selected_features': exp_dir / 'selected_features.csv',
        'holdout_predictions': exp_dir / 'holdout_predictions.csv'
    }

def bw_csv_exists(bw: int) -> bool:
    """
    Checks if the binwidth CSV file exists.
    """
    return resolve_binwidth_csv(bw).exists()

def checkpoint_exists(fold_paths: dict) -> bool:
    """
    Checks if the checkpoint (holdout_predictions.csv) exists.
    """
    return fold_paths['holdout_predictions'].exists()

def save_json(data: dict, path: Path) -> None:
    """
    Saves a dictionary as a JSON file, ensuring parent directories exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

# -----------------------------------------------------------------------------
# Result file I/O functions for long-format checkpoint
# -----------------------------------------------------------------------------
def load_results_df() -> pd.DataFrame:
    """Load existing pipeline_results_mi.csv or return empty DataFrame."""
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
    """Append rows to pipeline_results_mi.csv (header only on first write)."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    write_header = not RESULTS_FILE.exists()
    df_new.to_csv(RESULTS_FILE, mode='a', header=write_header, index=False)

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

def compute_variance_decomposition(results_df):
    """Decompose AUC variance into between-RS and within-RS components."""
    rs_means = results_df.groupby(['bw', 'outer_rs', 'classifier', 'k'])['auc'].mean().reset_index().rename(columns={'auc': 'mean_auc_per_rs'})
    between_rs = rs_means.groupby(['bw', 'classifier', 'k'])['mean_auc_per_rs'].std().reset_index().rename(columns={'mean_auc_per_rs': 'between_rs_std'})
    within_rs = results_df.groupby(['bw', 'outer_rs', 'classifier', 'k'])['auc'].std().reset_index().rename(columns={'auc': 'within_rs_std'})
    within_rs_avg = within_rs.groupby(['bw', 'classifier', 'k'])['within_rs_std'].mean().reset_index().rename(columns={'within_rs_std': 'within_rs_std_avg'})
    return pd.merge(between_rs, within_rs_avg, on=['bw', 'classifier', 'k'])

def print_summary(results_df, classifiers=None, top_k_values=None):
    """Pretty-print pipeline summary table."""
    if classifiers is None: classifiers = ['softvote-3', 'svm', 'rf', 'xgboost']
    if top_k_values is None: top_k_values = sorted(K_GRID)[:4]
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
# Block 2: Outer CV Generator
# =============================================================================
# This cell defines the OuterCVGenerator class which handles the generation
# and loading of the 8-fold outer cross-validation splits.

class OuterCVGenerator:
    @staticmethod
    def generate_splits(bw: int, overwrite: bool = False) -> dict:
        """
        Generates 8-fold outer cross-validation splits for a given bin width.
        Loads the feature CSV, applies StratifiedKFold, and saves the train/test splits.
        
        [DEPRECATED] This function writes CSV split files, replaced by load_splits_in_memory.
        Kept for backward compatibility and debugging.
        
        Parameters:
        -----------
        bw : int
            Bin width to process.
        overwrite : bool, default=False
            If True, overwrites existing split files.
            
        Returns:
        --------
        dict
            A dictionary mapping fold index to a dictionary of train/test file paths.
        """
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
        """
        Loads the train and test splits for a given fold and bin width.
        
        [DEPRECATED] This function reads from CSV split files, replaced by load_splits_in_memory.
        Kept for backward compatibility.
        
        Parameters:
        -----------
        bw : int
            Bin width.
        fold_idx : int
            Fold index (0 to N_OUTER_FOLDS-1).
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test)
        """
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

    # -----------------------------------------------------------------------------
    # In-memory split loading (no CSV I/O)
    # -----------------------------------------------------------------------------
    @staticmethod
    def load_splits_in_memory(bw: int, outer_rs: int) -> dict:
        """Load all outer fold splits for a given bin width and random state into memory."""
        csv_path = resolve_binwidth_csv(bw)
        df, labels, feature_cols = load_feature_csv(csv_path)
        # [ORIGINAL_ONLY] Filter to original_ features only if enabled
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
# This cell defines the main pipeline execution loop and the single-fold runner.
# It supports checkpointing via long-format CSV, multi-RS outer CV, in-memory I/O.

def run_single_fold(bw, outer_rs, fold_idx, X_train, y_train, X_test, y_test, test_case_ids, train_case_ids, results_df,
                    max_k=MAX_K, n_rs=MI_N_RS, n_folds=MI_N_FOLDS, random_seed=RANDOM_SEED, k_grid=None,
                    crossover_rate=None, mutation_rate=None):
    """
    Runs the MI feature selection and holdout evaluation for a single fold.
    Uses in-memory data — no CSV I/O for splits or predictions.
    
    Parameters:
    -----------
    bw, outer_rs, fold_idx : int
        Bin width, outer random state, fold index
    X_train, y_train, X_test, y_test : array-like
        Fold data (already split)
    test_case_ids, train_case_ids : pd.Series
        Case ID arrays
    results_df : pd.DataFrame
        Current results for checkpoint checking
    max_k : int
        Maximum number of features (kept for compatibility)
    n_rs : int
        Number of random states for MI computation
    n_folds : int
        Number of inner folds for MI computation
    random_seed : int
        Base random seed
    k_grid : list
        Top-K feature counts to evaluate
    crossover_rate, mutation_rate : float
        Kept for compatibility (not used in MI version)
        
    Returns:
    --------
    list of dict
        Rows appended to results_df (empty if skipped)
    """
    if k_grid is None: k_grid = K_GRID
    
    # Checkpoint: skip if all (classifier, k) combos already done
    if is_fold_done(results_df, bw, outer_rs, fold_idx, k_grid=k_grid):
        print(f"[SKIP] BW={bw}, RS={outer_rs}, fold={fold_idx} — already done.")
        return []
    
    # Prepare X_test for holdout evaluation (needs Label column)
    X_test_eval = X_test.copy()
    X_test_eval['Label'] = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Run MI on fold
    print(f"Running MI feature selection on fold {fold_idx} (n_rs={n_rs}, n_folds={n_folds})...")
    feature_frequency_df = run_mi_on_fold(
        X_train, y_train,
        classifier_type=CLASSIFIER_TYPE,
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
            X_train, y_train, X_test_eval,
            feature_frequency_df, clf,
            k_grid=k_grid, scaler_type=SCALER_TYPE_OUTER,
            case_ids=test_case_ids
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
    
    This is the MI-based version of the pipeline. Instead of running GA for feature
    selection, it uses Mutual Information to rank features.
    
    Parameters:
    -----------
    force_recompute : bool
        If True, backup existing RESULTS_FILE and start fresh
    bin_widths : list of int
        Bin widths to process
    outer_rs_list : list of int
        Random states for outer CV splits
    n_outer_folds : int
        Number of outer folds per RS
    max_k : int
        Maximum number of features (kept for compatibility)
    n_rs : int
        Number of random states for MI computation
    n_folds : int
        Number of inner folds for MI computation
    random_seed : int
        Base random seed
    k_grid : list
        Top-K feature counts to evaluate
    collect_preds : bool
        Kept for compatibility (not used in MI version)
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
    
    return results_df

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Starting MI-based CV pipeline...")
    print(f"Configuration:")
    print(f"  BIN_WIDTHS: {BIN_WIDTHS}")
    print(f"  OUTER_RS_LIST: {OUTER_RS_LIST}")
    print(f"  N_OUTER_FOLDS: {N_OUTER_FOLDS}")
    print(f"  MI_N_RS: {MI_N_RS}")
    print(f"  MI_N_FOLDS: {MI_N_FOLDS}")
    print(f"  K_GRID: {K_GRID}")
    print(f"  RESULTS_FILE: {RESULTS_FILE}")
    
    # Run the pipeline
    results = run_pipeline(force_recompute=False)
    
    print("\nPipeline execution completed successfully!")