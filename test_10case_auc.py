#!/usr/bin/env python3
"""
Evaluate GA-derived high-frequency feature subsets on the unseen 10-case holdout set.

Feature ranking logic is kept the same as the 70-case unrestricted pipeline:
1. Read GA `results_detail.json`.
2. Reconstruct the filtered feature list used by each GA fold using the 70-case CSV.
3. Count selected feature frequencies across GA runs/folds.
4. Use top-K stable features.

Difference from the 70-case CV pipeline:
- Models are trained on the 70-case CSV.
- Predictions are made once on the unseen 10-case CSV.
- Outputs are written under `10case_result/` by default.
"""

import os
import json
import warnings
import argparse
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression  # noqa: F401 - kept for parity with original imports
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: F401 - kept for parity with original imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.base import clone

warnings.filterwarnings("ignore")

FULL_RS = [42, 100, 2026, 888, 777, 1234, 5678, 9999, 314, 271, 1618, 500, 2025, 404, 1337, 6789]


def pearson_correlation_filter(X, threshold):
    """
    Pearson correlation filter matching the Kaggle notebook's logic.

    Includes the `if a in to_drop: continue` and `if b in to_drop: continue` checks.
    """
    if threshold >= 1.0 or X.shape[1] <= 1:
        return X, np.arange(X.shape[1])

    corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
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

    keep = np.array([i for i in range(n_features) if i not in to_drop])
    return X[:, keep], keep


def get_condition_feature_ranking(json_path, csv_path, r_thresh):
    """
    Reads `results_detail.json`, applies VarianceThreshold(0.0) and the Pearson filter
    to the 70-case CSV, maps GA-selected indices to true feature names, and returns
    a frequency-sorted ranking.

    The 10-case holdout CSV is intentionally not used here.
    """
    df = pd.read_csv(csv_path)
    feature_names = df.drop(columns=["CaseNumber", "Label"]).columns.tolist()
    X = df.drop(columns=["CaseNumber", "Label"]).values.astype(np.float64)
    y = df["Label"].values.astype(np.int32)

    with open(json_path, "r") as f:
        results = json.load(f)

    auc_unrestricted_results = [r for r in results if r.get("method") == "AUC_unrestricted"]

    feature_counts = Counter()

    for entry in auc_unrestricted_results:
        random_state = entry["random_state"]
        fold_features = entry["fold_features"]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
            if fold_idx >= len(fold_features):
                continue

            X_train = X[train_idx]

            vt = VarianceThreshold(threshold=0.0)
            vt.fit(X_train)
            vt_mask = vt.get_support()
            X_train_vt = X_train[:, vt_mask]

            _, corr_keep = pearson_correlation_filter(X_train_vt, r_thresh)

            vt_feature_names = [name for name, keep in zip(feature_names, vt_mask) if keep]
            filtered_feature_names = [vt_feature_names[i] for i in corr_keep]

            selected_indices = fold_features[fold_idx]
            selected_feature_names = [filtered_feature_names[idx] for idx in selected_indices]

            feature_counts.update(selected_feature_names)

    sorted_ranking = sorted(feature_counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_ranking


def run_external_prediction(
    X_train,
    y_train,
    cases_train,
    X_test,
    y_test,
    cases_test,
    rs,
    fixed_features,
    clfs,
):
    """
    Train on the 70-case training set and predict once on the unseen 10-case set.
    """
    del rs
    del cases_train

    X_train_restricted = X_train[fixed_features].values
    X_test_restricted = X_test[fixed_features].values

    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_k = scaler.fit_transform(X_train_restricted)
    X_test_k = scaler.transform(X_test_restricted)

    results = {}
    for clf_name, clf in clfs.items():
        clf_instance = clone(clf)
        try:
            clf_instance.fit(X_train_k, y_train)
            if hasattr(clf_instance, "predict_proba"):
                probs = clf_instance.predict_proba(X_test_k)[:, 1]
            else:
                decision = clf_instance.decision_function(X_test_k)
                probs = 1.0 / (1.0 + np.exp(-decision))
        except Exception:
            probs = np.full(len(y_test), 0.5)

        results[clf_name] = {
            "probs": probs.astype(float).tolist(),
            "y_true": y_test.astype(int).tolist(),
            "cases": cases_test.astype(str).tolist(),
        }

    return results


def aggregate_external_ensembles(rs_results):
    for rs_idx in range(len(rs_results)):
        rs_data = rs_results[rs_idx]
        ens_key = "SoftVote-3"
        base_clfs = ["SVM", "RandomForest", "xgboost"]
        base_probs = [rs_data[b_clf]["probs"] for b_clf in base_clfs]
        avg_probs = np.mean(base_probs, axis=0).astype(float).tolist()
        rs_data[ens_key] = {
            "probs": avg_probs,
            "y_true": rs_data[base_clfs[0]]["y_true"],
            "cases": rs_data[base_clfs[0]]["cases"],
        }
    return rs_results


def compute_external_metrics_df(all_rs_metrics_by_k, K_grid, classifiers):
    metrics = []
    for k in K_grid:
        rs_results = all_rs_metrics_by_k[k]
        for clf_name in classifiers:
            pooled_y_true = []
            pooled_probs = []
            per_rs_aucs = []

            for rs_idx, _ in enumerate(FULL_RS):
                rs_data = rs_results[rs_idx][clf_name]
                y_true = rs_data["y_true"]
                probs = rs_data["probs"]
                pooled_y_true.extend(y_true)
                pooled_probs.extend(probs)
                if len(set(y_true)) >= 2:
                    per_rs_aucs.append(roc_auc_score(y_true, probs))

            if len(set(pooled_y_true)) >= 2:
                pooled_auc = roc_auc_score(pooled_y_true, pooled_probs)
            else:
                pooled_auc = np.nan

            mean_rs_auc = np.mean(per_rs_aucs) if per_rs_aucs else np.nan
            std_rs_auc = np.std(per_rs_aucs, ddof=1) if len(per_rs_aucs) > 1 else np.nan

            metrics.append(
                {
                    "K": k,
                    "classifier": clf_name,
                    "pooled_auc": pooled_auc,
                    "mean_rs_auc": mean_rs_auc,
                    "std_rs_auc": std_rs_auc,
                }
            )
    return pd.DataFrame(metrics)


def save_case_probability_workbooks(all_rs_metrics_by_k, K_grid, classifiers, out_dir, bw=5):
    for clf_name in classifiers:
        out_path = os.path.join(
            out_dir, f"case_probabilities_binWidth_{bw}_10case_external_{clf_name}.xlsx"
        )
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for rs_idx, rs in enumerate(FULL_RS):
                rs_key = f"rs_{rs}"
                ref_k = K_grid[0]
                ref_data = all_rs_metrics_by_k[ref_k][rs_idx][clf_name]

                sheet_df = pd.DataFrame(
                    {
                        "case": ref_data["cases"],
                        "y_true": ref_data["y_true"],
                    }
                )

                for k in K_grid:
                    sheet_df[f"prob_K{k}"] = all_rs_metrics_by_k[k][rs_idx][clf_name]["probs"]

                sheet_df.to_excel(writer, sheet_name=rs_key, index=False)
        print(f"    Saved 10-case Workbook: {out_path}")


def generate_heatmap(
    df_metrics,
    K_grid,
    classifiers,
    out_dir,
    filename="heatmap_10case_external.png",
    metric_col="pooled_auc",
    title="Pooled AUC on Unseen 10 Cases by K and Model",
    colorbar_label="Pooled AUC",
):
    data = []
    annot = []

    for k in K_grid:
        row_vals = []
        row_annots = []
        for clf_name in classifiers:
            row = df_metrics[(df_metrics["K"] == k) & (df_metrics["classifier"] == clf_name)].iloc[0]
            auc = row[metric_col]
            std = row["std_rs_auc"]
            row_vals.append(auc)
            if np.isnan(auc):
                row_annots.append("nan")
            else:
                row_annots.append("nan" if np.isnan(std) else f"{auc:.3f}\n±{std:.3f}")
        data.append(row_vals)
        annot.append(row_annots)

    data = np.array(data, dtype=float)
    annot = np.array(annot)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.70, vmax=1.00)

    for i in range(len(K_grid)):
        for j in range(len(classifiers)):
            value = data[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                annot[i, j],
                ha="center",
                va="center",
                fontsize=9,
                color="black" if value < 0.81 else "white",
            )

    ax.set_xticks(np.arange(len(classifiers)))
    ax.set_yticks(np.arange(len(K_grid)))
    ax.set_xticklabels(classifiers, rotation=15)
    ax.set_yticklabels(K_grid)
    ax.set_xlabel("Classifier Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("K (Stable Feature Count)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=11)

    ax.set_xticks(np.arange(len(classifiers) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(K_grid) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    fig.tight_layout()
    heatmap_path = os.path.join(out_dir, filename)
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"    Saved 10-case Heatmap: {heatmap_path}")


def load_binwidth_data(file_path):
    df = pd.read_csv(file_path)
    case_col = None
    for col in ["CaseNumber", "CaseID"]:
        if col in df.columns:
            case_col = col
            break
    if case_col is None:
        raise ValueError(f"No CaseNumber or CaseID column found in {file_path}")
    if "Label" not in df.columns:
        raise ValueError(f"No Label column found in {file_path}")
    cases = df[case_col].values.astype(str)
    y = df["Label"].values.astype(int)
    X = df.drop(columns=[case_col, "Label"])
    return X, y, cases


def resolve_test_csv(csv_base_dir, bw, explicit_test_csv=None):
    candidates = []
    if explicit_test_csv is not None:
        candidates.append(Path(explicit_test_csv))

    candidates.extend(
        [
            Path(f"Cine_output_20260606_10cases_binWidth_{bw}.csv"),
            Path(f"sampled_cases_10_binWidth_{bw}.csv"),
            Path("sampled_cases_10.csv"),
            Path(f"Cine_output_20260606_10case_binWidth_{bw}.csv"),
        ]
    )

    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = csv_base_dir / candidate
        if candidate.exists():
            return candidate
    return None


def save_feature_lists(df_stability, K_grid, out_dir):
    for k in K_grid:
        features = df_stability["feature_name"].head(k).tolist()
        pd.DataFrame({"feature_name": features}).to_csv(
            os.path.join(out_dir, f"top{k}_features_10case_external.csv"), index=False
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate high-frequency features on the unseen 10-case holdout set.")
    parser.add_argument("--smoke-test", action="store_true", help="Run only on r_0.7 SVM at binWidth=5.")
    parser.add_argument("--base-dir", type=Path, default=Path("/home/ser/test"), help="GA result base directory.")
    parser.add_argument("--csv-base-dir", type=Path, default=Path("/home/ser/test"), help="Directory containing 70-case and 10-case CSVs.")
    parser.add_argument("--output-base-dir", type=Path, default=Path("/home/ser/test/10case_result"), help="Output directory.")
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Explicit 10-case CSV path. If omitted, auto-detect common 10-case filenames.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    csv_base_dir = args.csv_base_dir
    output_base_dir = args.output_base_dir

    if args.smoke_test:
        conditions = [("r_0.7 SVM", 5)]
    else:
        dirs = ["r_0.7 RF", "r_0.7 SVM", "r_0.7 xgboost", "r_0.9 RF", "r_0.9 SVM", "r_0.9 xgboost"]
        bin_widths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        conditions = [(d, bw) for d in dirs for bw in bin_widths]

    for dir_name, bw in conditions:
        print("=" * 80)
        print(f"Processing 10-case external condition: {dir_name} at binWidth={bw}")
        print("=" * 80)

        rest = dir_name.replace("r_", "")
        r_thresh_str, ga_clf = rest.split(" ")
        r_thresh = float(r_thresh_str)

        json_path = base_dir / dir_name / "result" / f"ga_binWidth_{bw}" / "results_detail.json"
        train_csv_path = csv_base_dir / f"Cine_output_20260606_70cases_binWidth_{bw}.csv"
        test_csv_path = resolve_test_csv(csv_base_dir, bw, args.test_csv)
        out_dir = output_base_dir / f"{r_thresh_str}_{ga_clf}_bw{bw}"

        if not json_path.exists():
            print(f"Warning: JSON file not found: {json_path}. Skipping.")
            continue
        if not train_csv_path.exists():
            print(f"Warning: 70-case CSV not found: {train_csv_path}. Skipping.")
            continue
        if test_csv_path is None:
            print(f"Warning: 10-case CSV not found. Skipping.")
            continue
        if (out_dir / "final_comparison.csv").exists():
            mean_heatmap_path = out_dir / f"heatmap_10case_external_mean_auc_{r_thresh_str}_{ga_clf}_bw{bw}.png"
            if mean_heatmap_path.exists():
                print(f"Condition {dir_name} at binWidth={bw} already completed. Skipping.")
            else:
                os.makedirs(out_dir, exist_ok=True)
                df_metrics = pd.read_csv(out_dir / "final_comparison.csv")
                K_grid = sorted(df_metrics["K"].unique().tolist())
                classifiers = ["SoftVote-3", "SVM", "RandomForest", "xgboost"]
                print("Condition already completed; regenerating mean AUC heatmap...")
                generate_heatmap(
                    df_metrics,
                    K_grid,
                    classifiers,
                    out_dir,
                    filename=mean_heatmap_path.name,
                    metric_col="mean_rs_auc",
                    title="Mean AUC on Unseen 10 Cases by K and Model",
                    colorbar_label="Mean AUC",
                )
            continue

        os.makedirs(out_dir, exist_ok=True)

        print("Step 1: Getting sorted high-frequency feature ranking from 70-case GA results...")
        ranking = get_condition_feature_ranking(json_path, train_csv_path, r_thresh)
        print(f"Total unique features discovered: {len(ranking)}")

        if len(ranking) == 0:
            print("Warning: No features discovered. Skipping.")
            continue

        print("Step 2: Loading 70-case training data and unseen 10-case test data...")
        X_train, y_train, cases_train = load_binwidth_data(train_csv_path)
        X_test, y_test, cases_test = load_binwidth_data(test_csv_path)

        K_grid = [k for k in [5, 10, 15, 20, 30, 40, 50] if k <= len(ranking)]
        print(f"Using K_grid: {K_grid}")
        print(f"Test CSV: {test_csv_path}")

        if len(K_grid) == 0:
            print("Warning: K_grid is empty after capping. Skipping.")
            continue

        classifiers = ["SoftVote-3", "SVM", "RandomForest", "xgboost"]
        clfs = {
            "SVM": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42),
            "xgboost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=1,
            ),
        }

        print("Step 3: Training on 70 cases and predicting the unseen 10 cases...")
        all_rs_metrics_by_k = {}
        df_stability = pd.DataFrame([{"feature_name": feat, "selection_count": count} for feat, count in ranking])

        for k in K_grid:
            top_k_features = df_stability["feature_name"].head(k).tolist()
            rs_pruned_raw = Parallel(n_jobs=-1)(
                delayed(run_external_prediction)(
                    X_train,
                    y_train,
                    cases_train,
                    X_test,
                    y_test,
                    cases_test,
                    rs,
                    top_k_features,
                    clfs,
                )
                for rs in FULL_RS
            )
            all_rs_metrics_by_k[k] = aggregate_external_ensembles(rs_pruned_raw)

        print("Step 4: Computing external 10-case aggregate metrics...")
        df_metrics = compute_external_metrics_df(all_rs_metrics_by_k, K_grid, classifiers)
        df_metrics.to_csv(out_dir / "final_comparison.csv", index=False)

        print("Step 5: Saving 10-case probability workbooks...")
        save_case_probability_workbooks(all_rs_metrics_by_k, K_grid, classifiers, out_dir, bw=bw)

        print("Step 6: Generating 10-case heatmaps...")
        generate_heatmap(
            df_metrics,
            K_grid,
            classifiers,
            out_dir,
            filename=f"heatmap_10case_external_pooled_auc_{r_thresh_str}_{ga_clf}_bw{bw}.png",
            metric_col="pooled_auc",
            title="Pooled AUC on Unseen 10 Cases by K and Model",
            colorbar_label="Pooled AUC",
        )
        generate_heatmap(
            df_metrics,
            K_grid,
            classifiers,
            out_dir,
            filename=f"heatmap_10case_external_mean_auc_{r_thresh_str}_{ga_clf}_bw{bw}.png",
            metric_col="mean_rs_auc",
            title="Mean AUC on Unseen 10 Cases by K and Model",
            colorbar_label="Mean AUC",
        )

        print("Step 7: Saving feature lists...")
        save_feature_lists(df_stability, K_grid, out_dir)

        checkpoint_path = out_dir / "oof_probabilities_10case_external.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(all_rs_metrics_by_k, f)
        print(f"Step 8: Saved raw 10-case predictions checkpoint to {checkpoint_path}")
        print(f"Condition {dir_name} at binWidth={bw} complete!\n")


if __name__ == "__main__":
    main()
