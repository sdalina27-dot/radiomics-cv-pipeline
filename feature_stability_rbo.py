"""
Feature selection stability analysis across 24 folds for three feature selection methods
using Rank-Biased Overlap (RBO).

Methods compared (per BW):
    - mi_simple_r0.7      (Mutual Information, single-pass)
    - original_only_r0.7  (GA + SVM, ORIGINAL_ONLY=True)
    - original_only_r0.7_RF (GA + RF, ORIGINAL_ONLY=True)

Each method exposes 3 outer random states (42, 123, 456) x 8 outer folds = 24 folds
per BW.  Four shared bin widths are evaluated: BW_10, BW_20, BW_30, BW_40.

For every (method, BW) pair we compute the full pairwise RBO matrix among the 24
folds (276 unique pairs) using p = RBO_P .
The mean of the upper-triangular entries is the internal stability score, and the
distribution of pairwise values is summarised with a box plot.

Outputs (under WORKSPACE_DIR/stability/):
    - rbo_heatmap_{method}_BW{bw}.png          per-method/BW RBO heatmap
    - jaccard_heatmap_{method}_BW{bw}.png       per-method/BW Jaccard heatmap
    - rbo_heatmaps_combined.png                4×3 RBO heatmap grid
    - jaccard_heatmaps_combined.png            4×3 Jaccard heatmap grid
    - rbo_boxplot_BW{bw}.png                    per-BW 3-method RBO boxplot
    - jaccard_boxplot_BW{bw}.png               per-BW 3-method Jaccard boxplot
    - rbo_boxplot_all.png                       combined RBO boxplots (12 boxes)
    - jaccard_boxplot_all.png                   combined Jaccard boxplots
    - stability_scores.csv                      long-format stats incl. Top-20 Jaccard

Reference: Webber, Moffat & Zobel (2010), "A Similarity Measure for Indefinite
Rankings", ACM TOIS 28(4), Article 20.

RBO is computed using the `rbo` package (pip install rbo) — specifically its
extrapolated variant `RankingSimilarity.rbo_ext(p)`, which is Eq. (32) in the
paper and the recommended point estimate when the two rankings have different
lengths (which is the case here: MI Simple yields ~36 features/fold while the
GA methods yield ~70–77).

Top-20 Jaccard overlap is reported in addition to RBO as a complementary
"top-of-the-list" metric that does not depend on rank order within the prefix:
    J@k = |top_k(S) ∩ top_k(T)| / |top_k(S) ∪ top_k(T)|
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.axes import Axes

# External RBO package: https://github.com/dlukes/rbo  (PyPI: rbo 0.1.3)
# Provides RankingSimilarity.rbo_ext(p) — extrapolated RBO for un-even rankings.
import rbo as rbo_lib

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PIPELINE_DIR = Path("/home/ser/pipeline")
WORKSPACE_DIR = PIPELINE_DIR / "workspace"

METHODS = {
    "mi_multirs_r0.7_StandardScaler": "MI StandardScaler",
    "OO_r0.7_StandardScaler_top5_NoRdundancyInFitness": "GA-SVM StandardScaler",
    "original_only_r0.7_RF": "GA-RF",
}
METHOD_ORDER = list(METHODS.keys())          # stable iteration order
METHOD_LABELS = {k: METHODS[k] for k in METHOD_ORDER}

COMMON_BWS = [10, 20, 30, 40]
COMMON_RS = [42, 123, 456]
N_FOLDS = 8

RBO_P = 0.8                                  # depth parameter: 前 5 名佔 1-RBO_P^5
TOP_K = 10                                    # report overlap at this cutoff too

OUTPUT_DIR = WORKSPACE_DIR / "stability" / f"{list(METHODS.keys())[0]}, {list(METHODS.keys())[1]}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# RBO via the external `rbo` package (extrapolated variant, Eq. 32)
# --------------------------------------------------------------------------- #

def rbo(S: list, T: list, p: float = RBO_P) -> float:
    """Extrapolated RBO between two ranked feature lists using the `rbo` package.

    `RankingSimilarity.rbo_ext(p)` is the recommended single-number point
    estimate when the two lists differ in length (Eq. 32 of Webber et al. 2010):
    it extrapolates the unobserved tail symmetrically and is bounded in [0, 1].
    """
    if not S or not T:
        return 0.0
    # The package asserts no duplicate entries; our CSVs already satisfy this.
    rs = rbo_lib.RankingSimilarity(S, T)
    return float(rs.rbo_ext(p=p))


def topk_jaccard(S: list, T: list, k: int = TOP_K) -> float:
    """Top-K Jaccard overlap: |top_k(S) ∩ top_k(T)| / |top_k(S) ∪ top_k(T)|.

    Unlike RBO it ignores the ordering within the prefix and so measures a
    cruder "set overlap at the top of the list".  We report it next to RBO so
    the reader can disentangle order-stability from set-stability.  When either
    list is shorter than k, only the elements actually present in the prefix are
    used, which keeps the metric well-defined (union is bounded by the observed
    prefix sizes, not by k itself)."""
    top_s = set(S[:k])
    top_t = set(T[:k])
    if not top_s or not top_t:
        return 0.0
    return len(top_s & top_t) / len(top_s | top_t)


def compute_jaccard_mean(rankings: list[list[str]], k: int = TOP_K) -> tuple[float, float, np.ndarray]:
    """Mean / std of pairwise top-K Jaccard overlap over the upper triangle.
    Returns (mean, std, flat_pair_values)."""
    n = len(rankings)
    vals = []
    for i, j in itertools.combinations(range(n), 2):
        vals.append(topk_jaccard(rankings[i], rankings[j], k=k))
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)), arr


# --------------------------------------------------------------------------- #
# Fold loaders
# --------------------------------------------------------------------------- #

def load_fold_ranking(csv_path: Path) -> list[str]:
    """Read feature_frequencies.csv and return the features in ranked order.

    Three CSV schemas exist in the workspace:
        * original_only, original_only_RF: feature, frequency, rank, raw_count
        * mi_simple:                       feature, frequency, rank
    All three carry an explicit 1-based `rank` column derived from the
    `frequency` column inside the producing script.  We sort by `rank` in ALL
    methods (not by `raw_count`) so the comparison is on a single column that is
    defined identically across methods.  This is appropriate here because
    `original_only` and `original_only_RF` share identical GA_N_RS (16) /
    GA_N_FOLDS (5) / GA_INNER_FOLDS (3) — see workspace/{exp}/config.txt — so the
    `frequency` column and the `rank` it implies are directly comparable across
    these methods.  (Across methods with different GA sampling one would still
    prefer `raw_count`, which is why it is retained for reference.)
    """
    df = pd.read_csv(csv_path)
    if "rank" not in df.columns:
        raise ValueError(f"{csv_path} is missing required 'rank' column")
    df = df.sort_values("rank", ascending=True, kind="stable")
    return df["feature"].tolist()


def enumerate_folds(method_dir: Path, bw: int) -> list[tuple[int, int, Path]]:
    """Return list of (outer_rs, fold_idx, feature_frequencies.csv path) for the
    requested BW, restricted to the COMMON_RS and N_FOLDS folds used by every
    method."""
    base = method_dir / "artifacts" / f"BW_{bw}"
    fold_paths: list[tuple[int, int, Path]] = []
    for rs in COMMON_RS:
        for f in range(N_FOLDS):
            p = base / f"outer_rs_{rs}" / f"fold_{f}" / "feature_frequencies.csv"
            if p.exists():
                fold_paths.append((rs, f, p))
    return fold_paths


def fold_label(idx: int, rs: int, fold: int) -> str:
    """Human-readable fold label like 'R42 F3'."""
    return f"R{rs}\nF{fold}"


# --------------------------------------------------------------------------- #
# Pairwise RBO matrix and stability score
# --------------------------------------------------------------------------- #

def compute_rbo_matrix(rankings: list[list[str]], p: float = RBO_P) -> np.ndarray:
    """Return a symmetric N×N matrix of RBO(p) values, diagonal = 1."""
    n = len(rankings)
    M = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        s = rbo(rankings[i], rankings[j], p=p)
        M[i, j] = s
        M[j, i] = s
    return M


def compute_jaccard_matrix(rankings: list[list[str]], k: int = TOP_K) -> np.ndarray:
    """Return a symmetric N×N matrix of top-K Jaccard overlaps, diagonal = 1.

    The diagonal is set to 1.0 (a ranking trivially overlaps itself perfectly
    at any prefix length).  Off-diagonals use the same `topk_jaccard` function
    used for the summary CSV so the heatmap and the box plot tell the same story.
    """
    n = len(rankings)
    M = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        s = topk_jaccard(rankings[i], rankings[j], k=k)
        M[i, j] = s
        M[j, i] = s
    return M


def stability_score(M: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Mean / std of upper-triangular entries of a pairwise matrix (= internal
    stability following Kuncheva 2004).  Returns (mean, std, flat_pair_values)."""
    iu = np.triu_indices(M.shape[0], k=1)
    vals = M[iu]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), vals


def boxplot_stats_match_matplotlib(values: np.ndarray | list[float]) -> dict:
    """Compute the boxplot statistics EXACTLY as `matplotlib.pyplot.boxplot`
    draws them (whis=1.5, default), so that Q1, Q3, whisker endpoints reported
    in the CSV are byte-for-byte the same numbers the figure shows.

    This is what the user asked for: at small TOP_K the Jaccard distribution can
    be very tall-spike (e.g. lots of 1.0s) and the visual box collapses against
    the whisker / fence, making the IQR invisible in the figure.  Reporting the
    underlying numbers lets the reader recover that information numerically.

    Returns a dict with keys:
        q1, median, q3, iqr, whisker_lo, whisker_hi, n_outliers_lo, n_outliers_hi
    Whiskers follow the matplotlib convention: the most extreme data point that
    is still within 1.5×IQR below Q1 / above Q3.  When no point sits within the
    fence on a given side (i.e. all values beyond the hinge are outliers), the
    whisker falls back to the hinge itself (Q1 / Q3).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"q1": float("nan"), "median": float("nan"), "q3": float("nan"),
                "iqr": float("nan"), "whisker_lo": float("nan"),
                "whisker_hi": float("nan"),
                "n_outliers_lo": 0, "n_outliers_hi": 0}
    # matplotlib default uses linear interpolation between points (numpy default)
    q1 = float(np.percentile(arr, 25, method="linear"))
    med = float(np.percentile(arr, 50, method="linear"))
    q3 = float(np.percentile(arr, 75, method="linear"))
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    within_lo = arr[arr >= lo_fence]
    within_hi = arr[arr <= hi_fence]
    whisker_lo = float(np.min(within_lo)) if within_lo.size else q1
    whisker_hi = float(np.max(within_hi)) if within_hi.size else q3
    n_out_lo = int(np.sum(arr < whisker_lo))
    n_out_hi = int(np.sum(arr > whisker_hi))
    return {"q1": q1, "median": med, "q3": q3, "iqr": iqr,
            "whisker_lo": whisker_lo, "whisker_hi": whisker_hi,
            "n_outliers_lo": n_out_lo, "n_outliers_hi": n_out_hi}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

METHOD_COLORS = {
    f"{list(METHODS.keys())[0]}": "#1f77b4",
    f"{list(METHODS.keys())[1]}": "#2ca02c",
    f"{list(METHODS.keys())[2]}": "#d62728",
}


def plot_heatmap(M: np.ndarray, fold_meta: list[tuple[int, int, Path]],
                 method_key: str, bw: int, ax: "Axes",
                 metric_name: str | None = None,
                 aux_mean: float | None = None,
                 aux_label: str = "J@20"):  # type: ignore[type-arg]
    """Draw one 24×24 stability heatmap on `ax` and return the image handle.

    `metric_name` controls the title's primary metric label.  When `aux_mean`
    is provided it is appended as a secondary statistic in the title (used to
    surface the complementary J@20 alongside RBO, or vice-versa).
    """
    if metric_name is None:
        metric_name = f"RBO({RBO_P})"
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(fold_meta)))
    ax.set_yticks(range(len(fold_meta)))
    labels = [fold_label(i, rs, f) for i, (rs, f, _) in enumerate(fold_meta)]
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticklabels(labels, fontsize=5)
    # block separators between RS blocks for readability
    for cut in [8, 16]:
        ax.axhline(cut - 0.5, color="white", linewidth=0.8)
        ax.axvline(cut - 0.5, color="white", linewidth=0.8)
    mean, std, _ = stability_score(M)
    title = (f"{METHOD_LABELS[method_key]}  BW={bw}\n"
             f"{metric_name}  mean={mean:.3f}  sd={std:.3f}")
    if aux_mean is not None:
        title += f"  |  {aux_label}={aux_mean:.3f}"
    ax.set_title(title, fontsize=9)
    return im


def main() -> None:
    summary_rows: list[dict] = []
    boxplot_data: list[dict] = []

    # Per (method, BW): heatmap matrix
    for method_key in METHOD_ORDER:
        method_dir = WORKSPACE_DIR / method_key
        # Per-BW figures (heatmaps) — one figure with 4 subplots per method,
        # plus one combined grid for side-by-side comparison.
        for bw in COMMON_BWS:
            folds = enumerate_folds(method_dir, bw)
            if not folds:
                print(f"[WARN] no folds for {method_key} BW={bw}, skipping")
                continue
            rankings = [load_fold_ranking(p) for _, _, p in folds]
            M = compute_rbo_matrix(rankings, p=RBO_P)
            mean, std, flat = stability_score(M)
            # Top-20 Jaccard overlap (independent of intra-prefix order)
            jac_mean, jac_std, jac_flat = compute_jaccard_mean(rankings, k=TOP_K)
            # Pairwise Jaccard heatmap matrix (drawn separately below)
            M_jac = compute_jaccard_matrix(rankings, k=TOP_K)
            # matplotlib-faithful boxplot stats (whis=1.5) for the CSV so that
            # missing / collapsed boxes at small TOP_K can be read off numerically
            rbo_bp = boxplot_stats_match_matplotlib(flat)
            jac_bp = boxplot_stats_match_matplotlib(jac_flat)

            # Per-method RBO heatmap with one subplot for this BW.
            fig, ax = plt.subplots(figsize=(7.5, 7))
            im = plot_heatmap(M, folds, method_key, bw, ax,
                              aux_mean=jac_mean, aux_label="J@20")
            fig.colorbar(im, ax=ax, fraction=0.046, label=f"RBO(p={RBO_P})")
            fig.tight_layout()
            out = OUTPUT_DIR / f"rbo_heatmap_{method_key}_BW{bw}.png"
            fig.savefig(out, dpi=160)
            plt.close(fig)

            # Per-method Jaccard heatmap (same layout, different matrix).
            fig, ax = plt.subplots(figsize=(7.5, 7))
            im = plot_heatmap(M_jac, folds, method_key, bw, ax,
                              metric_name=f"Jaccard@{TOP_K}", aux_mean=mean,
                              aux_label="RBO")
            fig.colorbar(im, ax=ax, fraction=0.046,
                         label=f"Top-{TOP_K} Jaccard overlap")
            fig.tight_layout()
            out = OUTPUT_DIR / f"jaccard_heatmap_{method_key}_BW{bw}.png"
            fig.savefig(out, dpi=160)
            plt.close(fig)
            print(f"[OK] RBO {out.name.replace('jaccard','rbo')}  "
                  f"RBO mean={mean:.3f} sd={std:.3f} | "
                  f"J@{TOP_K} mean={jac_mean:.3f} sd={jac_std:.3f}")
            print(f"[OK] {out.name}  J@{TOP_K} mean={jac_mean:.3f} sd={jac_std:.3f}")

            summary_rows.append({
                "method": method_key,
                "method_label": METHOD_LABELS[method_key],
                "bw": bw,
                "n_folds": len(folds),
                "rbo_p": RBO_P,
                "top_k_jaccard": TOP_K,
                # ---- RBO pairwise distribution ----
                "mean_rbo": mean,
                "std_rbo": std,
                "min_rbo": float(np.min(flat)),
                "max_rbo": float(np.max(flat)),
                "median_rbo": float(np.median(flat)),
                "q1_rbo": rbo_bp["q1"],
                "q3_rbo": rbo_bp["q3"],
                "whisker_low_rbo": rbo_bp["whisker_lo"],
                "whisker_high_rbo": rbo_bp["whisker_hi"],
                # ---- Top-K Jaccard pairwise distribution ----
                # these are the figures that get visually degenerate at small
                # TOP_K (box collapses against the whisker / fence), so we make
                # the Q1 / Q3 / whisker endpoints explicit in the CSV.
                "mean_topk_jaccard": jac_mean,
                "std_topk_jaccard": jac_std,
                "min_topk_jaccard": float(np.min(jac_flat)),
                "max_topk_jaccard": float(np.max(jac_flat)),
                "median_topk_jaccard": float(np.median(jac_flat)),
                "q1_topk_jaccard": jac_bp["q1"],
                "q3_topk_jaccard": jac_bp["q3"],
                "whisker_low_topk_jaccard": jac_bp["whisker_lo"],
                "whisker_high_topk_jaccard": jac_bp["whisker_hi"],
                "n_pairs": len(flat),
            })
            boxplot_data.append({
                "method_key": method_key,
                "method_label": METHOD_LABELS[method_key],
                "bw": bw,
                "pair_values": flat,
                "pair_jaccard": jac_flat,
            })

    # Combined heatmap grids: rows = methods, cols = BWs.
    # We produce one combined grid per metric (RBO and Jaccard) so the two
    # stability facets can be eyeballed side-by-side at the same scale.
    n_rows = len(METHOD_ORDER)
    n_cols = len(COMMON_BWS)

    def _draw_combined_grid(metric: str, cbar_label: str, out_name: str,
                            suptitle: str) -> None:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows),
                                 squeeze=False)
        for r, method_key in enumerate(METHOD_ORDER):
            method_dir = WORKSPACE_DIR / method_key
            for c, bw in enumerate(COMMON_BWS):
                ax = axes[r][c]
                folds = enumerate_folds(method_dir, bw)
                if not folds:
                    ax.set_axis_off()
                    ax.set_title(f"{METHOD_LABELS[method_key]}  BW={bw}\n(no data)",
                                 fontsize=9)
                    continue
                rankings = [load_fold_ranking(p) for _, _, p in folds]
                if metric == "rbo":
                    M = compute_rbo_matrix(rankings, p=RBO_P)
                    jac_mean, _, _ = compute_jaccard_mean(rankings, k=TOP_K)
                    im = plot_heatmap(M, folds, method_key, bw, ax,
                                      aux_mean=jac_mean, aux_label="J@20")
                else:
                    M = compute_jaccard_matrix(rankings, k=TOP_K)
                    rbo_mean, _, _ = stability_score(
                        compute_rbo_matrix(rankings, p=RBO_P))
                    im = plot_heatmap(M, folds, method_key, bw, ax,
                                      metric_name=f"Jaccard@{TOP_K}",
                                      aux_mean=rbo_mean, aux_label="RBO")
                if c == n_cols - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
        fig.suptitle(suptitle, fontsize=12, y=1.02)
        fig.tight_layout()
        out = OUTPUT_DIR / out_name
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out.name}")

    _draw_combined_grid(
        "rbo", f"RBO(p={RBO_P})", "rbo_heatmaps_combined.png",
        f"Pairwise RBO (p={RBO_P}) heatmaps across 24 folds (3 RS × 8 folds)",
    )
    _draw_combined_grid(
        "jaccard", f"Top-{TOP_K} Jaccard overlap",
        f"jaccard_heatmaps_combined.png",
        f"Pairwise top-{TOP_K} Jaccard overlap across 24 folds (3 RS × 8 folds)",
    )

    # Box plots: one per BW comparing three methods + one combined panel.
    # We emit one set of figures per metric (RBO and Jaccard) so both stability
    # facets are visualised symmetrically.

    def _draw_bw_boxplots(metric: str, ylab: str, fname: str) -> None:
        """Per-BW boxplots comparing the three methods for one metric."""
        value_key = "pair_values" if metric == "rbo" else "pair_jaccard"
        for bw in COMMON_BWS:
            bw_data = [d for d in boxplot_data if d["bw"] == bw]
            fig, ax = plt.subplots(figsize=(6, 5))
            positions = list(range(1, len(bw_data) + 1))
            bp = ax.boxplot(
                [d[value_key] for d in bw_data],
                positions=positions, widths=0.55, patch_artist=True,
                showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                           "markeredgecolor": "black", "markersize": 5},
            )
            for patch, d in zip(bp["boxes"], bw_data):
                patch.set_facecolor(METHOD_COLORS[d["method_key"]])
                patch.set_alpha(0.65)
            ax.set_xticks(positions)
            ax.set_xticklabels([d["method_label"] for d in bw_data], fontsize=9)
            ax.set_ylabel(ylab)
            ax.set_ylim(0, 1.05)
            metric_disp = f"RBO (p={RBO_P})" if metric == "rbo" else f"Top-{TOP_K} Jaccard"
            ax.set_title(f"Feature-selection stability across 24 folds  (BW={bw})\n"
                         f"{metric_disp} — 3 RS × 8 folds, n=276 pairs/method",
                         fontsize=10)
            # annotate mean
            for pos, d in zip(positions, bw_data):
                m = float(np.mean(d[value_key]))
                ax.text(pos, m + 0.01, f"μ={m:.3f}", ha="center", fontsize=8)
            # legend for mean marker
            legend_mean = mpatches.Patch(color="none", label="◇ = mean")
            ax.legend(handles=[legend_mean], loc="lower right", fontsize=8)
            fig.tight_layout()
            out = OUTPUT_DIR / fname.format(bw=bw)
            fig.savefig(out, dpi=160)
            plt.close(fig)
            print(f"[OK] {out.name}")

    _draw_bw_boxplots("rbo", f"Pairwise RBO (p={RBO_P})", f"rbo_boxplot_BW{{bw}}.png")
    _draw_bw_boxplots("jaccard", f"Pairwise top-{TOP_K} Jaccard",
                      f"jaccard_boxplot_BW{{bw}}.png")

    def _draw_combined_boxplot(metric: str, ylab: str, fname: str,
                               subtitle: str) -> None:
        """Combined boxplot: BW on x-axis, colored by method, grouped."""
        value_key = "pair_values" if metric == "rbo" else "pair_jaccard"
        fig, ax = plt.subplots(figsize=(9, 5.5))
        positions = []
        pos = 1
        width = 0.25
        for _mi, method_key in enumerate(METHOD_ORDER):
            sub_data = [d for d in boxplot_data if d["method_key"] == method_key]
            if not sub_data:
                continue
            for bi, bw in enumerate(COMMON_BWS):
                d = next((x for x in sub_data if x["bw"] == bw), None)
                if d is None:
                    continue
                positions.append((pos, d))
                pos += 1
            pos += 0.5                       # gap between method blocks

        bp = ax.boxplot(
            [d[value_key] for _, d in positions],
            positions=[p for p, _ in positions], widths=width, patch_artist=True,
            showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                       "markeredgecolor": "black", "markersize": 5},
        )
        for patch, (p, d) in zip(bp["boxes"], positions):
            patch.set_facecolor(METHOD_COLORS[d["method_key"]])
            patch.set_alpha(0.65)
        ax.set_xticks([p for p, _ in positions])
        ax.set_xticklabels([f"BW{d['bw']}" for _, d in positions],
                           fontsize=8, rotation=45)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{subtitle}\n"
                     f"({value_key}: lower = more fold-sensitive; "
                     f"276 pairs per (method, BW))", fontsize=10)
        # group separators between methods
        last_method = None
        sep_x = []
        for p, d in positions:
            if last_method is not None and d["method_key"] != last_method:
                sep_x.append(p - 0.5)
            last_method = d["method_key"]
        for x in sep_x:
            ax.axvline(x, color="grey", linewidth=0.6, linestyle="--")
        # color legend
        color_handles = [mpatches.Patch(color=METHOD_COLORS[k], label=v,
                                        alpha=0.65)
                         for k, v in METHOD_LABELS.items()]
        color_handles.append(mpatches.Patch(color="none", label="◇ = mean"))
        ax.legend(handles=color_handles, loc="lower left", fontsize=8)
        fig.tight_layout()
        out = OUTPUT_DIR / fname
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[OK] {out.name}")

    _draw_combined_boxplot("rbo", f"Pairwise RBO (p={RBO_P})",
                          "rbo_boxplot_all.png",
                          "Internal feature-selection stability across 24 folds")
    _draw_combined_boxplot("jaccard", f"Pairwise top-{TOP_K} Jaccard",
                           "jaccard_boxplot_all.png",
                           f"Internal feature-selection stability across 24 folds "
                           f"(top-{TOP_K} set overlap)")

    # Save summary csv
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "stability_scores.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] {summary_path.name}")
    print("\n=== Stability summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
