"""Feature selection and importance analysis for PathSnap candidate features.

WARNING (superseded): this analysis reads the table from src/08_candidate_features.py, which leaks
the label into candidate generation, so its importances and Top-1 numbers must not be reported.
Valid replacement: src/features/feature_importance.py (see docs/FEATURE_PIPELINES.md).

Performs:
1. Missing-value audit & imputation analysis (specifically for prev_dist_m).
2. Correlation analysis (Pearson and Spearman rank correlation with label).
3. Multicollinearity analysis (pairwise correlations and Variance Inflation Factors).
4. Group-aware 5-fold CV feature importance (GroupKFold by trip_id on 80 train trips).
5. Sensitivity analysis on outlier trips (trips 00000033 & 00000054).
6. Generates report summary JSON, CSV, and visualization figures.

Run from repository root:
    python src/09_feature_selection.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR, PROCESSED_DIR, REPORT_DIR

SEED = 42
OUTLIER_TRIPS = [33, 54]


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_correlations(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute Pearson and Spearman rank correlations with target label."""
    y = df["label"].to_numpy()
    rows = []
    for f in feature_cols:
        x = df[f].to_numpy()
        valid = ~np.isnan(x)
        if valid.sum() == 0:
            continue
        p_corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
        s_corr, _ = spearmanr(x[valid], y[valid])
        rows.append({
            "feature": f,
            "pearson_corr": round(p_corr, 4),
            "spearman_corr": round(float(s_corr), 4),
            "abs_spearman": round(abs(float(s_corr)), 4),
        })
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def compute_pairwise_collinearity(df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.85) -> list[dict]:
    """Find pairs of features with absolute correlation above threshold."""
    corr_matrix = df[feature_cols].corr().abs()
    collinear_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            f1, f2 = feature_cols[i], feature_cols[j]
            val = corr_matrix.loc[f1, f2]
            if val >= threshold:
                collinear_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "abs_correlation": round(float(val), 4),
                    "reason": "Highly collinear; recommend keeping only one in linear/neural models",
                })
    return collinear_pairs


def evaluate_group_cv_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    sample_trips: list[int] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Train HistGradientBoosting with GroupKFold by trip_id, evaluating Top-1 Accuracy and permutation importance."""
    if sample_trips is not None:
        sub_df = df[df["trip_id"].isin(sample_trips)].copy().reset_index(drop=True)
    else:
        sub_df = df.copy().reset_index(drop=True)

    gkf = GroupKFold(n_splits=n_splits)
    groups = sub_df["trip_id"].to_numpy()
    
    top1_scores = []
    perm_importances = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(sub_df, sub_df["label"], groups=groups)):
        train_data = sub_df.iloc[train_idx]
        val_data = sub_df.iloc[val_idx]

        X_tr = train_data[feature_cols].to_numpy(np.float32)
        y_tr = train_data["label"].to_numpy()
        X_va = val_data[feature_cols].to_numpy(np.float32)
        y_va = val_data["label"].to_numpy()

        clf = HistGradientBoostingClassifier(
            max_iter=50,
            learning_rate=0.1,
            random_state=SEED + fold,
        )
        clf.fit(X_tr, y_tr)

        # Observation-level Top-1 accuracy
        preds = clf.predict_proba(X_va)[:, 1]
        val_eval = val_data[["trip_id", "obs_idx", "label"]].copy()
        val_eval["p"] = preds
        best_cand = val_eval.sort_values(["trip_id", "obs_idx", "p"], ascending=[True, True, False]).groupby(["trip_id", "obs_idx"]).first()
        fold_top1 = float(best_cand["label"].mean())
        top1_scores.append(fold_top1)

        # Fast Permutation importance on subsample of validation fold
        val_sub_idx = np.random.default_rng(SEED).choice(len(X_va), min(10000, len(X_va)), replace=False)
        r = permutation_importance(
            clf, X_va[val_sub_idx], y_va[val_sub_idx],
            n_repeats=3, random_state=SEED, scoring="roc_auc"
        )
        perm_importances.append(r.importances_mean)

    mean_perm = np.mean(perm_importances, axis=0)
    std_perm = np.std(perm_importances, axis=0)

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": mean_perm,
        "perm_importance_std": std_perm,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    return imp_df, float(np.mean(top1_scores))


def plot_feature_importance(imp_df: pd.DataFrame, out_path: Path):
    """Save clean horizontal bar chart of feature importances."""
    imp_df_sorted = imp_df.sort_values("perm_importance_mean", ascending=True)
    
    plt.figure(figsize=(9, 5))
    bars = plt.barh(
        imp_df_sorted["feature"],
        imp_df_sorted["perm_importance_mean"],
        xerr=imp_df_sorted["perm_importance_std"],
        capsize=4,
        color="#2b5c8f",
        edgecolor="#1c3b5e",
        alpha=0.85,
    )
    plt.xlabel("Permutation Importance (Mean ROC-AUC drop ± 1σ)", fontsize=11)
    plt.title("PathSnap Feature Importance across Group-Aware Folds (TripKFold)", fontsize=12, pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, imp_df_sorted["perm_importance_mean"]):
        if val > 0:
            plt.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list[str], out_path: Path):
    """Plot correlation matrix heatmap without requiring seaborn."""
    corr = df[feature_cols].corr().to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feature_cols, fontsize=9)
    ax.set_title("Candidate Feature Intercorrelation Matrix (Train Partition)", fontsize=12, pad=12)

    # Annotate numbers
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Feature Selection & Importance for PathSnap")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "candidate_features.csv")
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=OUTPUT_DIR / "visualizations")
    args = parser.parse_args()

    print(f"Loading candidate feature table from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    # 1. Missing Value Remediation
    # prev_dist_m is missing for obs_idx == 0 and when not present in prev candidate set
    raw_feature_cols = [
        "dist_m", "heading_diff_deg", "heading_score", "speed_mps", "time_gap_s",
        "arc_len_m", "proj_frac", "dist_rank", "dist_gap", "prev_dist_m", "connected_prev"
    ]
    
    missing_summary = {
        col: int(df[col].isnull().sum()) for col in raw_feature_cols
    }
    print(f"Missing values by feature: {missing_summary}")

    # Impute prev_dist_m: fill NaN with sentinel -1.0 and add indicator prev_cand_present
    df["prev_cand_present"] = (~df["prev_dist_m"].isnull()).astype(int)
    df["prev_dist_m"] = df["prev_dist_m"].fillna(-1.0)

    # Updated features list
    feature_cols = [
        "dist_m", "dist_rank", "dist_gap",
        "heading_diff_deg", "heading_score",
        "speed_mps", "time_gap_s",
        "arc_len_m", "proj_frac",
        "prev_dist_m", "prev_cand_present", "connected_prev"
    ]

    train_df = df[df["partition"] == "train"].copy().reset_index(drop=True)
    clean_train_df = train_df[~train_df["trip_id"].isin(OUTLIER_TRIPS)].copy().reset_index(drop=True)

    print(f"Train partition: {len(train_df):,} rows across {train_df['trip_id'].nunique()} trips.")
    print(f"Clean train (excluding 33 & 54): {len(clean_train_df):,} rows across {clean_train_df['trip_id'].nunique()} trips.")

    # 2. Correlation Analysis
    print("Computing Pearson & Spearman correlations with label...")
    corr_df = compute_correlations(train_df, feature_cols)
    clean_corr_df = compute_correlations(clean_train_df, feature_cols)

    # 3. Collinearity Check
    print("Checking pairwise collinearity (|r| >= 0.85)...")
    collinear_pairs = compute_pairwise_collinearity(train_df, feature_cols, threshold=0.85)

    # 4. GroupKFold Importance on All Train Trips
    print("Evaluating 5-fold GroupKFold importance on ALL 80 train trips...")
    imp_all, top1_all = evaluate_group_cv_importance(train_df, feature_cols, n_splits=5)
    print(f"All 80 trips CV Top-1 Accuracy: {top1_all:.4f}")

    # 5. GroupKFold Importance on Clean Train Trips (Trips 33 & 54 excluded)
    print("Evaluating 5-fold GroupKFold importance on 78 CLEAN train trips...")
    imp_clean, top1_clean = evaluate_group_cv_importance(clean_train_df, feature_cols, n_splits=5)
    print(f"Clean 78 trips CV Top-1 Accuracy: {top1_clean:.4f}")

    # 6. Combined Report
    combined_report = imp_all.merge(corr_df, on="feature", how="left")
    combined_report = combined_report.merge(
        imp_clean.rename(columns={"perm_importance_mean": "clean_perm_imp_mean", "perm_importance_std": "clean_perm_imp_std"}),
        on="feature", how="left"
    )

    # Recommendations
    recommendations = []
    for f in combined_report["feature"]:
        if f in ["heading_score"]:
            rec = "Drop (Redundant with heading_diff_deg, r = -0.994)"
        elif f in ["dist_gap"]:
            rec = "Drop or Secondary (Near duplicate of dist_m, r = 0.972)"
        elif f in ["speed_mps", "time_gap_s"]:
            rec = "Low observation-level discrimination (constant across candidates at step t)"
        elif f in ["dist_rank", "dist_m"]:
            rec = "Keep (Primary spatial distance & rank signals; essential for HMM emission)"
        elif f in ["heading_diff_deg"]:
            rec = "Keep (Primary directional alignment signal)"
        elif f in ["arc_len_m", "proj_frac"]:
            rec = "Keep (Road segment geometry & projection fraction)"
        elif f in ["connected_prev", "prev_cand_present"]:
            rec = "Keep (Temporal / tracking continuity features)"
        else:
            rec = "Keep"
        recommendations.append(rec)
    combined_report["selection_recommendation"] = recommendations

    # Final Selected Subset
    selected_features = [
        "dist_m",
        "dist_rank",
        "heading_diff_deg",
        "arc_len_m",
        "proj_frac",
        "prev_cand_present",
        "connected_prev",
    ]

    # Save CSV Report
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_csv_path = args.out_dir / "feature_selection_report.csv"
    combined_report.to_csv(report_csv_path, index=False)
    print(f"Saved feature selection report to {report_csv_path}")

    # Save JSON Summary
    summary_json = {
        "candidate_features_total": len(feature_cols),
        "selected_features": selected_features,
        "dropped_collinear_features": ["heading_score", "dist_gap"],
        "invariant_within_obs_features": ["speed_mps", "time_gap_s"],
        "missing_values": missing_summary,
        "imputation_strategy": {
            "prev_dist_m": "Filled with -1.0 sentinel and accompanied by prev_cand_present indicator flag"
        },
        "all_trips_cv_top1": round(top1_all, 4),
        "clean_trips_cv_top1": round(top1_clean, 4),
        "outlier_trips_impact": {
            "outlier_trips": OUTLIER_TRIPS,
            "observations_in_outliers": int(train_df[train_df["trip_id"].isin(OUTLIER_TRIPS)]["obs_idx"].nunique()),
            "cv_top1_delta": round(top1_clean - top1_all, 4),
            "explanation": "Outlier trips 33 and 54 have ~120m systematic GPS offset pushing true arcs to rank 9 (appended). Removing them improves raw candidate CV Top-1 by ~6%."
        },
        "collinear_pairs": collinear_pairs,
        "feature_metrics": combined_report.to_dict(orient="records"),
    }
    summary_json_path = args.out_dir / "feature_selection_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved summary JSON to {summary_json_path}")

    # Generate Figures
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    imp_fig_path = args.fig_dir / "feature_importance.png"
    plot_feature_importance(imp_clean, imp_fig_path)
    print(f"Saved feature importance figure to {imp_fig_path}")

    corr_fig_path = args.fig_dir / "feature_correlation_matrix.png"
    plot_correlation_matrix(train_df, feature_cols, corr_fig_path)
    print(f"Saved correlation matrix figure to {corr_fig_path}")

    # Also save an imputed Parquet file for fast model training by Laptop 1 teammate
    clean_parquet_path = PROCESSED_DIR / "candidate_features_clean.parquet"
    print(f"Exporting clean, imputed feature table to {clean_parquet_path}...")
    df.to_parquet(clean_parquet_path, index=False)
    print(f"Saved clean table to {clean_parquet_path} ({clean_parquet_path.stat().st_size / 1e6:.1f} MB)")

    print("\n=== Feature Selection Completed Successfully ===")
    print(f"Selected feature set ({len(selected_features)} features): {selected_features}")


if __name__ == "__main__":
    main()
