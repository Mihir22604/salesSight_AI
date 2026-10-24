"""
Consolidated Train / Test Split Module for SalesSight AI (Step 7).
Combines:
1. TimeSeriesSplitter: Strict chronological 80:20 production holdout partition
   and expanding-window chronological cross-validation folds (K=4, horizon=90d).
   Enforces zero-shuffle leakage prohibition and date continuity assertions.
2. SplitVisualizer: Publication-grade diagnostic charts for timeline partitions,
   target distribution parity, and zero-sales sparsity density comparisons.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns

from src.utils import get_logger, set_seed

logger = get_logger("TrainTestSplit")

# Publication Styling Configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

PALETTE = {
    "train": "#2b5c8f",        # Executive Blue (Training)
    "test": "#d95f02",         # Deep Coral / Amber (Test Holdout)
    "fold1": "#7570b3",        # Purple
    "fold2": "#1b9e77",        # Emerald
    "fold3": "#e7298a",        # Magenta
    "fold4": "#e6ab02",        # Gold
    "neutral": "#95a5a6",      # Muted Gray
}


# ==============================================================================
# 1. TIME SERIES SPLITTER
# ==============================================================================
class TimeSeriesSplitter:
    """
    Implements strictly chronological partition protocols:
    1. Global Canonical Production Holdout Split (80:20 ratio, zero shuffling)
    2. Expanding-Window Chronological Cross-Validation Folds (K=4, horizon H=90 days)
    """

    def __init__(self, date_col: str = "Date"):
        self.date_col = date_col

    def create_canonical_holdout_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.80,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits a time-series DataFrame into Training and Test holdouts strictly by calendar date.
        Guarantees max(train_date) < min(test_date).
        """
        logger.info(f"Generating canonical {train_ratio*100:.0f}:{(1-train_ratio)*100:.0f} chronological holdout split...")
        sorted_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(sorted_df[self.date_col]):
            sorted_df[self.date_col] = pd.to_datetime(sorted_df[self.date_col])

        sorted_df = sorted_df.sort_values(self.date_col).reset_index(drop=True)

        n_total = len(sorted_df)
        split_idx = int(n_total * train_ratio)

        train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

        max_train_date = train_df[self.date_col].max()
        min_test_date = test_df[self.date_col].min()

        # Strict Temporal Leakage Assertion
        if max_train_date >= min_test_date:
            raise ValueError(
                f"FATAL TIME LEAKAGE: max train date ({max_train_date}) is not strictly before min test date ({min_test_date})!"
            )

        metadata = {
            "total_observations": n_total,
            "train_observations": len(train_df),
            "test_observations": len(test_df),
            "train_ratio": round(len(train_df) / n_total, 4),
            "test_ratio": round(len(test_df) / n_total, 4),
            "train_start_date": str(train_df[self.date_col].min().date()),
            "train_end_date": str(max_train_date.date()),
            "test_start_date": str(min_test_date.date()),
            "test_end_date": str(test_df[self.date_col].max().date()),
            "boundary_split_date": str(min_test_date.date()),
            "calendar_gap_days": (min_test_date - max_train_date).days - 1,
        }

        logger.info(
            f"Holdout Split Created: Train={len(train_df)} days ({metadata['train_start_date']} to {metadata['train_end_date']}) | "
            f"Test={len(test_df)} days ({metadata['test_start_date']} to {metadata['test_end_date']})"
        )
        return train_df, test_df, metadata

    def create_expanding_folds(
        self,
        df: pd.DataFrame,
        initial_train_size: int = 780,
        horizon: int = 90,
        n_splits: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Constructs expanding-window chronological cross-validation folds.
        For fold k:
        - Train Window: [0 : initial_train_size + (k * horizon)]
        - Test Horizon: [Train_End : Train_End + horizon]
        """
        logger.info(f"Generating {n_splits} expanding chronological folds (initial={initial_train_size}d, horizon={horizon}d)...")
        sorted_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(sorted_df[self.date_col]):
            sorted_df[self.date_col] = pd.to_datetime(sorted_df[self.date_col])

        sorted_df = sorted_df.sort_values(self.date_col).reset_index(drop=True)
        n_total = len(sorted_df)

        folds = []
        for k in range(n_splits):
            train_end_idx = initial_train_size + (k * horizon)
            test_end_idx = train_end_idx + horizon

            if test_end_idx > n_total:
                logger.warning(f"Fold {k+1} exceeds total dataset length ({test_end_idx} > {n_total}); truncation applied.")
                test_end_idx = n_total

            train_partition = sorted_df.iloc[:train_end_idx]
            test_partition = sorted_df.iloc[train_end_idx:test_end_idx]

            fold_meta = {
                "fold_index": k + 1,
                "train_start_idx": 0,
                "train_end_idx": train_end_idx - 1,
                "train_days_count": len(train_partition),
                "test_start_idx": train_end_idx,
                "test_end_idx": test_end_idx - 1,
                "test_days_count": len(test_partition),
                "train_start_date": str(train_partition[self.date_col].min().date()),
                "train_end_date": str(train_partition[self.date_col].max().date()),
                "test_start_date": str(test_partition[self.date_col].min().date()),
                "test_end_date": str(test_partition[self.date_col].max().date()),
                "horizon_days": len(test_partition),
            }
            folds.append(fold_meta)

        return folds

    def audit_statistical_representativeness(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_cols: List[str] = ["Sales", "Quantity"],
    ) -> Dict[str, Any]:
        """
        Audits distribution parity between Train and Test sets:
        Computes mean, median, standard deviation, zero-sales percentage,
        and two-sample Kolmogorov-Smirnov test statistics.
        """
        logger.info("Auditing statistical parity between Train and Test partitions...")
        audit_results = {}

        for col in target_cols:
            tr_vals = train_df[col].dropna()
            te_vals = test_df[col].dropna()

            ks_stat, p_val = stats.ks_2samp(tr_vals, te_vals)

            audit_results[col] = {
                "train": {
                    "count": int(len(tr_vals)),
                    "mean": round(float(tr_vals.mean()), 2),
                    "median": round(float(tr_vals.median()), 2),
                    "std": round(float(tr_vals.std()), 2),
                    "zero_count": int((tr_vals == 0).sum()),
                    "zero_pct": round(float((tr_vals == 0).mean() * 100), 2),
                },
                "test": {
                    "count": int(len(te_vals)),
                    "mean": round(float(te_vals.mean()), 2),
                    "median": round(float(te_vals.median()), 2),
                    "std": round(float(te_vals.std()), 2),
                    "zero_count": int((te_vals == 0).sum()),
                    "zero_pct": round(float((te_vals == 0).mean() * 100), 2),
                },
                "ks_test": {
                    "statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "distribution_shift_detected": bool(p_val < 0.01),
                }
            }

        return audit_results


# ==============================================================================
# 2. SPLIT VISUALIZER
# ==============================================================================
class SplitVisualizer:
    """
    Renders publication-grade diagnostic plots for timeline partitions,
    target distribution parity, and zero-sales sparsity density comparisons.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_timeline_partition(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        folds: List[Dict[str, Any]],
        filename: str = "step7_split_timeline_visualization.png",
    ) -> Path:
        """
        Renders a 2-panel timeline visualization:
        Top: Global 80:20 Production Holdout timeline showing Sales revenue.
        Bottom: Gantt chart of the 4 Expanding-Window Chronological Backtest Folds.
        """
        logger.info(f"Rendering timeline partition visualization to {filename}...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [2.2, 1.8]})

        # Panel 1: Continuous Historical Sales with Holdout Boundary
        ax1.plot(train_df["Date"], train_df["Sales"], color=PALETTE["train"], linewidth=1.2, label=f"Train Partition (80%: {len(train_df)} days)")
        ax1.plot(test_df["Date"], test_df["Sales"], color=PALETTE["test"], linewidth=1.2, label=f"Test Holdout (20%: {len(test_df)} days)")

        split_date = test_df["Date"].min()
        ax1.axvline(split_date, color="black", linestyle="--", linewidth=1.8, label=f"Holdout Cutoff Date ({split_date.date()})")

        ax1.set_title("SalesSight AI — Canonical Chronological 80:20 Production Holdout Timeline", pad=12)
        ax1.set_ylabel("Daily Sales Revenue ($)")
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.legend(loc="upper left", frameon=True)

        # Panel 2: Gantt Chart of 4 Expanding Chronological Backtest Folds
        y_pos = np.arange(len(folds))
        for idx, fold in enumerate(folds):
            t_start = pd.to_datetime(fold["train_start_date"])
            t_end = pd.to_datetime(fold["train_end_date"])
            te_start = pd.to_datetime(fold["test_start_date"])
            te_end = pd.to_datetime(fold["test_end_date"])

            # Train segment
            ax2.barh(idx, (t_end - t_start).days, left=t_start, color=PALETTE["train"], alpha=0.75, height=0.55, edgecolor="white")
            # Test horizon segment
            ax2.barh(idx, (te_end - te_start).days, left=te_start, color=PALETTE["test"], alpha=0.9, height=0.55, edgecolor="white")

            ax2.text(t_start + pd.Timedelta(days=20), idx, f"Train: {fold['train_days_count']}d", va="center", color="white", fontweight="bold", fontsize=8.5)
            ax2.text(te_start + pd.Timedelta(days=5), idx, f"H={fold['horizon_days']}d", va="center", color="white", fontweight="bold", fontsize=8.5)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"Fold {f['fold_index']}" for f in folds], fontweight="bold")
        ax2.set_title("Expanding-Window Chronological Backtest Validation Protocol (H = 90 Days)", pad=10)
        ax2.set_xlabel("Calendar Date")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.suptitle("SalesSight AI — Step 7 Chronological Partitioning & Expanding-Window Backtesting Architecture", fontsize=15, y=0.99)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_target_distributions(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        filename: str = "step7_split_target_distribution_comparison.png",
    ) -> Path:
        """Renders 4-panel KDE distributions and boxplots comparing Train vs Test target parity."""
        logger.info(f"Rendering target distribution comparison to {filename}...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))

        # 1. Sales KDE
        ax1 = axes[0, 0]
        sns.kdeplot(train_df["Sales"], color=PALETTE["train"], linewidth=2.0, fill=True, alpha=0.35, label=f"Train (Mean: ${train_df['Sales'].mean():,.0f})", ax=ax1)
        sns.kdeplot(test_df["Sales"], color=PALETTE["test"], linewidth=2.0, fill=True, alpha=0.35, label=f"Test (Mean: ${test_df['Sales'].mean():,.0f})", ax=ax1)
        ax1.set_title("Sales ($) Density Distribution: Train vs. Test", pad=10)
        ax1.set_xlabel("Daily Sales Revenue ($)")
        ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
        ax1.legend(loc="upper right", frameon=True)

        # 2. Sales Boxplot
        ax2 = axes[0, 1]
        combo_sales = pd.DataFrame({
            "Revenue": pd.concat([train_df["Sales"], test_df["Sales"]]),
            "Partition": ["Train (80%)"] * len(train_df) + ["Test (20%)"] * len(test_df)
        })
        sns.boxplot(data=combo_sales, x="Partition", y="Revenue", hue="Partition", palette=[PALETTE["train"], PALETTE["test"]], ax=ax2, width=0.45, legend=False)
        ax2.set_title("Sales ($) Outlier & Quartile Comparison", pad=10)
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

        # 3. Quantity KDE
        ax3 = axes[1, 0]
        sns.kdeplot(train_df["Quantity"], color=PALETTE["train"], linewidth=2.0, fill=True, alpha=0.35, label=f"Train (Mean: {train_df['Quantity'].mean():.1f}u)", ax=ax3)
        sns.kdeplot(test_df["Quantity"], color=PALETTE["test"], linewidth=2.0, fill=True, alpha=0.35, label=f"Test (Mean: {test_df['Quantity'].mean():.1f}u)", ax=ax3)
        ax3.set_title("Quantity (Units) Density Distribution: Train vs. Test", pad=10)
        ax3.set_xlabel("Daily Units Ordered")
        ax3.legend(loc="upper right", frameon=True)

        # 4. Quantity Boxplot
        ax4 = axes[1, 1]
        combo_qty = pd.DataFrame({
            "Units": pd.concat([train_df["Quantity"], test_df["Quantity"]]),
            "Partition": ["Train (80%)"] * len(train_df) + ["Test (20%)"] * len(test_df)
        })
        sns.boxplot(data=combo_qty, x="Partition", y="Units", hue="Partition", palette=[PALETTE["train"], PALETTE["test"]], ax=ax4, width=0.45, legend=False)
        ax4.set_title("Quantity Outlier & Quartile Comparison", pad=10)

        plt.suptitle("SalesSight AI — Step 7 Distribution Parity & Statistical Representativeness", fontsize=15, y=0.99)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_zero_density_comparison(
        self,
        audit_metrics: Dict[str, Any],
        filename: str = "step7_split_sparsity_and_zero_density.png",
    ) -> Path:
        """Renders grouped bar chart of zero-sales sparsity and active order rates."""
        logger.info(f"Rendering zero density comparison to {filename}...")
        fig, ax = plt.subplots(figsize=(10, 6))

        s_tr = audit_metrics["Sales"]["train"]["zero_pct"]
        s_te = audit_metrics["Sales"]["test"]["zero_pct"]
        q_tr = audit_metrics["Quantity"]["train"]["zero_pct"]
        q_te = audit_metrics["Quantity"]["test"]["zero_pct"]

        x = np.arange(2)
        width = 0.35

        bars1 = ax.bar(x - width / 2, [s_tr, q_tr], width, label="Train Partition (80%)", color=PALETTE["train"], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width / 2, [s_te, q_te], width, label="Test Holdout (20%)", color=PALETTE["test"], alpha=0.85, edgecolor="white")

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=9.5)

        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=9.5, color=PALETTE["test"])

        ax.set_title("SalesSight AI — Intermittent Sparsity Parity (Zero-Sales Days %)", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(["Sales Revenue ($)", "Order Quantity (Units)"], fontweight="bold")
        ax.set_ylabel("Zero-Sales Days (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def render_all_figures(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        folds: List[Dict[str, Any]],
        audit_metrics: Dict[str, Any],
    ) -> Dict[str, Path]:
        """Renders all Step 7 diagnostic visualization charts and returns dictionary of saved paths."""
        logger.info("Rendering all Step 7 publication-grade split diagnostic figures...")
        return {
            "timeline_partition": self.plot_timeline_partition(train_df, test_df, folds),
            "target_distributions": self.plot_target_distributions(train_df, test_df),
            "zero_density_comparison": self.plot_zero_density_comparison(audit_metrics),
        }


# ==============================================================================
# 3. PIPELINE ORCHESTRATOR
# ==============================================================================
def run_split_pipeline(
    processed_dir: str = "data/processed",
    output_dir: str = "report_stage",
) -> Dict[str, Any]:
    """
    Executes the end-to-end Step 7: Train / Test Split pipeline.
    Loads selected_macro_features.parquet and category_features.parquet,
    generates 80:20 canonical holdout partitions and expanding backtest folds,
    audits distribution parity, saves train/test files, and outputs diagnostic charts.
    """
    logger.info("=" * 80)
    logger.info("SALESSIGHT AI: STEP 7 (TRAIN / TEST SPLIT) PIPELINE")
    logger.info("=" * 80)
    set_seed(42)

    proc_path = Path(processed_dir)
    macro_feat_file = proc_path / "selected_macro_features.parquet"
    cat_feat_file = proc_path / "category_features.parquet"

    if not macro_feat_file.exists():
        raise FileNotFoundError(f"Missing required dataset: {macro_feat_file}. Run feature_selection.py first.")

    logger.info(f"Loading selected macro feature matrix from {macro_feat_file}...")
    macro_df = pd.read_parquet(macro_feat_file)

    cat_df = None
    if cat_feat_file.exists():
        logger.info(f"Loading category features matrix from {cat_feat_file}...")
        cat_df = pd.read_parquet(cat_feat_file)

    # 1. Generate Canonical 80:20 Production Holdout Split
    splitter = TimeSeriesSplitter(date_col="Date")
    train_macro, test_macro, holdout_meta = splitter.create_canonical_holdout_split(macro_df, train_ratio=0.80)

    # Category partition (aligned to exact cutoff date)
    train_cat, test_cat, cat_meta = None, None, {}
    if cat_df is not None:
        cutoff_date = pd.to_datetime(holdout_meta["boundary_split_date"])
        train_cat = cat_df[cat_df["Date"] < cutoff_date].copy().reset_index(drop=True)
        test_cat = cat_df[cat_df["Date"] >= cutoff_date].copy().reset_index(drop=True)
        cat_meta = {
            "category_train_rows": len(train_cat),
            "category_test_rows": len(test_cat),
            "cutoff_date": str(cutoff_date.date()),
        }

    # 2. Generate Expanding Chronological Backtest Folds (K=4, H=90 days)
    folds_meta = splitter.create_expanding_folds(macro_df, initial_train_size=780, horizon=90, n_splits=4)

    # 3. Statistical Distribution Parity Audit
    audit_results = splitter.audit_statistical_representativeness(train_macro, test_macro, target_cols=["Sales", "Quantity"])

    # 4. Save Partitioned Production Datasets
    train_macro_pq = proc_path / "train_macro_features.parquet"
    train_macro_csv = proc_path / "train_macro_features.csv"
    test_macro_pq = proc_path / "test_macro_features.parquet"
    test_macro_csv = proc_path / "test_macro_features.csv"

    train_macro.to_parquet(train_macro_pq, index=False)
    train_macro.to_csv(train_macro_csv, index=False)
    test_macro.to_parquet(test_macro_pq, index=False)
    test_macro.to_csv(test_macro_csv, index=False)

    saved_datasets = {
        "train_macro_features_parquet": str(train_macro_pq),
        "train_macro_features_csv": str(train_macro_csv),
        "test_macro_features_parquet": str(test_macro_pq),
        "test_macro_features_csv": str(test_macro_csv),
    }

    if train_cat is not None and test_cat is not None:
        train_cat_pq = proc_path / "train_category_features.parquet"
        train_cat_csv = proc_path / "train_category_features.csv"
        test_cat_pq = proc_path / "test_category_features.parquet"
        test_cat_csv = proc_path / "test_category_features.csv"

        train_cat.to_parquet(train_cat_pq, index=False)
        train_cat.to_csv(train_cat_csv, index=False)
        test_cat.to_parquet(test_cat_pq, index=False)
        test_cat.to_csv(test_cat_csv, index=False)

        saved_datasets.update({
            "train_category_features_parquet": str(train_cat_pq),
            "train_category_features_csv": str(train_cat_csv),
            "test_category_features_parquet": str(test_cat_pq),
            "test_category_features_csv": str(test_cat_csv),
        })

    logger.info("All partitioned train and test datasets saved successfully.")

    # 5. Render Diagnostic Figures
    out_dir = Path(output_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    visualizer = SplitVisualizer(output_dir=assets_dir)
    figures = visualizer.render_all_figures(
        train_df=train_macro,
        test_df=test_macro,
        folds=folds_meta,
        audit_metrics=audit_results,
    )

    # 6. Serialize JSON Metrics
    metrics = {
        "holdout_split_metadata": holdout_meta,
        "category_split_metadata": cat_meta,
        "expanding_folds": folds_meta,
        "statistical_parity_audit": audit_results,
        "figure_artifacts": {k: str(v) for k, v in figures.items()},
        "saved_datasets": saved_datasets,
    }

    metrics_path = out_dir / "step7_train_test_split_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Serialized complete Step 7 Train/Test Split metrics to: {metrics_path}")

    logger.info("-" * 80)
    logger.info("STEP 7 TRAIN / TEST SPLIT SUMMARY HIGHLIGHTS:")
    logger.info(f"  - Canonical Partition: 80% Train ({holdout_meta['train_observations']} days) : 20% Test ({holdout_meta['test_observations']} days)")
    logger.info(f"  - Boundary Split Cutoff: {holdout_meta['boundary_split_date']} (Strictly zero calendar overlap)")
    logger.info(f"  - Expanding Folds: {len(folds_meta)} folds configured (Horizon H=90 days)")
    logger.info(f"  - Sparsity Parity: Train zero days = {audit_results['Sales']['train']['zero_pct']}% | Test zero days = {audit_results['Sales']['test']['zero_pct']}%")
    logger.info(f"  - Diagnostic Charts: {len(figures)} figures saved directly in {assets_dir.name}/")
    logger.info("=" * 80)
    logger.info("STEP 7 TRAIN / TEST SPLIT PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    return {
        "train_macro": train_macro,
        "test_macro": test_macro,
        "metrics": metrics,
        "figures": figures,
    }


if __name__ == "__main__":
    run_split_pipeline()
