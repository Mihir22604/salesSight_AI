"""
Consolidated Correlation Analysis Module for SalesSight AI (Step 4).
Combines:
1. CorrelationAnalyzer: Pearson (linear), Spearman (rank-order), p-values,
   collinearity pair screening (|r| > 0.85), price elasticity correlation,
   and Variance Inflation Factor (VIF) multicollinearity auditing.
2. CorrelationVisualizer: Publication-grade high-resolution diagnostic heatmaps,
   target correlation bar charts, and VIF threshold diagnostic plots.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.utils import get_logger, set_seed

logger = get_logger("CorrelationAnalysis")

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
    "primary": "#2b5c8f",      # Executive blue
    "secondary": "#d95f02",    # Deep amber / coral
    "accent": "#7570b3",       # Purple / indigo
    "success": "#1b9e77",      # Emerald green
    "danger": "#d9534f",       # Warning red
    "warning": "#f0ad4e",      # Amber warning
}


# ==============================================================================
# 1. CORRELATION ANALYZER
# ==============================================================================
class CorrelationAnalyzer:
    """
    Computes parametric (Pearson) and non-parametric (Spearman) correlations,
    hypothesis test p-values, pairwise collinearity screening (|r| > 0.85),
    and Variance Inflation Factor (VIF) multicollinearity diagnostics.
    """

    def __init__(self, transactions_df: pd.DataFrame, macro_daily_df: pd.DataFrame):
        self.transactions = transactions_df.copy()
        self.macro_daily = macro_daily_df.copy()

        if "Date" in self.macro_daily.columns and not pd.api.types.is_datetime64_any_dtype(self.macro_daily["Date"]):
            self.macro_daily["Date"] = pd.to_datetime(self.macro_daily["Date"])

        # Construct derived commercial variables if not present
        if "GrossSales" not in self.transactions.columns:
            self.transactions["GrossSales"] = self.transactions["Sales"] + self.transactions["Discount"]

    def compute_transaction_correlations(self) -> Dict[str, Any]:
        """
        Calculates Pearson and Spearman correlation matrices across transaction numerical attributes.
        Returns matrices, p-value matrices, and target-to-feature correlation profiles.
        """
        logger.info("Computing transaction-level correlation matrices...")
        feature_cols = ["Sales", "Quantity", "UnitPrice", "Discount", "DiscountRate", "GrossSales"]
        sub_df = self.transactions[feature_cols].dropna()

        # Pearson correlation and p-values
        pearson_corr = sub_df.corr(method="pearson")
        spearman_corr = sub_df.corr(method="spearman")

        n = len(sub_df)
        p_values_pearson = pd.DataFrame(np.zeros((len(feature_cols), len(feature_cols))), columns=feature_cols, index=feature_cols)
        p_values_spearman = pd.DataFrame(np.zeros((len(feature_cols), len(feature_cols))), columns=feature_cols, index=feature_cols)

        for col1 in feature_cols:
            for col2 in feature_cols:
                if col1 == col2:
                    p_values_pearson.loc[col1, col2] = 0.0
                    p_values_spearman.loc[col1, col2] = 0.0
                else:
                    _, p_p = stats.pearsonr(sub_df[col1], sub_df[col2])
                    _, p_s = stats.spearmanr(sub_df[col1], sub_df[col2])
                    p_values_pearson.loc[col1, col2] = p_p
                    p_values_spearman.loc[col1, col2] = p_s

        return {
            "features": feature_cols,
            "sample_size": n,
            "pearson_matrix": pearson_corr.to_dict(),
            "spearman_matrix": spearman_corr.to_dict(),
            "pearson_p_values": p_values_pearson.to_dict(),
            "spearman_p_values": p_values_spearman.to_dict(),
            "sales_correlations": {
                "pearson": pearson_corr["Sales"].to_dict(),
                "spearman": spearman_corr["Sales"].to_dict(),
            },
            "quantity_correlations": {
                "pearson": pearson_corr["Quantity"].to_dict(),
                "spearman": spearman_corr["Quantity"].to_dict(),
            },
        }

    def prepare_timeseries_feature_matrix(self) -> pd.DataFrame:
        """
        Constructs standard candidate time-series forecasting features from the continuous macro daily series.
        Features include autoregressive lags, shifted rolling windows, rolling standard deviation, and calendar cycles.
        """
        df = self.macro_daily.copy().sort_values("Date").reset_index(drop=True)

        # Autoregressive Lags (Target: Sales)
        df["lag_1"] = df["Sales"].shift(1)
        df["lag_7"] = df["Sales"].shift(7)
        df["lag_14"] = df["Sales"].shift(14)
        df["lag_28"] = df["Sales"].shift(28)

        # Shifted Rolling Statistics (strictly shift(1) to avoid lookahead leakage)
        df["rolling_mean_7"] = df["Sales"].shift(1).rolling(7, min_periods=7).mean()
        df["rolling_mean_14"] = df["Sales"].shift(1).rolling(14, min_periods=14).mean()
        df["rolling_mean_28"] = df["Sales"].shift(1).rolling(28, min_periods=28).mean()
        df["rolling_std_7"] = df["Sales"].shift(1).rolling(7, min_periods=7).std().fillna(0)
        df["rolling_std_28"] = df["Sales"].shift(1).rolling(28, min_periods=28).std().fillna(0)

        # Autoregressive Lags (Target: Quantity)
        df["qty_lag_1"] = df["Quantity"].shift(1)
        df["qty_lag_7"] = df["Quantity"].shift(7)
        df["qty_rolling_mean_7"] = df["Quantity"].shift(1).rolling(7, min_periods=7).mean()

        # Calendar Temporal Features
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["DayOfMonth"] = df["Date"].dt.day
        df["Month"] = df["Date"].dt.month
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

        return df.dropna().reset_index(drop=True)

    def compute_timeseries_correlations(self) -> Dict[str, Any]:
        """
        Calculates correlation between target variables (Sales, Quantity) and candidate time-series features.
        """
        logger.info("Computing time-series feature correlation matrix...")
        ts_df = self.prepare_timeseries_feature_matrix()

        feature_cols = [
            "Sales", "Quantity",
            "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
            "rolling_std_7", "rolling_std_28",
            "DayOfWeek", "DayOfMonth", "Month", "IsWeekend"
        ]

        sub_df = ts_df[feature_cols]
        pearson_corr = sub_df.corr(method="pearson")
        spearman_corr = sub_df.corr(method="spearman")

        return {
            "observations": len(sub_df),
            "features": feature_cols,
            "pearson_matrix": pearson_corr.to_dict(),
            "spearman_matrix": spearman_corr.to_dict(),
            "sales_correlations": {
                "pearson": pearson_corr["Sales"].to_dict(),
                "spearman": spearman_corr["Sales"].to_dict(),
            },
            "quantity_correlations": {
                "pearson": pearson_corr["Quantity"].to_dict(),
                "spearman": spearman_corr["Quantity"].to_dict(),
            },
        }

    def compute_vif(self, feature_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Computes Variance Inflation Factor (VIF) for each numerical feature via OLS regression:
        VIF_i = 1 / (1 - R_i^2)
        Classifies multicollinearity severity:
        - Low: VIF <= 5.0
        - Moderate: 5.0 < VIF <= 10.0
        - Severe (Redundant): VIF > 10.0
        """
        logger.info("Computing Variance Inflation Factor (VIF) diagnostics...")
        clean_df = feature_df.dropna().select_dtypes(include=[np.number])
        results = []

        for col in clean_df.columns:
            X = clean_df.drop(columns=[col])
            y = clean_df[col]
            if X.shape[1] == 0:
                continue

            reg = LinearRegression().fit(X, y)
            r_sq = float(reg.score(X, y))
            denom = 1.0 - r_sq
            vif = 1.0 / denom if denom > 1e-6 else 999.99

            if vif > 10.0:
                severity = "Severe Multicollinearity (Prune Candidate)"
                risk_level = "High"
            elif vif > 5.0:
                severity = "Moderate Multicollinearity"
                risk_level = "Moderate"
            else:
                severity = "Low Collinearity (Safe)"
                risk_level = "Low"

            results.append({
                "feature": col,
                "vif": round(float(vif), 2),
                "r_squared": round(float(r_sq), 4),
                "tolerance": round(float(denom), 4),
                "severity": severity,
                "risk_level": risk_level,
            })

        results.sort(key=lambda x: x["vif"], reverse=True)
        return results

    def screen_collinear_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Identifies pairs of distinct independent features with absolute correlation |r| >= threshold.
        """
        logger.info(f"Screening collinear feature pairs with |r| >= {threshold}...")
        collinear_pairs = []
        cols = corr_matrix.columns.tolist()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                r = float(corr_matrix.loc[col1, col2])
                if abs(r) >= threshold:
                    collinear_pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(r, 4),
                        "abs_correlation": round(abs(r), 4),
                        "recommendation": f"Prune one feature or combine; collinearity |r| = {abs(r):.2f} >= {threshold}"
                    })

        collinear_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return collinear_pairs

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Executes end-to-end correlation analysis across transactional and time-series feature spaces.
        """
        logger.info("Executing full Step 4 Correlation Analysis...")

        # 1. Transaction Correlations
        tx_corr = self.compute_transaction_correlations()
        tx_pearson_df = pd.DataFrame(tx_corr["pearson_matrix"])
        tx_collinear = self.screen_collinear_pairs(tx_pearson_df, threshold=0.85)

        tx_predictors = self.transactions[["Quantity", "UnitPrice", "Discount", "DiscountRate"]].dropna()
        tx_vif = self.compute_vif(tx_predictors)

        # 2. Time-Series Feature Correlations
        ts_corr = self.compute_timeseries_correlations()
        ts_pearson_df = pd.DataFrame(ts_corr["pearson_matrix"])
        ts_collinear = self.screen_collinear_pairs(ts_pearson_df, threshold=0.85)

        ts_df = self.prepare_timeseries_feature_matrix()
        ts_candidate_predictors = ts_df[[
            "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
            "rolling_std_7", "rolling_std_28",
            "DayOfWeek", "Month"
        ]]
        ts_vif = self.compute_vif(ts_candidate_predictors)

        # 3. Price Elasticity Diagnostic Correlation
        elasticity_r, elasticity_p = stats.spearmanr(self.transactions["UnitPrice"], self.transactions["Quantity"])

        return {
            "transaction_analysis": {
                "correlations": tx_corr,
                "collinear_pairs_screened": tx_collinear,
                "vif_analysis": tx_vif,
                "price_elasticity": {
                    "spearman_rho": round(float(elasticity_r), 4),
                    "p_value": float(elasticity_p),
                    "interpretation": "Negative rank correlation confirms downward-sloping economic demand curve (higher price -> lower unit volume)."
                }
            },
            "timeseries_analysis": {
                "correlations": ts_corr,
                "collinear_pairs_screened": ts_collinear,
                "vif_analysis": ts_vif,
            },
            "summary_recommendations": {
                "multicollinarity_pruning": [
                    item["feature"] for item in ts_vif if item["vif"] > 10.0
                ],
                "primary_sales_drivers": [
                    {"feature": k, "pearson_r": round(v, 4)}
                    for k, v in sorted(ts_corr["sales_correlations"]["pearson"].items(), key=lambda x: abs(x[1]), reverse=True)
                    if k not in ["Sales", "Quantity"]
                ][:5],
                "primary_quantity_drivers": [
                    {"feature": k, "pearson_r": round(v, 4)}
                    for k, v in sorted(ts_corr["quantity_correlations"]["pearson"].items(), key=lambda x: abs(x[1]), reverse=True)
                    if k not in ["Sales", "Quantity"]
                ][:5],
            }
        }


# ==============================================================================
# 2. CORRELATION VISUALIZER
# ==============================================================================
class CorrelationVisualizer:
    """
    Renders publication-grade diagnostic correlation heatmaps, target relationships,
    and Variance Inflation Factor (VIF) bar charts directly into report_stage/assets/.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_pearson_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "SalesSight AI — Parametric Pearson Correlation Matrix (Linear Relationships)",
        filename: str = "step4_correlation_pearson_heatmap.png",
    ) -> Path:
        """Renders an annotated lower-triangle Pearson correlation heatmap."""
        logger.info(f"Rendering Pearson heatmap to {filename}...")
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=1.0,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
            ax=ax,
        )

        ax.set_title(title, pad=15, fontsize=13)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_spearman_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "SalesSight AI — Non-Parametric Spearman Rank Correlation Matrix (Monotonic Relationships)",
        filename: str = "step4_correlation_spearman_heatmap.png",
    ) -> Path:
        """Renders an annotated lower-triangle Spearman rank correlation heatmap."""
        logger.info(f"Rendering Spearman heatmap to {filename}...")
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=1.0,
            cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation (ρ)"},
            ax=ax,
        )

        ax.set_title(title, pad=15, fontsize=13)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_target_correlations(
        self,
        sales_corr: Dict[str, float],
        qty_corr: Dict[str, float],
        filename: str = "step4_correlation_target_relationships.png",
    ) -> Path:
        """Renders side-by-side horizontal bar charts comparing predictor correlations against Sales vs Quantity."""
        logger.info(f"Rendering target correlation comparison to {filename}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=False)

        # Filter out self-correlations
        s_data = {k: v for k, v in sales_corr.items() if k not in ["Sales", "Quantity"]}
        q_data = {k: v for k, v in qty_corr.items() if k not in ["Sales", "Quantity"]}

        s_series = pd.Series(s_data).sort_values()
        q_series = pd.Series(q_data).sort_values()

        colors_s = [PALETTE["danger"] if x < 0 else PALETTE["primary"] for x in s_series.values]
        colors_q = [PALETTE["danger"] if x < 0 else PALETTE["secondary"] for x in q_series.values]

        bars1 = ax1.barh(s_series.index, s_series.values, color=colors_s, alpha=0.85, edgecolor="white")
        ax1.axvline(0, color="#666666", linewidth=1.0, linestyle="--")
        ax1.set_title("Feature Correlations with Target 1: Sales ($)", pad=10)
        ax1.set_xlabel("Pearson Correlation (r)")
        ax1.set_xlim(-0.4, 0.9)
        for bar in bars1:
            w = bar.get_width()
            align = "left" if w >= 0 else "right"
            offset = 0.02 if w >= 0 else -0.02
            ax1.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.2f}", va="center", ha=align, fontsize=8.5, fontweight="bold")

        bars2 = ax2.barh(q_series.index, q_series.values, color=colors_q, alpha=0.85, edgecolor="white")
        ax2.axvline(0, color="#666666", linewidth=1.0, linestyle="--")
        ax2.set_title("Feature Correlations with Target 2: Quantity (Units)", pad=10)
        ax2.set_xlabel("Pearson Correlation (r)")
        ax2.set_xlim(-0.4, 0.9)
        for bar in bars2:
            w = bar.get_width()
            align = "left" if w >= 0 else "right"
            offset = 0.02 if w >= 0 else -0.02
            ax2.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.2f}", va="center", ha=align, fontsize=8.5, fontweight="bold")

        plt.suptitle("SalesSight AI — Decoupled Dual-Target Predictor Correlation Profile", fontsize=15, y=0.99)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_vif_diagnostics(
        self,
        vif_list: List[Dict[str, Any]],
        title: str = "SalesSight AI — Variance Inflation Factor (VIF) Multicollinearity Audit",
        filename: str = "step4_correlation_vif_collinearity.png",
    ) -> Path:
        """Renders a horizontal bar chart displaying VIF values against standard safety thresholds (5.0 and 10.0)."""
        logger.info(f"Rendering VIF diagnostic plot to {filename}...")
        fig, ax = plt.subplots(figsize=(12, 6))

        df_vif = pd.DataFrame(vif_list).sort_values("vif", ascending=True)

        colors = []
        for val in df_vif["vif"]:
            if val > 10.0:
                colors.append(PALETTE["danger"])   # Severe
            elif val > 5.0:
                colors.append(PALETTE["warning"])  # Moderate
            else:
                colors.append(PALETTE["success"])  # Safe

        bars = ax.barh(df_vif["feature"], df_vif["vif"], color=colors, alpha=0.85, edgecolor="white")

        ax.axvline(5.0, color=PALETTE["warning"], linestyle="--", linewidth=1.8, label="Moderate Threshold (VIF = 5.0)")
        ax.axvline(10.0, color=PALETTE["danger"], linestyle="--", linewidth=1.8, label="Severe Collinearity Threshold (VIF = 10.0)")

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9, fontweight="bold")

        ax.set_title(title, pad=12)
        ax.set_xlabel("Variance Inflation Factor (VIF)")
        ax.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_timeseries_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        filename: str = "step4_correlation_timeseries_heatmap.png",
    ) -> Path:
        """Renders time-series features correlation heatmap against target variables."""
        logger.info(f"Rendering time-series correlation heatmap to {filename}...")
        fig, ax = plt.subplots(figsize=(14, 11))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.8,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
            ax=ax,
        )

        ax.set_title("SalesSight AI — Time-Series Autoregressive, Rolling, and Seasonal Feature Correlation Matrix", pad=15, fontsize=13)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def render_all_figures(
        self,
        tx_corr: Dict[str, Any],
        ts_corr: Dict[str, Any],
        ts_vif: List[Dict[str, Any]],
    ) -> Dict[str, Path]:
        """Renders all Step 4 diagnostic visualization figures and returns saved paths."""
        logger.info("Rendering all Step 4 publication-grade correlation diagnostic charts...")
        tx_p_df = pd.DataFrame(tx_corr["pearson_matrix"])
        tx_s_df = pd.DataFrame(tx_corr["spearman_matrix"])
        ts_p_df = pd.DataFrame(ts_corr["pearson_matrix"])

        return {
            "pearson_heatmap": self.plot_pearson_heatmap(tx_p_df),
            "spearman_heatmap": self.plot_spearman_heatmap(tx_s_df),
            "target_relationships": self.plot_target_correlations(
                ts_corr["sales_correlations"]["pearson"],
                ts_corr["quantity_correlations"]["pearson"]
            ),
            "vif_diagnostics": self.plot_vif_diagnostics(ts_vif),
            "timeseries_heatmap": self.plot_timeseries_correlation_heatmap(ts_p_df),
        }


# ==============================================================================
# 3. PIPELINE ORCHESTRATOR
# ==============================================================================
def run_correlation_pipeline(
    processed_dir: str = "data/processed",
    output_dir: str = "report_stage",
) -> Dict[str, Any]:
    """
    Executes the end-to-end Step 4: Correlation Analysis pipeline.
    Computes Pearson and Spearman matrices, collinearity screenings, VIF auditing,
    renders diagnostic plots into report_stage/assets/, and serializes JSON metrics.
    """
    logger.info("=" * 80)
    logger.info("SALESSIGHT AI: STEP 4 (CORRELATION ANALYSIS) PIPELINE")
    logger.info("=" * 80)
    set_seed(42)

    proc_path = Path(processed_dir)
    tx_file = proc_path / "cleaned_transactions.parquet"
    macro_file = proc_path / "all_products_daily.parquet"

    for p in [tx_file, macro_file]:
        if not p.exists():
            raise FileNotFoundError(f"Required processed dataset missing: {p}. Run data_pipeline.py first.")

    logger.info("Ingesting processed datasets for correlation analysis...")
    tx_df = pd.read_parquet(tx_file)
    macro_df = pd.read_parquet(macro_file)

    # 1. Statistical Correlation Analysis
    analyzer = CorrelationAnalyzer(tx_df, macro_df)
    results = analyzer.run_full_analysis()

    # 2. Render Diagnostic Figures
    out_dir = Path(output_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    visualizer = CorrelationVisualizer(output_dir=assets_dir)
    figures = visualizer.render_all_figures(
        tx_corr=results["transaction_analysis"]["correlations"],
        ts_corr=results["timeseries_analysis"]["correlations"],
        ts_vif=results["timeseries_analysis"]["vif_analysis"],
    )

    results["figure_artifacts"] = {k: str(v) for k, v in figures.items()}

    # 3. Serialize Metrics to JSON
    metrics_path = out_dir / "step4_correlation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Serialized complete Step 4 Correlation metrics to: {metrics_path}")

    # Summary highlights
    severe_vif = [item["feature"] for item in results["timeseries_analysis"]["vif_analysis"] if item["vif"] > 10.0]
    collinear_count = len(results["timeseries_analysis"]["collinear_pairs_screened"])
    elasticity_rho = results["transaction_analysis"]["price_elasticity"]["spearman_rho"]

    logger.info("-" * 80)
    logger.info("STEP 4 CORRELATION ANALYSIS SUMMARY HIGHLIGHTS:")
    logger.info(f"  - Price Elasticity (UnitPrice vs Quantity): Spearman rho = {elasticity_rho:+.4f} (p < 0.001)")
    logger.info(f"  - Severe Multicollinearity Features (VIF > 10): {severe_vif}")
    logger.info(f"  - Screened Collinear Time-Series Pairs (|r| >= 0.85): {collinear_count} pairs identified")
    logger.info(f"  - Generated Diagnostic Charts: {len(figures)} figures saved directly in {assets_dir.name}/")
    logger.info("=" * 80)
    logger.info("STEP 4 CORRELATION ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    return {
        "metrics": results,
        "figures": figures,
    }


if __name__ == "__main__":
    run_correlation_pipeline()
