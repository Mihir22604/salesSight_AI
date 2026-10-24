"""
Consolidated Exploratory Data Analysis (EDA) Module for SalesSight AI (Step 3).
Combines:
1. EDAAnalyzer: Statistical profiling, moment auditing, category concentration, Pareto curves, seasonality, and price elasticity
2. EDAVisualizer: Publication-grade high-resolution diagnostic plotting engine
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.utils import get_logger

logger = get_logger("EDA")

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
    "highlight": "#e7298a",    # Vibrant magenta
}


# ==============================================================================
# 1. EDA ANALYZER
# ==============================================================================
class EDAAnalyzer:
    """Computes comprehensive statistical, seasonal, Pareto, and elasticity metrics."""

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        macro_daily_df: pd.DataFrame,
        category_daily_df: pd.DataFrame,
        sku_daily_df: pd.DataFrame,
    ):
        self.transactions = transactions_df.copy()
        self.macro_daily = macro_daily_df.copy()
        self.category_daily = category_daily_df.copy()
        self.sku_daily = sku_daily_df.copy()

        # Ensure datetime typing
        if not pd.api.types.is_datetime64_any_dtype(self.transactions["Order_Date"]):
            self.transactions["Order_Date"] = pd.to_datetime(self.transactions["Order_Date"], utc=True)
        if not pd.api.types.is_datetime64_any_dtype(self.macro_daily["Date"]):
            self.macro_daily["Date"] = pd.to_datetime(self.macro_daily["Date"])
        if not pd.api.types.is_datetime64_any_dtype(self.category_daily["Date"]):
            self.category_daily["Date"] = pd.to_datetime(self.category_daily["Date"])

    def analyze_target_distributions(self) -> Dict[str, Any]:
        """Calculates statistical moments, skewness, kurtosis, and zero ratios."""
        logger.info("Analyzing target distributions for Sales and Quantity...")

        sales_tx = self.transactions["Sales"]
        qty_tx = self.transactions["Quantity"]
        sales_macro = self.macro_daily["Sales"]
        qty_macro = self.macro_daily["Quantity"]

        def _compute_profile(series: pd.Series, name: str) -> Dict[str, Any]:
            vals = series.values
            nonzero_vals = vals[vals > 0]
            return {
                "name": name,
                "count": int(len(vals)),
                "sum": float(np.sum(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
                "p90": float(np.percentile(vals, 90)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "skewness": float(stats.skew(vals)),
                "kurtosis": float(stats.kurtosis(vals)),
                "zero_count": int(np.sum(vals == 0)),
                "zero_pct": float(np.mean(vals == 0) * 100),
                "nonzero_mean": float(np.mean(nonzero_vals)) if len(nonzero_vals) > 0 else 0.0,
                "nonzero_median": float(np.median(nonzero_vals)) if len(nonzero_vals) > 0 else 0.0,
            }

        return {
            "transaction_level": {
                "sales": _compute_profile(sales_tx, "Transaction Sales ($)"),
                "quantity": _compute_profile(qty_tx, "Transaction Quantity (units)"),
            },
            "macro_daily_level": {
                "sales": _compute_profile(sales_macro, "Daily Total Sales ($)"),
                "quantity": _compute_profile(qty_macro, "Daily Total Quantity (units)"),
            },
        }

    def analyze_category_performance(self) -> Dict[str, Any]:
        """Analyzes revenue and volume concentration across merchandise categories."""
        logger.info("Computing category performance and share of wallet...")
        cat_group = self.transactions.groupby("CategoryName").agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            transaction_count=("OrderID", "count"),
            unique_orders=("OrderID", "nunique"),
            avg_unit_price=("UnitPrice", "mean"),
            median_unit_price=("UnitPrice", "median"),
            avg_line_sales=("Sales", "mean"),
            avg_line_quantity=("Quantity", "mean"),
        ).reset_index()

        total_system_sales = self.transactions["Sales"].sum()
        total_system_qty = self.transactions["Quantity"].sum()

        cat_group["sales_share_pct"] = (cat_group["total_sales"] / total_system_sales) * 100
        cat_group["quantity_share_pct"] = (cat_group["total_quantity"] / total_system_qty) * 100
        cat_group = cat_group.sort_values(by="total_sales", ascending=False)

        records = []
        for _, row in cat_group.iterrows():
            records.append({
                "category": str(row["CategoryName"]),
                "total_sales": float(row["total_sales"]),
                "total_quantity": int(row["total_quantity"]),
                "transaction_count": int(row["transaction_count"]),
                "unique_orders": int(row["unique_orders"]),
                "sales_share_pct": round(float(row["sales_share_pct"]), 2),
                "quantity_share_pct": round(float(row["quantity_share_pct"]), 2),
                "avg_unit_price": round(float(row["avg_unit_price"]), 2),
                "median_unit_price": round(float(row["median_unit_price"]), 2),
                "avg_line_sales": round(float(row["avg_line_sales"]), 2),
                "avg_line_quantity": round(float(row["avg_line_quantity"]), 2),
            })

        return {"total_categories": len(records), "category_rankings": records}

    def analyze_pareto_concentration(self) -> Dict[str, Any]:
        """Computes Pareto concentration (80/20 rule) across all 77 catalog products."""
        logger.info("Evaluating SKU Pareto concentration...")
        sku_summary = self.transactions.groupby(["ProductID", "ProductName", "CategoryName"]).agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            avg_unit_price=("UnitPrice", "mean"),
            order_count=("OrderID", "count"),
        ).reset_index()

        total_sales = sku_summary["total_sales"].sum()
        total_quantity = sku_summary["total_quantity"].sum()
        total_skus = len(sku_summary)

        sku_sales_sorted = sku_summary.sort_values(by="total_sales", ascending=False).reset_index(drop=True)
        sku_sales_sorted["cum_sales"] = sku_sales_sorted["total_sales"].cumsum()
        sku_sales_sorted["cum_sales_pct"] = (sku_sales_sorted["cum_sales"] / total_sales) * 100

        sku_qty_sorted = sku_summary.sort_values(by="total_quantity", ascending=False).reset_index(drop=True)
        sku_qty_sorted["cum_qty"] = sku_qty_sorted["total_quantity"].cumsum()
        sku_qty_sorted["cum_qty_pct"] = (sku_qty_sorted["cum_qty"] / total_quantity) * 100

        def _get_milestone(df_sorted: pd.DataFrame, cum_col: str, target_pct: float) -> Dict[str, Any]:
            hit = df_sorted[df_sorted[cum_col] >= target_pct].iloc[0]
            sku_count = int(df_sorted[df_sorted[cum_col] <= target_pct].shape[0]) + 1
            return {
                "target_pct": target_pct,
                "skus_required": sku_count,
                "sku_percentage": round((sku_count / total_skus) * 100, 2),
                "marginal_sku": str(hit["ProductName"]),
            }

        return {
            "total_skus": total_skus,
            "sales_pareto": {
                "50_percent_revenue": _get_milestone(sku_sales_sorted, "cum_sales_pct", 50.0),
                "80_percent_revenue": _get_milestone(sku_sales_sorted, "cum_sales_pct", 80.0),
            },
            "quantity_pareto": {
                "50_percent_volume": _get_milestone(sku_qty_sorted, "cum_qty_pct", 50.0),
                "80_percent_volume": _get_milestone(sku_qty_sorted, "cum_qty_pct", 80.0),
            },
            "top_10_by_revenue": sku_sales_sorted.head(10)[["ProductID", "ProductName", "CategoryName", "total_sales", "total_quantity", "avg_unit_price"]].to_dict(orient="records"),
            "top_10_by_quantity": sku_qty_sorted.head(10)[["ProductID", "ProductName", "CategoryName", "total_sales", "total_quantity", "avg_unit_price"]].to_dict(orient="records"),
        }

    def analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """Analyzes day-of-week cyclicality, monthly seasonality, and annual trends."""
        logger.info("Computing temporal cyclicality and seasonality...")
        df = self.macro_daily.copy()
        df["DayOfWeek"] = df["Date"].dt.day_name()
        df["DayOfWeekNum"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month_name()
        df["MonthNum"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year

        dow_agg = df.groupby(["DayOfWeekNum", "DayOfWeek"]).agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            avg_daily_sales=("Sales", "mean"),
            avg_daily_quantity=("Quantity", "mean"),
            active_days=("Sales", lambda x: int(np.sum(x > 0))),
            total_days=("Sales", "count"),
        ).reset_index().sort_values("DayOfWeekNum")

        dow_records = []
        for _, row in dow_agg.iterrows():
            dow_records.append({
                "day_name": str(row["DayOfWeek"]),
                "day_num": int(row["DayOfWeekNum"]),
                "total_sales": float(row["total_sales"]),
                "total_quantity": int(row["total_quantity"]),
                "avg_daily_sales": round(float(row["avg_daily_sales"]), 2),
                "avg_daily_quantity": round(float(row["avg_daily_quantity"]), 2),
                "active_days": int(row["active_days"]),
                "total_days": int(row["total_days"]),
                "activity_rate_pct": round(float(row["active_days"] / row["total_days"]) * 100, 2),
            })

        month_agg = df.groupby(["MonthNum", "Month"]).agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            avg_daily_sales=("Sales", "mean"),
            avg_daily_quantity=("Quantity", "mean"),
        ).reset_index().sort_values("MonthNum")

        month_records = []
        for _, row in month_agg.iterrows():
            month_records.append({
                "month_name": str(row["Month"]),
                "month_num": int(row["MonthNum"]),
                "total_sales": float(row["total_sales"]),
                "total_quantity": int(row["total_quantity"]),
                "avg_daily_sales": round(float(row["avg_daily_sales"]), 2),
                "avg_daily_quantity": round(float(row["avg_daily_quantity"]), 2),
            })

        year_agg = df.groupby("Year").agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            active_days=("Sales", lambda x: int(np.sum(x > 0))),
            calendar_days=("Sales", "count"),
        ).reset_index().sort_values("Year")

        year_records = []
        for _, row in year_agg.iterrows():
            year_records.append({
                "year": int(row["Year"]),
                "total_sales": float(row["total_sales"]),
                "total_quantity": int(row["total_quantity"]),
                "active_days": int(row["active_days"]),
                "calendar_days": int(row["calendar_days"]),
                "avg_sales_per_calendar_day": round(float(row["total_sales"] / row["calendar_days"]), 2),
            })

        is_zero = (df["Sales"] == 0).astype(int)
        zero_runs = []
        current_run = 0
        for val in is_zero:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    zero_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            zero_runs.append(current_run)

        return {
            "day_of_week_seasonality": dow_records,
            "monthly_seasonality": month_records,
            "annual_progression": year_records,
            "zero_day_runs": {
                "total_zero_runs": len(zero_runs),
                "mean_zero_run_days": round(float(np.mean(zero_runs)), 2) if zero_runs else 0.0,
                "median_zero_run_days": float(np.median(zero_runs)) if zero_runs else 0.0,
                "max_consecutive_zero_days": int(np.max(zero_runs)) if zero_runs else 0,
            },
        }

    def analyze_price_and_discount_elasticity(self) -> Dict[str, Any]:
        """Analyzes price variance, discount bands, and quantity elasticity."""
        logger.info("Computing price and discount elasticity profiles...")
        tx = self.transactions.copy()
        disc_col = tx["DiscountRate"] if "DiscountRate" in tx.columns else (tx["Discount"] / (tx["Sales"] + tx["Discount"])).fillna(0.0)
        tx["DiscountBand"] = pd.cut(
            disc_col,
            bins=[-0.01, 0.001, 0.051, 0.101, 0.151, 0.201, 0.30],
            labels=["0% (Full Price)", "1-5%", "6-10%", "11-15%", "16-20%", "21-25%"]
        )

        disc_agg = tx.groupby("DiscountBand", observed=False).agg(
            line_count=("OrderID", "count"),
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            avg_quantity=("Quantity", "mean"),
            median_quantity=("Quantity", "median"),
            avg_unit_price=("UnitPrice", "mean"),
        ).reset_index()

        disc_records = []
        for _, row in disc_agg.iterrows():
            disc_records.append({
                "discount_band": str(row["DiscountBand"]),
                "line_count": int(row["line_count"]),
                "share_pct": round(float(row["line_count"] / len(tx)) * 100, 2),
                "total_sales": float(row["total_sales"]),
                "total_quantity": int(row["total_quantity"]),
                "avg_quantity": round(float(row["avg_quantity"]), 2),
                "median_quantity": round(float(row["median_quantity"]), 2),
                "avg_unit_price": round(float(row["avg_unit_price"]), 2),
            })

        tx["PriceTier"] = pd.qcut(
            tx["UnitPrice"],
            q=[0, 0.333, 0.666, 1.0],
            labels=["Economy (Lower 33%)", "Standard (Mid 33%)", "Premium (Upper 33%)"]
        )
        tier_agg = tx.groupby("PriceTier", observed=False).agg(
            line_count=("OrderID", "count"),
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            min_price=("UnitPrice", "min"),
            max_price=("UnitPrice", "max"),
        ).reset_index()

        tier_records = []
        for _, row in tier_agg.iterrows():
            tier_records.append({
                "price_tier": str(row["PriceTier"]),
                "line_count": int(row["line_count"]),
                "total_sales": float(row["total_sales"]),
                "total_quantity": int(row["total_quantity"]),
                "min_price": round(float(row["min_price"]), 2),
                "max_price": round(float(row["max_price"]), 2),
            })

        p_corr, p_val = stats.pearsonr(tx["UnitPrice"], tx["Quantity"])
        s_corr, s_val = stats.spearmanr(tx["UnitPrice"], tx["Quantity"])

        return {
            "discount_bands": disc_records,
            "price_tiers": tier_records,
            "price_quantity_elasticity": {
                "pearson_r": round(float(p_corr), 4),
                "pearson_p_value": float(p_val),
                "spearman_rho": round(float(s_corr), 4),
                "spearman_p_value": float(s_val),
                "economic_interpretation": "Slight negative rank correlation indicating higher unit price is weakly associated with lower volume per order line, consistent with downward-sloping demand curves."
            },
        }

    def analyze_geographic_distribution(self) -> Dict[str, Any]:
        """Analyzes sovereign country and regional revenue and volume."""
        logger.info("Computing geographic distribution of sales...")
        geo_agg = self.transactions.groupby("Country").agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
            order_count=("OrderID", "count"),
            unique_orders=("OrderID", "nunique"),
            unique_customers=("CustomerID", "nunique"),
        ).reset_index().sort_values(by="total_sales", ascending=False)

        total_sales = self.transactions["Sales"].sum()
        geo_records = []
        for _, row in geo_agg.iterrows():
            geo_records.append({
                "country": str(row["Country"]),
                "total_sales": float(row["total_sales"]),
                "total_quantity": int(row["total_quantity"]),
                "order_count": int(row["order_count"]),
                "unique_orders": int(row["unique_orders"]),
                "unique_customers": int(row["unique_customers"]),
                "sales_share_pct": round(float(row["total_sales"] / total_sales) * 100, 2),
            })

        return {"total_countries": len(geo_records), "country_rankings": geo_records}

    def run_full_analysis(self) -> Dict[str, Any]:
        """Executes all analytical dimensions and returns a comprehensive report dictionary."""
        logger.info("Running complete statistical Exploratory Data Analysis...")
        return {
            "target_distributions": self.analyze_target_distributions(),
            "category_performance": self.analyze_category_performance(),
            "pareto_concentration": self.analyze_pareto_concentration(),
            "temporal_dynamics": self.analyze_temporal_dynamics(),
            "price_and_discount_elasticity": self.analyze_price_and_discount_elasticity(),
            "geographic_distribution": self.analyze_geographic_distribution(),
        }


# ==============================================================================
# 2. EDA VISUALIZER
# ==============================================================================
class EDAVisualizer:
    """Renders and exports diagnostic visualization assets for Exploratory Data Analysis."""

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        macro_daily_df: pd.DataFrame,
        category_daily_df: pd.DataFrame,
        sku_daily_df: pd.DataFrame,
        output_dir: Path,
    ):
        self.transactions = transactions_df.copy()
        self.macro_daily = macro_daily_df.copy()
        self.category_daily = category_daily_df.copy()
        self.sku_daily = sku_daily_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not pd.api.types.is_datetime64_any_dtype(self.macro_daily["Date"]):
            self.macro_daily["Date"] = pd.to_datetime(self.macro_daily["Date"])

    def plot_target_distributions(self, filename: str = "step3_eda_target_distributions.png") -> Path:
        """Renders 4-panel distribution charts: raw vs log for Sales and Quantity."""
        logger.info("Rendering target distribution figure...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        sales = self.transactions["Sales"]
        sns.histplot(sales, bins=50, kde=True, color=PALETTE["primary"], ax=ax1, edgecolor="white", alpha=0.7)
        ax1.axvline(sales.median(), color=PALETTE["secondary"], linestyle="--", linewidth=1.8, label=f"Median: ${sales.median():,.2f}")
        ax1.axvline(sales.mean(), color="red", linestyle=":", linewidth=1.8, label=f"Mean: ${sales.mean():,.2f}")
        ax1.set_title("Transaction Sales ($) — Heavy Right Skew", pad=10)
        ax1.set_xlabel("Sales Revenue ($)")
        ax1.set_ylabel("Transaction Count")
        ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
        ax1.legend(loc="upper right", frameon=True)

        ax2 = axes[0, 1]
        log_sales = np.log1p(sales)
        sns.histplot(log_sales, bins=40, kde=True, color=PALETTE["accent"], ax=ax2, edgecolor="white", alpha=0.7)
        ax2.set_title("Transaction Log-Sales [log(1 + Sales)] — Quasi-Normal", pad=10)
        ax2.set_xlabel("log(1 + Sales)")
        ax2.set_ylabel("Transaction Count")

        ax3 = axes[1, 0]
        macro_sales = self.macro_daily["Sales"]
        sns.histplot(macro_sales, bins=50, kde=True, color=PALETTE["success"], ax=ax3, edgecolor="white", alpha=0.7)
        zero_pct = (macro_sales == 0).mean() * 100
        ax3.set_title(f"Daily Macro Sales ($) — {zero_pct:.1f}% Structural Zeros", pad=10)
        ax3.set_xlabel("Daily Sales Revenue ($)")
        ax3.set_ylabel("Calendar Days")
        ax3.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

        ax4 = axes[1, 1]
        qty = self.transactions["Quantity"]
        sns.histplot(qty, bins=35, kde=True, color=PALETTE["secondary"], ax=ax4, edgecolor="white", alpha=0.7)
        ax4.axvline(qty.median(), color=PALETTE["primary"], linestyle="--", linewidth=1.8, label=f"Median: {qty.median():.0f} units")
        ax4.axvline(qty.mean(), color="red", linestyle=":", linewidth=1.8, label=f"Mean: {qty.mean():.1f} units")
        ax4.set_title("Transaction Quantity (Units) — Discrete Volume", pad=10)
        ax4.set_xlabel("Units Ordered")
        ax4.set_ylabel("Transaction Count")
        ax4.legend(loc="upper right", frameon=True)

        plt.suptitle("SalesSight AI — Target Variable Distribution & Moment Diagnostics", fontsize=15, y=0.99)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_temporal_trajectory(self, filename: str = "step3_eda_temporal_trajectory.png") -> Path:
        """Renders 1,348-day daily trajectory with 7-day and 30-day moving averages."""
        logger.info("Rendering temporal trajectory figure...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

        df = self.macro_daily.copy().sort_values("Date")
        df["Sales_MA7"] = df["Sales"].rolling(7, min_periods=1).mean()
        df["Sales_MA30"] = df["Sales"].rolling(30, min_periods=1).mean()
        df["Qty_MA7"] = df["Quantity"].rolling(7, min_periods=1).mean()
        df["Qty_MA30"] = df["Quantity"].rolling(30, min_periods=1).mean()

        ax1.plot(df["Date"], df["Sales"], color="#bdc3c7", alpha=0.55, linewidth=0.9, label="Daily Sales (Raw)")
        ax1.plot(df["Date"], df["Sales_MA7"], color=PALETTE["primary"], linewidth=1.5, label="7-Day Moving Avg")
        ax1.plot(df["Date"], df["Sales_MA30"], color=PALETTE["secondary"], linewidth=2.2, label="30-Day Trend (Macro)")
        ax1.set_title("Commercial Sales Revenue ($) — Multi-Year Historical Trajectory (2011–2014)")
        ax1.set_ylabel("Daily Revenue ($)")
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
        ax1.legend(loc="upper left", frameon=True)

        ax2.plot(df["Date"], df["Quantity"], color="#bdc3c7", alpha=0.55, linewidth=0.9, label="Daily Units (Raw)")
        ax2.plot(df["Date"], df["Qty_MA7"], color=PALETTE["accent"], linewidth=1.5, label="7-Day Moving Avg")
        ax2.plot(df["Date"], df["Qty_MA30"], color=PALETTE["success"], linewidth=2.2, label="30-Day Trend (Volume)")
        ax2.set_title("Product Unit Demand (Quantity) — Multi-Year Historical Trajectory")
        ax2.set_xlabel("Calendar Date")
        ax2.set_ylabel("Daily Units Sold")
        ax2.legend(loc="upper left", frameon=True)

        plt.suptitle("SalesSight AI — Long-Term Temporal Dynamics & Rolling Demand Momentum", fontsize=15, y=0.99)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_seasonality_patterns(self, filename: str = "step3_eda_seasonality_patterns.png") -> Path:
        """Renders Day-of-Week and Monthly cyclical seasonality profiles."""
        logger.info("Rendering seasonality patterns figure...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        df = self.macro_daily.copy()
        df["DayOfWeek"] = df["Date"].dt.day_name()
        df["DayOfWeekNum"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.strftime("%b")
        df["MonthNum"] = df["Date"].dt.month

        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_summary = df.groupby(["DayOfWeekNum", "DayOfWeek"]).agg(
            mean_sales=("Sales", "mean"),
            mean_qty=("Quantity", "mean"),
            active_rate=("Sales", lambda x: (x > 0).mean() * 100),
        ).reset_index().sort_values("DayOfWeekNum")

        ax1 = axes[0]
        bars1 = ax1.bar(dow_summary["DayOfWeek"], dow_summary["mean_sales"], color=PALETTE["primary"], edgecolor="white", alpha=0.85)
        ax1.set_title("Day-of-Week Intra-Week Seasonality (Average Daily Sales)", pad=10)
        ax1.set_ylabel("Mean Daily Sales ($)")
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
        ax1.set_xticks(range(len(dow_order)))
        ax1.set_xticklabels(dow_order, rotation=25)

        for bar, rate in zip(bars1, dow_summary["active_rate"]):
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, yval + 25, f"{rate:.0f}% act.", ha="center", va="bottom", fontsize=8.5, color="#444444")

        month_summary = df.groupby(["MonthNum", "Month"]).agg(
            mean_sales=("Sales", "mean"),
            total_sales=("Sales", "sum"),
        ).reset_index().sort_values("MonthNum")

        ax2 = axes[1]
        ax2.bar(month_summary["Month"], month_summary["mean_sales"], color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
        ax2.set_title("Monthly Seasonality (Average Daily Sales Across Year)", pad=10)
        ax2.set_ylabel("Mean Daily Sales ($)")
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

        plt.suptitle("SalesSight AI — Intra-Week (m=7) and Annual Cyclicality Profiles", fontsize=15, y=0.99)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_category_performance(self, filename: str = "step3_eda_category_performance.png") -> Path:
        """Renders merchandise category revenue and volume breakdown."""
        logger.info("Rendering category performance figure...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        cat_summary = self.transactions.groupby("CategoryName").agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
        ).reset_index()

        total_sales = cat_summary["total_sales"].sum()
        total_qty = cat_summary["total_quantity"].sum()
        cat_summary["sales_share"] = (cat_summary["total_sales"] / total_sales) * 100
        cat_summary["qty_share"] = (cat_summary["total_quantity"] / total_qty) * 100

        cat_sales_sorted = cat_summary.sort_values(by="total_sales", ascending=True)
        bars1 = ax1.barh(cat_sales_sorted["CategoryName"], cat_sales_sorted["total_sales"], color=PALETTE["primary"], alpha=0.85, edgecolor="white")
        ax1.set_title("Total Revenue by Merchandise Category ($)", pad=10)
        ax1.set_xlabel("Gross Sales ($)")
        ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
        for bar, share in zip(bars1, cat_sales_sorted["sales_share"]):
            w = bar.get_width()
            ax1.text(w + 5000, bar.get_y() + bar.get_height() / 2, f"{share:.1f}%", va="center", fontsize=9, fontweight="bold", color=PALETTE["primary"])

        cat_qty_sorted = cat_summary.sort_values(by="total_quantity", ascending=True)
        bars2 = ax2.barh(cat_qty_sorted["CategoryName"], cat_qty_sorted["total_quantity"], color=PALETTE["success"], alpha=0.85, edgecolor="white")
        ax2.set_title("Total Unit Demand by Merchandise Category (Units)", pad=10)
        ax2.set_xlabel("Units Sold")
        ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        for bar, share in zip(bars2, cat_qty_sorted["qty_share"]):
            w = bar.get_width()
            ax2.text(w + 200, bar.get_y() + bar.get_height() / 2, f"{share:.1f}%", va="center", fontsize=9, fontweight="bold", color=PALETTE["success"])

        plt.suptitle("SalesSight AI — Category Share of Wallet and Physical Volume", fontsize=15, y=0.99)
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_pareto_concentration(self, filename: str = "step3_eda_pareto_concentration.png") -> Path:
        """Renders cumulative Pareto Lorenz curves for revenue and unit volume across all 77 SKUs."""
        logger.info("Rendering Pareto concentration curve figure...")
        fig, ax = plt.subplots(figsize=(10, 7))

        sku_agg = self.transactions.groupby(["ProductID", "ProductName"]).agg(
            total_sales=("Sales", "sum"),
            total_quantity=("Quantity", "sum"),
        ).reset_index()

        total_skus = len(sku_agg)
        sku_sales = sku_agg.sort_values(by="total_sales", ascending=False).reset_index(drop=True)
        sku_qty = sku_agg.sort_values(by="total_quantity", ascending=False).reset_index(drop=True)

        sku_sales["cum_sales_pct"] = (sku_sales["total_sales"].cumsum() / sku_sales["total_sales"].sum()) * 100
        sku_qty["cum_qty_pct"] = (sku_qty["total_quantity"].cumsum() / sku_qty["total_quantity"].sum()) * 100
        sku_pct = ((np.arange(total_skus) + 1) / total_skus) * 100

        ax.plot(sku_pct, sku_sales["cum_sales_pct"], color=PALETTE["primary"], linewidth=2.5, label="Cumulative Sales Revenue (%)")
        ax.plot(sku_pct, sku_qty["cum_qty_pct"], color=PALETTE["secondary"], linewidth=2.2, linestyle="--", label="Cumulative Unit Demand (%)")
        ax.plot([0, 100], [0, 100], color="#888888", linestyle=":", linewidth=1.2, label="Line of Perfect Equality (1:1)")
        ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% Concentration Threshold")

        s80_idx = int((sku_sales["cum_sales_pct"] >= 80).idxmax()) + 1
        s80_pct = (s80_idx / total_skus) * 100
        ax.axvline(s80_pct, color=PALETTE["primary"], linestyle=":", alpha=0.7)
        ax.plot([s80_pct], [80], marker="o", markersize=7, color=PALETTE["primary"])
        ax.annotate(
            f"Top {s80_idx} SKUs ({s80_pct:.1f}%)\ngenerate 80% of Sales",
            xy=(s80_pct, 80),
            xytext=(s80_pct + 4, 70),
            arrowprops=dict(arrowstyle="->", color=PALETTE["primary"], lw=1.2),
            fontweight="bold",
            color=PALETTE["primary"],
        )

        ax.set_title("SalesSight AI — Cumulative Pareto Concentration Curve (80/20 Rule)", pad=12)
        ax.set_xlabel("Cumulative Percentage of Catalog SKUs (N = 77)")
        ax.set_ylabel("Cumulative Percentage of Turnover (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_price_elasticity(self, filename: str = "step3_eda_price_elasticity_scatter.png") -> Path:
        """Renders scatter plot of UnitPrice vs Quantity colored by CategoryName."""
        logger.info("Rendering price elasticity scatter figure...")
        fig, ax = plt.subplots(figsize=(11, 7))

        sns.scatterplot(
            data=self.transactions,
            x="UnitPrice",
            y="Quantity",
            hue="CategoryName",
            palette="tab10",
            alpha=0.65,
            s=50,
            ax=ax,
        )

        sns.regplot(
            data=self.transactions,
            x="UnitPrice",
            y="Quantity",
            scatter=False,
            ax=ax,
            color="#2c3e50",
            line_kws={"linewidth": 2, "linestyle": "--", "label": "OLS Demand Curve Trendline"},
        )

        ax.set_title("SalesSight AI — Price-Demand Empirical Elasticity (UnitPrice vs Quantity)", pad=12)
        ax.set_xlabel("Derived Unit Catalog Price ($)")
        ax.set_ylabel("Line Item Order Quantity (Units)")
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
        ax.legend(title="Merchandise Category", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def render_all_figures(self) -> Dict[str, Path]:
        """Renders all standard EDA diagnostic charts and returns dictionary of saved paths."""
        logger.info("Rendering all publication-grade EDA diagnostic figures...")
        return {
            "target_distributions": self.plot_target_distributions(),
            "temporal_trajectory": self.plot_temporal_trajectory(),
            "seasonality_patterns": self.plot_seasonality_patterns(),
            "category_performance": self.plot_category_performance(),
            "pareto_concentration": self.plot_pareto_concentration(),
            "price_elasticity": self.plot_price_elasticity(),
        }


def run_eda_pipeline(
    processed_dir: str = "data/processed",
    output_dir: str = "report_stage",
) -> Dict[str, Any]:
    """
    Executes the end-to-end Step 3: Exploratory Data Analysis (EDA) pipeline.
    Loads processed dataframes, computes statistical metrics, generates high-res diagnostic
    visualizations stored directly in report_stage/assets/, and serializes JSON metrics.
    """
    logger.info("=" * 80)
    logger.info("SALESSIGHT AI: STEP 3 (EXPLORATORY DATA ANALYSIS) PIPELINE")
    logger.info("=" * 80)

    proc_path = Path(processed_dir)
    tx_file = proc_path / "cleaned_transactions.parquet"
    macro_file = proc_path / "all_products_daily.parquet"
    cat_file = proc_path / "category_daily.parquet"
    sku_file = proc_path / "sku_daily.parquet"

    for p in [tx_file, macro_file, cat_file, sku_file]:
        if not p.exists():
            raise FileNotFoundError(f"Required processed dataset missing: {p}. Run data_pipeline.py first.")

    logger.info("Ingesting processed datasets...")
    tx_df = pd.read_parquet(tx_file)
    macro_df = pd.read_parquet(macro_file)
    cat_df = pd.read_parquet(cat_file)
    sku_df = pd.read_parquet(sku_file)

    logger.info(f"Loaded Cleaned Transactions: {tx_df.shape[0]} rows, {tx_df.shape[1]} columns")
    logger.info(f"Loaded Macro Daily Series: {macro_df.shape[0]} continuous days")
    logger.info(f"Loaded Category Daily Grid: {cat_df.shape[0]} rows")
    logger.info(f"Loaded SKU Daily Series: {sku_df.shape[0]} rows")

    # 1. Statistical Analysis
    logger.info("Computing EDA statistical analytics...")
    analyzer = EDAAnalyzer(tx_df, macro_df, cat_df, sku_df)
    metrics = analyzer.run_full_analysis()

    # 2. Render diagnostic figures directly into report_stage/assets/ (single folder for all images)
    out_dir = Path(output_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering diagnostic visualization figures into unified folder: {assets_dir}")
    visualizer = EDAVisualizer(tx_df, macro_df, cat_df, sku_df, output_dir=assets_dir)
    figures = visualizer.render_all_figures()

    # Track figure paths in metrics
    metrics["figure_artifacts"] = {k: str(v) for k, v in figures.items()}

    # 3. Serialize metrics to JSON
    metrics_path = out_dir / "step3_eda_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Serialized complete Step 3 EDA metrics to: {metrics_path}")

    # Summary highlights
    tx_sales = metrics["target_distributions"]["transaction_level"]["sales"]
    macro_sales = metrics["target_distributions"]["macro_daily_level"]["sales"]
    pareto_sales_80 = metrics["pareto_concentration"]["sales_pareto"]["80_percent_revenue"]

    logger.info("-" * 80)
    logger.info("STEP 3 EXPLORATORY DATA ANALYSIS SUMMARY HIGHLIGHTS:")
    logger.info(f"  • Total Gross Revenue: ${tx_sales['sum']:,.2f}")
    logger.info(f"  • Transaction Sales Skewness: +{tx_sales['skewness']:.2f}")
    logger.info(f"  • Macro Daily Continuous Days: {macro_sales['count']} (Zero Days: {macro_sales['zero_pct']:.2f}%)")
    logger.info(f"  • Pareto Concentration: Top {pareto_sales_80['skus_required']} SKUs ({pareto_sales_80['sku_percentage']}%) generate 80% of Sales")
    logger.info(f"  • Diagnostic Charts: {len(figures)} figures saved directly in {assets_dir.name}/")
    logger.info("=" * 80)
    logger.info("STEP 3 EDA PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    return {
        "metrics": metrics,
        "figures": figures,
    }


if __name__ == "__main__":
    run_eda_pipeline()

