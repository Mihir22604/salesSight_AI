"""
Unit Test Suite for Step 3: Exploratory Data Analysis (EDA).
Validates statistical calculations, moments, Pareto distribution,
temporal cycles, and diagnostic visualization generation.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.eda import EDAAnalyzer, EDAVisualizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def eda_datasets():
    """Load cleaned transaction and daily series datasets for testing."""
    tx_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.parquet")
    macro_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "all_products_daily.parquet")
    cat_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "category_daily.parquet")
    sku_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "sku_daily.parquet")
    return tx_df, macro_df, cat_df, sku_df


def test_eda_target_distributions(eda_datasets):
    tx_df, macro_df, cat_df, sku_df = eda_datasets
    analyzer = EDAAnalyzer(tx_df, macro_df, cat_df, sku_df)
    dist = analyzer.analyze_target_distributions()

    # Transaction level assertions
    tx_sales = dist["transaction_level"]["sales"]
    tx_qty = dist["transaction_level"]["quantity"]
    assert tx_sales["count"] == 2172
    assert tx_sales["min"] > 0
    assert tx_sales["skewness"] > 5.0  # Highly right-skewed
    assert tx_qty["min"] >= 1
    assert tx_qty["median"] == 20.0

    # Macro daily level assertions
    macro_sales = dist["macro_daily_level"]["sales"]
    assert macro_sales["count"] == 1348
    assert macro_sales["zero_count"] == 815
    assert abs(macro_sales["zero_pct"] - 60.46) < 0.1


def test_eda_category_performance(eda_datasets):
    tx_df, macro_df, cat_df, sku_df = eda_datasets
    analyzer = EDAAnalyzer(tx_df, macro_df, cat_df, sku_df)
    cat_perf = analyzer.analyze_category_performance()

    assert cat_perf["total_categories"] == 8
    # Total revenue across categories must equal $1,783,147.70
    total_sales = sum(c["total_sales"] for c in cat_perf["category_rankings"])
    assert abs(total_sales - 1783147.70) < 1.0


def test_eda_pareto_concentration(eda_datasets):
    tx_df, macro_df, cat_df, sku_df = eda_datasets
    analyzer = EDAAnalyzer(tx_df, macro_df, cat_df, sku_df)
    pareto = analyzer.analyze_pareto_concentration()

    assert pareto["total_skus"] == 77
    assert len(pareto["top_10_by_revenue"]) == 10
    assert len(pareto["top_10_by_quantity"]) == 10

    # Verify 80% sales concentration requires <= 40% of SKUs
    s80 = pareto["sales_pareto"]["80_percent_revenue"]
    assert s80["sku_percentage"] <= 40.0
    assert s80["skus_required"] <= 35


def test_eda_temporal_dynamics(eda_datasets):
    tx_df, macro_df, cat_df, sku_df = eda_datasets
    analyzer = EDAAnalyzer(tx_df, macro_df, cat_df, sku_df)
    temporal = analyzer.analyze_temporal_dynamics()

    # 7 days of week
    assert len(temporal["day_of_week_seasonality"]) == 7
    # 12 months
    assert len(temporal["monthly_seasonality"]) == 12
    # 4 calendar years
    assert len(temporal["annual_progression"]) == 4


def test_eda_visualizer_renders_figures(eda_datasets, tmp_path):
    tx_df, macro_df, cat_df, sku_df = eda_datasets
    visualizer = EDAVisualizer(tx_df, macro_df, cat_df, sku_df, output_dir=tmp_path)
    figures = visualizer.render_all_figures()

    assert len(figures) == 6
    for name, path in figures.items():
        assert path.exists(), f"Figure {name} was not saved!"
        assert path.stat().st_size > 1000, f"Figure {name} file is too small/empty!"
