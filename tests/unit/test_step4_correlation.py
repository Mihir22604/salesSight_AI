"""
Unit Test Suite for Step 4: Correlation Analysis.
Validates Pearson/Spearman matrices, price elasticity relationships,
collinear pair screening, Variance Inflation Factor (VIF) auditing,
and diagnostic figure rendering.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.correlation import CorrelationAnalyzer, CorrelationVisualizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def correlation_datasets():
    """Load cleaned transaction and daily series datasets for testing."""
    tx_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.parquet")
    macro_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "all_products_daily.parquet")
    return tx_df, macro_df


def test_transaction_correlations(correlation_datasets):
    tx_df, macro_df = correlation_datasets
    analyzer = CorrelationAnalyzer(tx_df, macro_df)
    results = analyzer.compute_transaction_correlations()

    assert results["sample_size"] == 2172
    p_mat = pd.DataFrame(results["pearson_matrix"])
    s_mat = pd.DataFrame(results["spearman_matrix"])

    # Diagonal must be identically 1.0
    for col in p_mat.columns:
        assert abs(p_mat.loc[col, col] - 1.0) < 1e-6
        assert abs(s_mat.loc[col, col] - 1.0) < 1e-6

    # GrossSales and Sales must be near-perfectly collinear (|r| > 0.99)
    assert p_mat.loc["Sales", "GrossSales"] > 0.99
    assert s_mat.loc["Sales", "GrossSales"] > 0.99

    # UnitPrice should correlate positively with Sales revenue (r > 0.70)
    assert p_mat.loc["Sales", "UnitPrice"] > 0.70


def test_price_elasticity_economic_relationship(correlation_datasets):
    tx_df, macro_df = correlation_datasets
    analyzer = CorrelationAnalyzer(tx_df, macro_df)
    full_res = analyzer.run_full_analysis()

    elasticity = full_res["transaction_analysis"]["price_elasticity"]
    # Spearman rank correlation between UnitPrice and Quantity must be negative (confirming downward-sloping demand curve)
    assert elasticity["spearman_rho"] < 0.0


def test_vif_calculation_and_thresholding(correlation_datasets):
    tx_df, macro_df = correlation_datasets
    analyzer = CorrelationAnalyzer(tx_df, macro_df)

    # Test synthetic collinear matrix: b = 2*a + small noise -> severe VIF
    np.random.seed(42)
    a = np.random.randn(100)
    b = 2.0 * a + np.random.randn(100) * 0.01
    c = np.random.randn(100)
    synth_df = pd.DataFrame({"a": a, "b": b, "c": c})

    vif_res = analyzer.compute_vif(synth_df)
    assert len(vif_res) == 3
    # Features a and b should have extreme VIF (> 10.0)
    vif_map = {item["feature"]: item["vif"] for item in vif_res}
    assert vif_map["a"] > 10.0
    assert vif_map["b"] > 10.0
    assert vif_map["c"] < 5.0  # Independent feature


def test_collinear_pair_screening(correlation_datasets):
    tx_df, macro_df = correlation_datasets
    analyzer = CorrelationAnalyzer(tx_df, macro_df)
    tx_corr = analyzer.compute_transaction_correlations()
    p_mat = pd.DataFrame(tx_corr["pearson_matrix"])

    collinear_pairs = analyzer.screen_collinear_pairs(p_mat, threshold=0.85)
    assert len(collinear_pairs) >= 1

    # GrossSales vs Sales must be captured
    pair_features = [
        set([item["feature_1"], item["feature_2"]])
        for item in collinear_pairs
    ]
    assert {"Sales", "GrossSales"} in pair_features


def test_correlation_visualizer_renders_figures(correlation_datasets, tmp_path):
    tx_df, macro_df = correlation_datasets
    analyzer = CorrelationAnalyzer(tx_df, macro_df)
    results = analyzer.run_full_analysis()

    visualizer = CorrelationVisualizer(output_dir=tmp_path)
    figures = visualizer.render_all_figures(
        tx_corr=results["transaction_analysis"]["correlations"],
        ts_corr=results["timeseries_analysis"]["correlations"],
        ts_vif=results["timeseries_analysis"]["vif_analysis"],
    )

    assert len(figures) == 5
    for name, path in figures.items():
        assert path.exists(), f"Figure {name} was not saved!"
        assert path.stat().st_size > 1000, f"Figure {name} file is too small/empty!"
