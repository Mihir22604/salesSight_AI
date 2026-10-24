"""
Unit Tests for Step 7: Train / Test Split Module (TimeSeriesSplitter & SplitVisualizer)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.split import TimeSeriesSplitter, SplitVisualizer, run_split_pipeline


@pytest.fixture
def synthetic_macro_df():
    """Create a deterministic synthetic time-series dataset with 100 daily points."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    sales = np.random.poisson(lam=5, size=100).astype(float)
    # inject zero sales
    sales[::3] = 0.0
    qty = sales * 2.0

    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales,
        "Quantity": qty,
        "Lag_1": np.roll(sales, 1),
        "Rolling_Mean_7": pd.Series(sales).rolling(7, min_periods=1).mean(),
        "DayOfWeek": dates.dayofweek,
        "IsWeekend": dates.dayofweek.isin([5, 6]).astype(int),
    })
    return df


def test_canonical_holdout_split_integrity(synthetic_macro_df):
    """Verify 80:20 split ratio and absolute absence of temporal lookahead leakage."""
    splitter = TimeSeriesSplitter(date_col="Date")
    train_df, test_df, meta = splitter.create_canonical_holdout_split(
        synthetic_macro_df, train_ratio=0.80
    )

    # 1. Total row count matches
    assert len(train_df) + len(test_df) == len(synthetic_macro_df)
    assert len(train_df) == 80
    assert len(test_df) == 20

    # 2. Strict chronological boundary check
    max_train_date = train_df["Date"].max()
    min_test_date = test_df["Date"].min()
    assert max_train_date < min_test_date
    assert (min_test_date - max_train_date).days == 1

    # 3. Metadata assertions
    assert meta["train_ratio"] == 0.80
    assert meta["train_observations"] == 80
    assert meta["test_observations"] == 20
    assert meta["calendar_gap_days"] == 0


def test_expanding_folds_generation(synthetic_macro_df):
    """Verify K-fold expanding-window cross-validation fold generation."""
    splitter = TimeSeriesSplitter(date_col="Date")
    # For 100 rows, test K=3 folds: initial=60, horizon=10
    folds = splitter.create_expanding_folds(
        synthetic_macro_df, initial_train_size=60, horizon=10, n_splits=3
    )

    assert len(folds) == 3

    prev_train_len = 0
    for fold in folds:
        train_len = fold["train_days_count"]
        test_len = fold["test_days_count"]

        # Train set must expand monotonically
        assert train_len > prev_train_len
        prev_train_len = train_len

        # Test set must match horizon
        assert test_len == 10

        # Boundary condition
        assert fold["train_end_date"] < fold["test_start_date"]


def test_distribution_parity_audit(synthetic_macro_df):
    """Verify that statistical parity and zero density metrics are computed accurately."""
    splitter = TimeSeriesSplitter(date_col="Date")
    train_df, test_df, _ = splitter.create_canonical_holdout_split(synthetic_macro_df, 0.80)

    audit = splitter.audit_statistical_representativeness(
        train_df, test_df, target_cols=["Sales", "Quantity"]
    )

    assert "Sales" in audit
    assert "Quantity" in audit
    sales_audit = audit["Sales"]
    assert "ks_test" in sales_audit
    assert "statistic" in sales_audit["ks_test"]
    assert "p_value" in sales_audit["ks_test"]
    assert "train" in sales_audit
    assert "test" in sales_audit
    assert 0.0 <= sales_audit["train"]["zero_pct"] <= 100.0
    assert 0.0 <= sales_audit["test"]["zero_pct"] <= 100.0


def test_visualizer_renders_split_figures():
    """Verify that all 3 publication figures exist in report_stage/assets/."""
    assets_dir = Path("report_stage/assets")
    fig1 = assets_dir / "step7_split_timeline_visualization.png"
    fig2 = assets_dir / "step7_split_target_distribution_comparison.png"
    fig3 = assets_dir / "step7_split_sparsity_and_zero_density.png"

    assert fig1.exists(), f"Missing {fig1}"
    assert fig2.exists(), f"Missing {fig2}"
    assert fig3.exists(), f"Missing {fig3}"
    assert fig1.stat().st_size > 1000
    assert fig2.stat().st_size > 1000
    assert fig3.stat().st_size > 1000
