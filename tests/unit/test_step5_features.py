"""
Unit Test Suite for Step 5: Feature Engineering.
Validates feature matrix shapes, calendar harmonic encodings,
lookahead leakage assertions, commercial ratios, binning discretizations,
feature scaling / log-transforms, and diagnostic figure rendering.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.features import FeatureBuilder, FeatureTransformer, LeakageGuard, FeatureVisualizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def macro_daily_dataset():
    """Load continuous macro daily series dataset for testing."""
    return pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "all_products_daily.parquet")


def test_macro_feature_matrix_generation(macro_daily_dataset):
    builder = FeatureBuilder(macro_daily_dataset)
    features_df = builder.build_macro_feature_matrix()

    # 1,348 days - 28 warmup days = 1,320 rows
    assert features_df.shape[0] == 1320
    # Must have >= 40 total columns
    assert features_df.shape[1] >= 40
    # No NaNs permitted in the supervised feature matrix
    assert features_df.isna().sum().sum() == 0


def test_cyclical_harmonic_encodings(macro_daily_dataset):
    builder = FeatureBuilder(macro_daily_dataset)
    features_df = builder.build_macro_feature_matrix()

    # Harmonic bounds [-1.0, 1.0]
    for col in ["sin_dow", "cos_dow", "sin_month", "cos_month"]:
        assert features_df[col].min() >= -1.0 - 1e-6
        assert features_df[col].max() <= 1.0 + 1e-6

    # Trigonometric identity: sin^2 + cos^2 == 1.0
    dow_identity = features_df["sin_dow"]**2 + features_df["cos_dow"]**2
    month_identity = features_df["sin_month"]**2 + features_df["cos_month"]**2

    assert np.allclose(dow_identity, 1.0, atol=1e-5)
    assert np.allclose(month_identity, 1.0, atol=1e-5)


def test_leakage_guardrail_integrity(macro_daily_dataset):
    builder = FeatureBuilder(macro_daily_dataset)
    features_df = builder.build_macro_feature_matrix()

    # Verified features pass without error
    assert LeakageGuard.verify_lookahead_integrity(features_df, "lag_1_rev", target_col="Sales", shift_expected=1) is True
    assert LeakageGuard.verify_lookahead_integrity(features_df, "lag_7_rev", target_col="Sales", shift_expected=7) is True

    # Intentionally corrupted leakage feature (current observation t used directly)
    features_df["leaked_feature"] = features_df["Sales"]
    with pytest.raises(ValueError, match="Fatal leakage in leaked_feature"):
        LeakageGuard.verify_lookahead_integrity(features_df, "leaked_feature", target_col="Sales")


def test_commercial_ratios_and_binning(macro_daily_dataset):
    builder = FeatureBuilder(macro_daily_dataset)
    features_df = builder.build_macro_feature_matrix()

    # Ratio values must be strictly positive
    assert (features_df["asp_proxy_t1"] >= 0.0).all()
    assert (features_df["mom_ratio_7_28"] >= 0.0).all()
    assert (features_df["vol_ratio_7_28"] >= 0.0).all()

    # Binning tiers must be integers in [0, 1, 2]
    assert set(features_df["volume_tier"].unique()).issubset({0, 1, 2})
    assert set(features_df["price_tier"].unique()).issubset({0, 1, 2})


def test_feature_transformer_scaling_and_log(macro_daily_dataset):
    builder = FeatureBuilder(macro_daily_dataset)
    features_df = builder.build_macro_feature_matrix()

    feature_cols = ["lag_1_rev", "rolling_mean_7_rev", "asp_proxy_t1"]
    transformer = FeatureTransformer(feature_cols)

    # Log target transformation
    log_sales = transformer.transform_target_log(features_df["Sales"])
    assert (log_sales >= 0.0).all()
    inv_sales = transformer.inverse_transform_target_log(log_sales.values)
    assert np.allclose(inv_sales, features_df["Sales"].values, atol=1e-5)

    # Scaling
    transformer.fit_scaler(features_df)
    scaled_df = transformer.transform_features(features_df)
    for col in feature_cols:
        assert abs(scaled_df[col].mean()) < 1e-4
        assert abs(scaled_df[col].std() - 1.0) < 1e-2


def test_feature_visualizer_renders_figures(macro_daily_dataset, tmp_path):
    builder = FeatureBuilder(macro_daily_dataset)
    features_df = builder.build_macro_feature_matrix()

    visualizer = FeatureVisualizer(output_dir=tmp_path)
    figures = visualizer.render_all_figures(features_df)

    assert len(figures) == 4
    for name, path in figures.items():
        assert path.exists(), f"Figure {name} was not saved!"
        assert path.stat().st_size > 1000, f"Figure {name} file is too small/empty!"
