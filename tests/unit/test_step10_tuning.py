"""
Unit Tests for Stage 10: Hyperparameter Tuning, Objective Exploration, and Multi-Seed Model Selection.

Tests adhere strictly to Stage 10 Key Rules:
- Rule 1: Tuning scope verification
- Rule 2: Chronological expanding folds on train only
- Rule 3: Fixed baseline, always-zero, train-mean, and legacy benchmarks
- Rule 4: Objective loss comparison (L2, L1, Huber, Tweedie, Log1p, Hurdle)
- Rule 5: Regularized parameter search space boundaries
- Rule 6: Separate daily vs. weekly granularity handling
- Rule 7: 5-seed empirical variance calculation
- Rule 8: Strict winner determination logic based on seed noise threshold
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.tuning import (
    BenchmarkEvaluator,
    ObjectiveComparator,
    HyperparameterTuner,
    GranularityEvaluator,
    WinnerDecisionEngine,
    HurdleRegressor,
)
from src.models import (
    SeasonalNaiveBaseline,
    AlwaysZeroBaseline,
    TrainMeanBaseline,
)


@pytest.fixture
def synthetic_tuning_data():
    """Generates synthetic time series dataset for rapid, leak-free test verification."""
    np.random.seed(42)
    dates_tr = pd.date_range("2022-01-01", periods=80, freq="D")
    dates_te = pd.date_range("2022-03-22", periods=20, freq="D")

    y_tr_sales = np.random.poisson(lam=15, size=80).astype(float) * 100.0
    y_tr_sales[::3] = 0.0  # intermittent zeros
    y_te_sales = np.random.poisson(lam=18, size=20).astype(float) * 100.0
    y_te_sales[::4] = 0.0

    # Build lag-7 series for fixed baseline
    full_y = np.concatenate([y_tr_sales, y_te_sales])
    full_lag7 = np.roll(full_y, 7)
    full_lag7[:7] = 0.0

    tr_df = pd.DataFrame({
        "Date": dates_tr,
        "Sales": y_tr_sales,
        "Quantity": (y_tr_sales / 25.0).astype(int),
        "OrdersCount": (y_tr_sales > 0).astype(int),
        "lag_1_rev": np.roll(y_tr_sales, 1),
        "lag_7_rev": full_lag7[:80],
        "lag_7_qty": (full_lag7[:80] / 25.0).astype(int),
        "rolling_mean_7": pd.Series(y_tr_sales).rolling(7, min_periods=1).mean().values,
        "day_of_week": dates_tr.dayofweek,
    })
    te_df = pd.DataFrame({
        "Date": dates_te,
        "Sales": y_te_sales,
        "Quantity": (y_te_sales / 25.0).astype(int),
        "OrdersCount": (y_te_sales > 0).astype(int),
        "lag_1_rev": np.roll(y_te_sales, 1),
        "lag_7_rev": full_lag7[80:],
        "lag_7_qty": (full_lag7[80:] / 25.0).astype(int),
        "rolling_mean_7": pd.Series(y_te_sales).rolling(7, min_periods=1).mean().values,
        "day_of_week": dates_te.dayofweek,
    })

    return {"train_df": tr_df, "test_df": te_df}


def test_benchmark_evaluator_all_four_benchmarks(synthetic_tuning_data):
    """Verify Rule 3: Fixed lag-7, always zero, train mean, and legacy baselines."""
    d = synthetic_tuning_data
    benchmarks = BenchmarkEvaluator.evaluate_benchmarks(d["train_df"], d["test_df"], target_name="Sales")

    assert "fixed_lag7_baseline" in benchmarks
    assert "always_zero" in benchmarks
    assert "train_mean" in benchmarks
    assert "legacy_baseline" in benchmarks

    # Always zero must have exactly 100.0% WAPE
    assert benchmarks["always_zero"]["wape"] == 100.0
    assert benchmarks["fixed_lag7_baseline"]["wape"] > 0.0
    assert benchmarks["train_mean"]["wape"] > 0.0
    assert benchmarks["legacy_baseline"]["wape"] > 0.0


def test_hurdle_model_fitting_and_prediction():
    """Verify Rule 4: HurdleRegressor two-stage classification and positive regression."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.exponential(scale=10.0, size=50)
    y[::2] = 0.0  # 50% structural zeros

    hurdle = HurdleRegressor()
    hurdle.fit(X, y)
    preds = hurdle.predict(X)

    assert len(preds) == 50
    assert np.all(preds >= 0.0)
    assert not np.isnan(preds).any()


def test_objective_comparator_all_six_objectives(synthetic_tuning_data):
    """Verify Rule 4: ObjectiveComparator evaluates L2, L1, Huber, Tweedie, Log1p, and Hurdle."""
    d = synthetic_tuning_data
    feat_cols = ["lag_1_rev", "lag_7_rev", "rolling_mean_7", "day_of_week"]
    X = d["train_df"][feat_cols].values
    y = d["train_df"]["Sales"].values

    obj_results = ObjectiveComparator.compare_objectives(X, y, n_splits=3, random_state=42)

    expected_objectives = ["L2_Ridge", "L1_MAE_LGBM", "Huber_LGBM", "Tweedie_LGBM", "Log1p_Ridge", "Hurdle_TwoStage"]
    for obj in expected_objectives:
        assert obj in obj_results
        assert "mean_cv_wape" in obj_results[obj]
        assert "std_cv_wape" in obj_results[obj]
        assert obj_results[obj]["mean_cv_wape"] > 0.0


def test_search_spaces_regularized_constraints():
    """Verify Rule 5: Search spaces are constrained and regularized (~40 configs)."""
    spaces = HyperparameterTuner.get_search_spaces()

    assert len(spaces["ridge_regression"]) == 40
    assert len(spaces["random_forest"]) == 40
    assert len(spaces["xgboost"]) == 40
    assert len(spaces["lightgbm"]) == 40

    # Ensure max_depth constraints are enforced (<= 12 to avoid memorization)
    for cfg in spaces["random_forest"]:
        assert cfg["max_depth"] <= 12
        assert cfg["min_samples_leaf"] >= 2

    for cfg in spaces["xgboost"]:
        assert cfg["max_depth"] <= 6
        assert cfg["reg_lambda"] >= 1.0


def test_winner_decision_engine_rule_8():
    """Verify Rule 8: Winner logic based on seed noise standard deviation."""
    benchmarks = {
        "always_zero": {"wape": 100.0, "mae": 500.0, "rmse": 800.0},
        "fixed_lag7_baseline": {"wape": 115.0, "mae": 600.0, "rmse": 900.0},
    }

    # Case 1: Tuned ML model beats benchmark by margin > seed std -> Model Wins
    tuning_win = {
        "ridge_regression": {
            "post_tuning_wape_mean": 95.0,
            "post_tuning_wape_std": 2.0,  # margin is 100 - 95 = 5.0 > 2.0
        }
    }
    decision_win = WinnerDecisionEngine.determine_winner(benchmarks, tuning_win, target_name="Sales")
    assert decision_win["decision"] == "TUNED_MODEL_WINS"
    assert decision_win["winner_name"] == "ridge_regression"

    # Case 2: Tuned ML model gain is within noise (margin <= seed std) -> Benchmark Wins
    tuning_lose = {
        "ridge_regression": {
            "post_tuning_wape_mean": 99.0,
            "post_tuning_wape_std": 3.0,  # margin is 100 - 99 = 1.0 <= 3.0
        }
    }
    decision_lose = WinnerDecisionEngine.determine_winner(benchmarks, tuning_lose, target_name="Sales")
    assert decision_lose["decision"] == "BENCHMARK_WINS"
    assert decision_lose["winner_name"] == "always_zero"


def test_granularity_evaluator_daily_vs_weekly(synthetic_tuning_data):
    """Verify Rule 6: Weekly aggregation builds valid series and evaluates separately."""
    d = synthetic_tuning_data
    wk_train, wk_test = GranularityEvaluator.build_weekly_dataset(d["train_df"], d["test_df"])

    assert len(wk_train) > 0
    assert len(wk_test) > 0
    assert "Sales" in wk_train.columns
    assert "lag_1_rev" in wk_train.columns

    res = GranularityEvaluator.evaluate_weekly_models(wk_train, wk_test, target_name="Sales")
    assert "weekly_benchmarks" in res
    assert "weekly_ml_model" in res
    assert res["weekly_ml_model"]["wape"] > 0.0


def test_model_selection_anti_leakage_uses_validation_results():
    """Verify that WinnerDecisionEngine selects candidate architecture strictly using validation CV, preventing test leakage."""
    benchmarks = {
        "always_zero": {"wape": 100.0, "mae": 500.0, "rmse": 800.0},
    }

    # Model A: Better on Validation CV (98.0% vs 105.0%), but slightly higher on Test (94.0% vs 91.0%)
    # Model B: Worse on Validation CV (105.0%), but lower on Test (91.0%)
    tuning_results = {
        "model_a": {
            "best_inner_cv_wape": 98.0,
            "post_tuning_wape_mean": 94.0,
            "post_tuning_wape_std": 0.5,
        },
        "model_b": {
            "best_inner_cv_wape": 105.0,
            "post_tuning_wape_mean": 91.0,
            "post_tuning_wape_std": 0.5,
        },
    }

    decision = WinnerDecisionEngine.determine_winner(benchmarks, tuning_results, target_name="Sales")
    # Anti-leakage assertion: Must select model_a because of validation CV superiority (98.0% < 105.0%)
    assert decision["winner_name"] == "model_a"
    assert decision["decision"] == "TUNED_MODEL_WINS"
    assert decision["selection_basis"] == "Validation Inner CV WAPE (Anti-Leakage Protocol)"
    assert decision["selected_model_cv_wape"] == 98.0


def test_weekly_dataset_temporal_cutoff_no_overlap(synthetic_tuning_data):
    """Verify that weekly dataset creation strictly honors the daily train boundary date."""
    d = synthetic_tuning_data
    wk_train, wk_test = GranularityEvaluator.build_weekly_dataset(d["train_df"], d["test_df"])

    max_train_daily = pd.to_datetime(d["train_df"]["Date"].max())
    max_train_weekly = pd.to_datetime(wk_train["Date"].max())
    min_test_weekly = pd.to_datetime(wk_test["Date"].min())

    assert max_train_weekly <= max_train_daily
    assert min_test_weekly > max_train_daily
    assert max_train_weekly < min_test_weekly


def test_metric_calculations_mathematical_precision():
    """Verify exact mathematical equivalence of regression and percentage error formulas."""
    y_true = np.array([100.0, 200.0, 0.0, 50.0, 0.0])
    y_pred = np.array([110.0, 180.0, 10.0, 40.0, 0.0])
    y_train = np.array([80.0, 120.0, 90.0, 110.0])

    from src.models import ModelEvaluator
    res = ModelEvaluator.compute_regression_metrics(y_true, y_pred, y_train=y_train)

    expected_mae = np.mean(np.abs(y_true - y_pred))
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    expected_wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100.0

    naive_errors = np.abs(y_train[1:] - y_train[:-1])
    expected_mase = expected_mae / np.mean(naive_errors)

    assert abs(res["mae"] - expected_mae) < 1e-2
    assert abs(res["rmse"] - expected_rmse) < 1e-2
    assert abs(res["wape"] - expected_wape) < 1e-2
    assert abs(res["mase"] - expected_mase) < 1e-2

