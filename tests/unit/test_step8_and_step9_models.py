"""
Unit Tests for Steps 8 and 9: Model Building, Evaluation, and Classification Translation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models import (
    SeasonalNaiveBaseline,
    SARIMAForecastModel,
    ModelFactory,
    DemandDiscretizer,
    ModelEvaluator,
    ModelVisualizer,
)


@pytest.fixture
def synthetic_train_test_data():
    """Generates synthetic training and testing time series for fast test execution."""
    np.random.seed(42)
    dates_train = pd.date_range("2021-01-01", periods=60, freq="D")
    dates_test = pd.date_range("2021-03-02", periods=14, freq="D")

    y_train = np.random.poisson(lam=10, size=60).astype(float) * 100.0
    y_train[::4] = 0.0  # intermittent zeros

    y_test = np.random.poisson(lam=12, size=14).astype(float) * 100.0
    y_test[::3] = 0.0

    X_train = pd.DataFrame({
        "lag_1": np.roll(y_train, 1),
        "rolling_mean_7": pd.Series(y_train).rolling(7, min_periods=1).mean(),
        "day_of_week": dates_train.dayofweek,
    })
    X_test = pd.DataFrame({
        "lag_1": np.roll(y_test, 1),
        "rolling_mean_7": pd.Series(y_test).rolling(7, min_periods=1).mean(),
        "day_of_week": dates_test.dayofweek,
    })

    return {
        "dates_train": dates_train,
        "dates_test": dates_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,
        "X_test": X_test,
    }


def test_seasonal_naive_baseline(synthetic_train_test_data):
    """Verify SeasonalNaiveBaseline persistence and day-of-week fallback."""
    d = synthetic_train_test_data
    baseline = SeasonalNaiveBaseline(lag=7)
    baseline.fit(pd.Series(d["y_train"]), d["dates_train"])

    preds = baseline.predict(d["X_test"], d["dates_test"])
    assert len(preds) == len(d["dates_test"])
    assert np.all(preds >= 0.0)
    assert not np.isnan(preds).any()


def test_sarima_model_fit_and_predict(synthetic_train_test_data):
    """Verify SARIMAForecastModel fitting and multi-step forecast generation."""
    d = synthetic_train_test_data
    sarima = SARIMAForecastModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 7))
    sarima.fit(pd.Series(d["y_train"]))

    preds = sarima.predict(steps=len(d["y_test"]))
    assert len(preds) == len(d["y_test"])
    assert np.all(preds >= 0.0)


def test_model_factory_instantiation():
    """Verify that ModelFactory returns all 6 candidate models with valid interfaces."""
    models = ModelFactory.get_candidate_models(random_state=42)
    expected_keys = ["baseline", "ridge", "sarima", "random_forest", "xgboost", "lightgbm"]
    for k in expected_keys:
        assert k in models
        assert hasattr(models[k], "fit")
        assert hasattr(models[k], "predict")


def test_demand_discretizer_thresholds():
    """Verify continuous-to-discrete binary demand regime transformation (Class 0 vs Class 1)."""
    # Sales thresholds: Class 0: <= 1500, Class 1: > 1500
    sales_vals = np.array([0.0, 4.5, 50.0, 1500.0, 1501.0, 2050.0, 10000.0])
    sales_classes = DemandDiscretizer.discretize_sales(sales_vals, threshold_spike=1500.0)
    expected_sales_classes = np.array([0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(sales_classes, expected_sales_classes)

    # Quantity thresholds: Class 0: <= 40, Class 1: > 40
    qty_vals = np.array([0, 1, 35, 40, 41, 250])
    qty_classes = DemandDiscretizer.discretize_quantity(qty_vals, threshold_spike=40.0)
    expected_qty_classes = np.array([0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(qty_classes, expected_qty_classes)


def test_model_evaluator_regression_and_classification(synthetic_train_test_data):
    """Verify calculation of WAPE, MAE, RMSE, Confusion Matrix, and Classification Report."""
    d = synthetic_train_test_data
    evaluator = ModelEvaluator()

    # Synthetic predictions
    y_pred = d["y_test"] * 0.9 + 50.0

    # Regression metrics
    reg_m = evaluator.compute_regression_metrics(
        y_true=d["y_test"],
        y_pred=y_pred,
        y_train=d["y_train"],
    )
    assert "wape" in reg_m
    assert "mae" in reg_m
    assert "rmse" in reg_m
    assert "mase" in reg_m
    assert "r2" in reg_m
    assert reg_m["wape"] >= 0.0

    # Classification translation metrics
    cls_m = evaluator.compute_classification_metrics(
        y_true_cont=d["y_test"],
        y_pred_cont=y_pred,
        target_name="Sales",
    )
    assert "accuracy" in cls_m
    assert "macro_f1" in cls_m
    assert "confusion_matrix" in cls_m
    assert len(cls_m["confusion_matrix"]) == 2  # 2x2 binary matrix
    assert "classification_report" in cls_m
    assert "report_text" in cls_m
    assert "Class 0" in cls_m["report_text"]
    assert "Class 1" in cls_m["report_text"]


def test_insufficiency_of_accuracy_audit(synthetic_train_test_data):
    """Verify the technical proof metrics on intermittent data."""
    d = synthetic_train_test_data
    evaluator = ModelEvaluator()

    audit = evaluator.audit_insufficiency_of_accuracy(
        y_true=d["y_test"],
        y_pred=d["y_test"] * 0.9,
    )
    assert "zero_percentage" in audit
    assert "mape_division_by_zero_count" in audit
    assert "trivial_always_zero_accuracy_pct" in audit
    assert "mathematical_takeaway" in audit


def test_visualizer_renders_classification_and_regression_figures():
    """Verify that all classification and regression figures exist in report_stage/assets/."""
    assets_dir = Path("report_stage/assets")
    
    # 6 Individual Classification Reports
    for m in ["baseline", "ridge", "sarima", "random_forest", "xgboost", "lightgbm"]:
        fig_path = assets_dir / f"step9_classification_report_{m}.png"
        assert fig_path.exists(), f"Missing individual classification report: {fig_path}"
        assert fig_path.stat().st_size > 1000

    # Combined Classification Report
    combined_fig = assets_dir / "step9_classification_report_combined.png"
    assert combined_fig.exists(), f"Missing combined classification report: {combined_fig}"
    assert combined_fig.stat().st_size > 1000

    # Regression Visual Artifacts
    fig_act = assets_dir / "step8_model_actual_vs_predicted_comparison.png"
    fig_reg = assets_dir / "step9_eval_metrics_regression_benchmark.png"
    fig_res = assets_dir / "step9_eval_residual_distributions.png"
    assert fig_act.exists(), f"Missing actual vs predicted: {fig_act}"
    assert fig_reg.exists(), f"Missing regression benchmark: {fig_reg}"
    assert fig_res.exists(), f"Missing residual distributions: {fig_res}"


def test_ridge_regression_model_specification_and_pipeline(synthetic_train_test_data):
    """Verify Model 5: Ridge Regression pipeline structure, alpha regularization, and predictions."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    d = synthetic_train_test_data
    models = ModelFactory.get_candidate_models(random_state=42)
    assert "ridge" in models, "Model 5: Ridge Regression must be instantiated by ModelFactory"
    
    ridge_pipe = models["ridge"]
    assert isinstance(ridge_pipe, Pipeline), "Ridge must be encapsulated in a StandardScaler pipeline"
    assert isinstance(ridge_pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(ridge_pipe.named_steps["reg"], Ridge)
    assert ridge_pipe.named_steps["reg"].alpha == 10.0

    # Fit and Predict
    ridge_pipe.fit(d["X_train"], d["y_train"])
    preds = ridge_pipe.predict(d["X_test"])
    assert len(preds) == len(d["y_test"])
    assert not np.isnan(preds).any()

