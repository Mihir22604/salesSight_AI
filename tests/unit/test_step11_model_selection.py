"""
Unit tests for Stage 11: Final Model Selection and Tuned Classification Evaluation.
Tests:
1. MultiCriteriaSelectionEngine: Clear WAPE winner selection
2. MultiCriteriaSelectionEngine: 2% relative WAPE tolerance tie-break on RMSE
3. MultiCriteriaSelectionEngine: Tie-break on Occam's razor (simpler linear architecture)
4. MultiCriteriaSelectionEngine: Fallback to baseline safety rule
5. MultiCriteriaSelectionEngine: Sub-500ms latency constraint filter
6. TunedModelEvaluator: Holdout regression and operational classification translation
7. TunedClassificationVisualizer: Renders individual and combined PNG report cards
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.selection import (
    MultiCriteriaSelectionEngine,
    TunedModelEvaluator,
    TunedClassificationVisualizer,
)


@pytest.fixture
def mock_selection_fixtures():
    """Provides synthetic regression and benchmark metrics for selection testing."""
    benchmarks = {
        "always_zero": {"wape": 100.0, "mae": 500.0, "rmse": 800.0},
        "fixed_lag7_baseline": {"wape": 115.0, "mae": 600.0, "rmse": 900.0},
        "legacy_baseline": {"wape": 99.48, "mae": 450.0, "rmse": 750.0},
    }

    # Case 1: Clear WAPE Winner (Ridge is 92.78%, LightGBM is 97.00% -> rel gap = 4.5% > 2%)
    reg_clear = {
        "baseline": {"wape": 99.48, "rmse": 750.0, "mae": 450.0, "r2": 0.05, "pred_time_ms": 0.5},
        "ridge_regression": {"wape": 92.78, "rmse": 700.0, "mae": 400.0, "r2": 0.15, "pred_time_ms": 1.2},
        "lightgbm": {"wape": 97.00, "rmse": 720.0, "mae": 410.0, "r2": 0.12, "pred_time_ms": 4.5},
    }
    scorecard_clear = {
        "ridge_regression": {"post_tuning_wape_std": 0.0},
        "lightgbm": {"post_tuning_wape_std": 0.25},
    }

    # Case 2: Within 2% Relative WAPE -> Tie broken by RMSE
    # Ridge: 94.0%, RMSE 750.0 | LightGBM: 93.0%, RMSE 680.0 | Rel gap = 1.07% <= 2%
    reg_tie_rmse = {
        "baseline": {"wape": 99.48, "rmse": 750.0, "mae": 450.0, "r2": 0.05, "pred_time_ms": 0.5},
        "ridge_regression": {"wape": 94.0, "rmse": 750.0, "mae": 410.0, "r2": 0.12, "pred_time_ms": 1.2},
        "lightgbm": {"wape": 93.0, "rmse": 680.0, "mae": 400.0, "r2": 0.16, "pred_time_ms": 4.5},
    }
    scorecard_tie_rmse = {
        "ridge_regression": {"post_tuning_wape_std": 0.0},
        "lightgbm": {"post_tuning_wape_std": 0.25},
    }

    # Case 3: Both WAPE and RMSE tied -> Occam's razor selects simpler model (Ridge > LightGBM)
    reg_tie_simplicity = {
        "baseline": {"wape": 99.48, "rmse": 750.0, "mae": 450.0, "r2": 0.05, "pred_time_ms": 0.5},
        "ridge_regression": {"wape": 93.0, "rmse": 700.0, "mae": 400.0, "r2": 0.15, "pred_time_ms": 1.2},
        "lightgbm": {"wape": 93.0, "rmse": 700.0, "mae": 400.0, "r2": 0.15, "pred_time_ms": 4.5},
    }

    # Case 4: No ML model beats benchmark -> Fallback to Baseline
    reg_fallback = {
        "baseline": {"wape": 99.48, "rmse": 750.0, "mae": 450.0, "r2": 0.05, "pred_time_ms": 0.5},
        "ridge_regression": {"wape": 105.0, "rmse": 800.0, "mae": 500.0, "r2": -0.1, "pred_time_ms": 1.2},
        "lightgbm": {"wape": 102.0, "rmse": 780.0, "mae": 490.0, "r2": -0.05, "pred_time_ms": 4.5},
    }

    return {
        "benchmarks": benchmarks,
        "reg_clear": reg_clear,
        "scorecard_clear": scorecard_clear,
        "reg_tie_rmse": reg_tie_rmse,
        "scorecard_tie_rmse": scorecard_tie_rmse,
        "reg_tie_simplicity": reg_tie_simplicity,
        "reg_fallback": reg_fallback,
    }


def test_multi_criteria_selection_clear_wape_winner(mock_selection_fixtures):
    """Verify that a model with WAPE exceeding 2% relative tolerance wins as CLEAR_WAPE_WINNER."""
    d = mock_selection_fixtures
    selector = MultiCriteriaSelectionEngine()
    verdict = selector.select_best_model(
        d["reg_clear"], d["scorecard_clear"], d["benchmarks"], target_name="Sales"
    )

    assert verdict["selected_model"] == "ridge_regression"
    assert verdict["decision_type"] == "CLEAR_WAPE_WINNER"
    assert verdict["winner_wape"] == 92.78
    assert not verdict["is_fallback"]


def test_multi_criteria_selection_tie_broken_by_rmse(mock_selection_fixtures):
    """Verify that models within 2% relative WAPE break ties by strictly lower RMSE."""
    d = mock_selection_fixtures
    selector = MultiCriteriaSelectionEngine()
    verdict = selector.select_best_model(
        d["reg_tie_rmse"], d["scorecard_tie_rmse"], d["benchmarks"], target_name="Sales"
    )

    assert verdict["selected_model"] == "lightgbm"
    assert verdict["decision_type"] == "TIE_BROKEN_BY_RMSE"
    assert verdict["winner_rmse"] == 680.0
    assert len(verdict["tied_candidates"]) == 2


def test_multi_criteria_selection_tie_broken_by_simplicity(mock_selection_fixtures):
    """Verify that when both WAPE and RMSE are tied within tolerance, Occam's razor selects simpler model."""
    d = mock_selection_fixtures
    selector = MultiCriteriaSelectionEngine()
    verdict = selector.select_best_model(
        d["reg_tie_simplicity"], d["scorecard_clear"], d["benchmarks"], target_name="Sales"
    )

    assert verdict["selected_model"] == "ridge_regression"
    assert verdict["decision_type"] == "TIE_BROKEN_BY_SIMPLICITY"


def test_multi_criteria_selection_fallback_to_baseline(mock_selection_fixtures):
    """Verify that if no ML model outperforms the benchmark, the system safely falls back to baseline."""
    d = mock_selection_fixtures
    selector = MultiCriteriaSelectionEngine()
    verdict = selector.select_best_model(
        d["reg_fallback"], d["scorecard_clear"], d["benchmarks"], target_name="Sales"
    )

    assert verdict["selected_model"] == "baseline"
    assert verdict["decision_type"] == "FALLBACK_TO_BASELINE"
    assert verdict["is_fallback"] is True


def test_latency_constraint_enforcement(mock_selection_fixtures):
    """Verify that candidate models exceeding the 500ms latency ceiling are disqualified."""
    d = mock_selection_fixtures
    # Artificially set Ridge latency to 600ms (> 500ms ceiling)
    reg_slow = {
        "baseline": {"wape": 99.48, "rmse": 750.0, "mae": 450.0, "r2": 0.05, "pred_time_ms": 0.5},
        "ridge_regression": {"wape": 92.78, "rmse": 700.0, "mae": 400.0, "r2": 0.15, "pred_time_ms": 600.0},
        "lightgbm": {"wape": 95.00, "rmse": 720.0, "mae": 410.0, "r2": 0.12, "pred_time_ms": 4.5},
    }
    selector = MultiCriteriaSelectionEngine()
    verdict = selector.select_best_model(
        reg_slow, d["scorecard_clear"], d["benchmarks"], target_name="Sales", latency_threshold_ms=500.0
    )

    assert verdict["selected_model"] == "lightgbm"
    assert verdict["winner_latency_ms"] == 4.5


def test_tuned_classification_reports_structure():
    """Verify that operational regime classification produces valid confusion matrix and scores."""
    from src.models import DemandDiscretizer, ModelEvaluator

    y_true = np.array([1000.0, 2000.0, 500.0, 1800.0, 0.0])
    y_pred = np.array([1200.0, 2200.0, 400.0, 1400.0, 100.0])

    cls_res = ModelEvaluator.compute_classification_metrics(y_true, y_pred, target_name="Sales")
    assert "accuracy_pct" in cls_res
    assert "confusion_matrix" in cls_res
    assert "report_text" in cls_res
    assert len(cls_res["confusion_matrix"]) == 2
    assert "Class 1" in cls_res["classification_report"]


def test_visualizer_generates_combined_and_individual_images(tmp_path):
    """Verify that TunedClassificationVisualizer renders polished PNG artifacts on disk."""
    from src.models import ModelEvaluator

    y_true = np.array([1000.0, 2000.0, 500.0, 1800.0, 0.0, 3000.0])
    y_pred = np.array([1200.0, 2200.0, 400.0, 1400.0, 100.0, 2900.0])

    m_cls = ModelEvaluator.compute_classification_metrics(y_true, y_pred, target_name="Sales")
    all_models_cls = {
        "baseline": m_cls,
        "ridge_regression": m_cls,
        "lightgbm": m_cls,
        "xgboost": m_cls,
        "random_forest": m_cls,
        "sarima": m_cls,
    }

    viz = TunedClassificationVisualizer(output_dir=tmp_path)
    p_indiv = viz.plot_individual_report(m_cls, "ridge_regression", target_name="Sales")
    p_comb = viz.plot_combined_classification_reports(all_models_cls, target_name="Sales")
    p_cm = viz.plot_confusion_matrices(all_models_cls, target_name="Sales")

    assert p_indiv.exists()
    assert p_indiv.stat().st_size > 1000
    assert p_comb.exists()
    assert p_comb.stat().st_size > 1000
    assert p_cm.exists()
    assert p_cm.stat().st_size > 1000
