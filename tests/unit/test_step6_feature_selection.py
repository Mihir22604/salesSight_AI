"""
Unit Test Suite for Step 6: Feature Selection.
Validates Mutual Information calculations, Random Forest Gini importances,
chronological RFECV execution, consensus feature selection logic,
Before-vs-After model evaluation benchmarks, and diagnostic figure rendering.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.feature_selection import (
    FeatureSelector,
    FeatureSelectionEvaluator,
    FeatureSelectionVisualizer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def macro_feature_dataset():
    """Load macro features dataset generated in Step 5."""
    return pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "macro_features.parquet")


def test_mutual_information_computation(macro_feature_dataset):
    feature_cols = [c for c in macro_feature_dataset.columns if c not in ["Date", "Sales", "Quantity", "OrdersCount"]]
    X = macro_feature_dataset[feature_cols]
    y_sales = macro_feature_dataset["Sales"]
    y_qty = macro_feature_dataset["Quantity"]

    selector = FeatureSelector(X, y_sales, y_qty)
    mi_scores = selector.compute_mutual_information()

    assert "sales_mi" in mi_scores
    assert "quantity_mi" in mi_scores
    assert len(mi_scores["sales_mi"]) == len(feature_cols)

    # MI must be strictly non-negative
    for col, val in mi_scores["sales_mi"].items():
        assert val >= 0.0, f"Negative MI score for {col}: {val}"


def test_tree_importance_computation(macro_feature_dataset):
    feature_cols = [c for c in macro_feature_dataset.columns if c not in ["Date", "Sales", "Quantity", "OrdersCount"]]
    X = macro_feature_dataset[feature_cols]
    y_sales = macro_feature_dataset["Sales"]
    y_qty = macro_feature_dataset["Quantity"]

    selector = FeatureSelector(X, y_sales, y_qty)
    importances = selector.compute_tree_importance()

    assert len(importances) == len(feature_cols)
    total_imp = sum(importances.values())
    assert abs(total_imp - 1.0) < 1e-3, f"Gini importances must sum to 1.0 (got {total_imp})"


def test_rfecv_execution(macro_feature_dataset):
    feature_cols = [c for c in macro_feature_dataset.columns if c not in ["Date", "Sales", "Quantity", "OrdersCount"]][:15]
    X = macro_feature_dataset[feature_cols]
    y_sales = macro_feature_dataset["Sales"]
    y_qty = macro_feature_dataset["Quantity"]

    selector = FeatureSelector(X, y_sales, y_qty)
    rfecv_res = selector.run_rfecv(n_splits=3)

    assert rfecv_res["optimal_feature_count"] >= 10
    assert len(rfecv_res["selected_features"]) == rfecv_res["optimal_feature_count"]
    assert len(rfecv_res["feature_ranking"]) == 15


def test_consensus_feature_selection(macro_feature_dataset):
    feature_cols = [c for c in macro_feature_dataset.columns if c not in ["Date", "Sales", "Quantity", "OrdersCount"]]
    X = macro_feature_dataset[feature_cols]
    y_sales = macro_feature_dataset["Sales"]
    y_qty = macro_feature_dataset["Quantity"]

    selector = FeatureSelector(X, y_sales, y_qty)
    mi_scores = selector.compute_mutual_information()["sales_mi"]
    rf_importances = selector.compute_tree_importance()

    vif_prune_list = ["rolling_mean_7_rev", "rolling_std_7_rev"]
    rfecv_selected = list(feature_cols)[:20]

    selected = selector.select_consensus_features(
        mi_scores=mi_scores,
        rf_importances=rf_importances,
        rfecv_selected=rfecv_selected,
        vif_prune_list=vif_prune_list,
        target_count=24,
    )

    # Pruned VIF features must not be in selected set
    for pruned in vif_prune_list:
        assert pruned not in selected, f"Pruned feature {pruned} unexpectedly found in selected features!"

    # Essential anchors must be retained
    assert "lag_1_rev" in selected
    assert "lag_7_rev" in selected
    assert len(selected) <= 30


def test_evaluator_benchmark_execution(macro_feature_dataset):
    feature_cols = [c for c in macro_feature_dataset.columns if c not in ["Date", "Sales", "Quantity", "OrdersCount"]]
    X_raw = macro_feature_dataset[feature_cols]
    X_selected = macro_feature_dataset[feature_cols[:20]]
    y = macro_feature_dataset["Sales"]

    evaluator = FeatureSelectionEvaluator(X_raw, X_selected, y, n_splits=3)
    bench_results = evaluator.run_benchmark()

    # 3 models x 2 stages = 6 rows
    assert len(bench_results) == 6
    for row in bench_results:
        assert row["wape_pct"] > 0.0
        assert row["fit_latency_sec"] >= 0.0


def test_visualizer_renders_figures(macro_feature_dataset, tmp_path):
    feature_cols = [c for c in macro_feature_dataset.columns if c not in ["Date", "Sales", "Quantity", "OrdersCount"]]
    X_raw = macro_feature_dataset[feature_cols]
    X_selected = macro_feature_dataset[feature_cols[:20]]
    y = macro_feature_dataset["Sales"]

    selector = FeatureSelector(X_raw, y, y)
    mi_scores = selector.compute_mutual_information()
    rf_importances = selector.compute_tree_importance()
    rfecv_results = {
        "optimal_feature_count": 15,
        "selected_features": feature_cols[:15],
        "feature_ranking": {c: 1 for c in feature_cols},
        "cv_scores": [-1800.0, -1750.0, -1700.0, -1720.0],
        "grid_feature_counts": [10, 15, 20, 25],
    }

    evaluator = FeatureSelectionEvaluator(X_raw, X_selected, y, n_splits=3)
    bench_results = evaluator.run_benchmark()

    visualizer = FeatureSelectionVisualizer(output_dir=tmp_path)
    figures = visualizer.render_all_figures(
        mi_scores=mi_scores,
        rf_importances=rf_importances,
        rfecv_results=rfecv_results,
        benchmark_rows=bench_results,
    )

    assert len(figures) == 5
    for name, path in figures.items():
        assert path.exists(), f"Figure {name} was not saved!"
        assert path.stat().st_size > 1000, f"Figure {name} file is too small/empty!"
