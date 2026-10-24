"""
Unit Test Suite for Step 12: Model Persistence, Packaging, and Provenance Audit
==============================================================================
Tests artifact serialization, SHA-256 cryptographic verification, ProductionModelPackage
inference execution, residual quantile calibration, and model manifest schema.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

from src.persistence import (
    ProvenanceAuditEngine,
    ResidualCalibrator,
    ProductionModelPackage,
    ModelPersistenceManager,
)


def test_provenance_audit_sha256_computation(tmp_path):
    """Test SHA-256 cryptographic hash computation and integrity verification."""
    import hashlib
    test_file = tmp_path / "sample_artifact.bin"
    payload = b"SalesSight AI Model Weights Binary Data"
    test_file.write_bytes(payload)

    expected_hash = hashlib.sha256(payload).hexdigest()
    computed_hash = ProvenanceAuditEngine.compute_sha256(test_file)
    assert computed_hash == expected_hash
    assert ProvenanceAuditEngine.verify_artifact_integrity(test_file, expected_hash)
    assert not ProvenanceAuditEngine.verify_artifact_integrity(test_file, "wrong_hash")


def test_residual_calibrator_quantiles(tmp_path):
    """Test step-wise empirical residual quantile generation and parquet storage."""
    calibrator = ResidualCalibrator(residuals_dir=tmp_path)
    y_true = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
    y_pred = np.array([90.0, 160.0, 190.0, 260.0, 290.0, 360.0])

    res = calibrator.calibrate_stepwise_residuals(y_true, y_pred, horizon_days=30, target_name="TestSales")
    assert res["sample_size"] == 6
    assert "overall_q10" in res
    assert "overall_q90" in res
    assert len(res["stepwise_quantiles"]) == 30
    assert Path(res["stepwise_parquet_path"]).exists()

    df_parquet = pd.read_parquet(res["stepwise_parquet_path"])
    assert len(df_parquet) == 30
    assert "q10_error" in df_parquet.columns
    assert "q90_error" in df_parquet.columns


def test_production_model_package_predict_and_intervals():
    """Test ProductionModelPackage point predictions and interval calculations."""
    class DummyEstimator:
        def predict(self, X):
            return np.ones(len(X)) * 100.0

    dummy_est = DummyEstimator()
    pkg = ProductionModelPackage(
        estimator=dummy_est,
        target_name="Sales",
        feature_columns=["f1", "f2"],
        model_version="v1.0.0-test",
        residual_quantiles={
            "overall_q10": -20.0,
            "overall_q90": 30.0,
            "stepwise_quantiles": [
                {"step_h": 1, "q10_error": -15.0, "q90_error": 25.0},
                {"step_h": 2, "q10_error": -20.0, "q90_error": 30.0},
            ],
        },
    )

    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    preds = pkg.predict(X)
    assert len(preds) == 2
    assert np.allclose(preds, [100.0, 100.0])

    intervals = pkg.predict_with_intervals(X)
    assert "predictions" in intervals
    assert "lower_bounds_80" in intervals
    assert "upper_bounds_80" in intervals
    # Lower bound = 100 + (-15) = 85, Upper bound = 100 + 25 = 125
    assert np.allclose(intervals["lower_bounds_80"], [85.0, 80.0])
    assert np.allclose(intervals["upper_bounds_80"], [125.0, 130.0])


def test_production_artifacts_exist_and_deserialize():
    """Verify production champion artifacts exist, are non-empty, and load properly."""
    sales_pkg_path = Path("models/artifacts/production/production_champion_sales_model.joblib")
    qty_pkg_path = Path("models/artifacts/production/production_champion_quantity_model.joblib")

    assert sales_pkg_path.exists(), "Production sales package artifact missing"
    assert qty_pkg_path.exists(), "Production quantity package artifact missing"

    sales_pkg = joblib.load(sales_pkg_path)
    qty_pkg = joblib.load(qty_pkg_path)

    assert isinstance(sales_pkg, ProductionModelPackage)
    assert isinstance(qty_pkg, ProductionModelPackage)
    assert sales_pkg.target_name == "Sales"
    assert qty_pkg.target_name == "Quantity"
    assert len(sales_pkg.feature_columns) > 10
    assert len(qty_pkg.feature_columns) > 10


def test_model_manifest_schema_and_integrity():
    """Verify models/artifacts/model_manifest.json conforms to Master Blueprint schema."""
    manifest_path = Path("models/artifacts/model_manifest.json")
    assert manifest_path.exists(), "model_manifest.json not found"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["manifest_version"] == "2.0.0"
    assert "data_provenance" in data
    assert "dataset_checksum_sha256" in data["data_provenance"]
    assert "production_selected_models" in data
    assert "Sales" in data["production_selected_models"]
    assert "Quantity" in data["production_selected_models"]

    sales_meta = data["production_selected_models"]["Sales"]
    assert sales_meta["display_name"] == "Model 5: Ridge Regression"
    assert sales_meta["validation_metrics"]["out_of_sample_wape"] == 92.78
    assert Path(sales_meta["package_artifact"]).exists()

    qty_meta = data["production_selected_models"]["Quantity"]
    assert qty_meta["display_name"] == "Model 5: Ridge Regression"
    assert qty_meta["validation_metrics"]["out_of_sample_wape"] == 88.43
    assert Path(qty_meta["package_artifact"]).exists()
