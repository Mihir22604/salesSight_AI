"""
Unit Test Suite for Step 13: Build an ML API (FastAPI REST Server)
==================================================================
Tests all REST endpoints (/health, /forecast, /models, /intelligence),
validates start-date mode resolutions, schema validation, and error boundaries.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify GET /health returns HTTP 200 OK with healthy status and loaded models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service_name"] == "SalesSight AI ML Inference Service"
    assert data["artifacts_verified"] is True
    assert "Sales" in data["production_models_loaded"]
    assert "Quantity" in data["production_models_loaded"]
    assert data["last_historical_date"] == "2014-09-29"


def test_api_forecast_default_mode():
    """Verify POST /forecast in Default Mode (30d Sales forecast from 2014-09-30)."""
    payload = {
        "target": "Sales",
        "scope": "All",
        "horizon_days": 30,
        "start_date": "2014-09-30",
        "enable_demo_mode": False,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["start_date_mode"] == "DEFAULT"
    assert data["horizon_days"] == 30
    assert "Sales" in data["targets"]

    sales_resp = data["targets"]["Sales"]
    assert len(sales_resp["forecasts"]) == 30
    first_pt = sales_resp["forecasts"][0]
    assert first_pt["date"] == "2014-09-30"
    assert first_pt["predicted_value"] >= 0.0
    assert first_pt["lower_bound_80"] >= 0.0
    assert first_pt["upper_bound_80"] >= first_pt["lower_bound_80"]

    summary = sales_resp["metrics_summary"]
    assert summary["total_forecast"] > 0.0
    assert summary["daily_average"] > 0.0
    assert summary["peak_value"] >= summary["lowest_value"]


def test_api_forecast_both_targets_60d():
    """Verify POST /forecast with Dual Target (Both) for 60-day horizon."""
    payload = {
        "target": "Both",
        "scope": "All",
        "horizon_days": 60,
        "start_date": "2014-09-30",
        "enable_demo_mode": False,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "Sales" in data["targets"]
    assert "Quantity" in data["targets"]
    assert len(data["targets"]["Sales"]["forecasts"]) == 60
    assert len(data["targets"]["Quantity"]["forecasts"]) == 60


def test_api_forecast_bridge_mode():
    """Verify POST /forecast in Bridge Mode (dates within 90 days of 2014-09-29)."""
    payload = {
        "target": "Sales",
        "horizon_days": 30,
        "start_date": "2014-10-15",
        "enable_demo_mode": False,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["start_date_mode"] == "BRIDGE"
    assert "Bridge Mode" in data["banner_message"]
    assert data["start_date"] == "2014-10-15"
    assert len(data["targets"]["Sales"]["forecasts"]) == 30


def test_api_forecast_beyond_limit_mode_rejected():
    """Verify POST /forecast beyond 90-day forward limit returns HTTP 400."""
    payload = {
        "target": "Sales",
        "horizon_days": 30,
        "start_date": "2026-10-24",
        "enable_demo_mode": False,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data["detail"]
    assert data["detail"]["error"] == "DateBeyondLimitError"


def test_api_forecast_demonstration_mode():
    """Verify POST /forecast with Demonstration Mode enabled for 2026 dates."""
    payload = {
        "target": "Sales",
        "horizon_days": 30,
        "start_date": "2026-10-24",
        "enable_demo_mode": True,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["start_date_mode"] == "DEMONSTRATION"
    assert "DEMONSTRATION ONLY" in data["banner_message"]
    assert data["start_date"] == "2026-10-24"
    assert len(data["targets"]["Sales"]["forecasts"]) == 30


def test_api_forecast_invalid_horizon_validation():
    """Verify invalid horizon values (e.g., 45 days) trigger Pydantic HTTP 422."""
    payload = {
        "target": "Sales",
        "horizon_days": 45,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 422


def test_api_models_endpoint():
    """Verify GET /models returns production model manifest scorecards and provenance."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["manifest_version"] == "2.0.0"
    assert "Sales" in data["production_selected_models"]
    assert "Quantity" in data["production_selected_models"]
    assert data["total_artifacts_audited"] > 0


def test_api_intelligence_endpoint():
    """Verify GET /intelligence returns Top-5 ranking, momentum, attention list, and disclaimer."""
    response = client.get("/intelligence")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert len(data["top_5_demand_ranking"]) == 5
    assert len(data["planning_attention_list"]) > 0
    assert len(data["category_rankings"]) > 0
    assert len(data["narrative_insights"]) > 0
    assert "Planning recommendations reflect relative demand velocity" in data["mandatory_inventory_disclaimer"]
