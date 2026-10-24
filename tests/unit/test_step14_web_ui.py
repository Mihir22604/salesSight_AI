"""
Unit Test Suite for Step 14: Build a Web UI (Streamlit Dashboard & Components)
=============================================================================
Tests UI component logic, client communication, Plotly forecast chart generation,
RFC-4180 CSV export generation, and error boundary handling.
"""

import io
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.ui.client import ForecastAPIClient
from src.ui.components import UIComponents


@pytest.fixture
def api_client():
    return ForecastAPIClient()


def test_ui_client_health_check(api_client):
    """Verify ForecastAPIClient retrieves health status properly."""
    health = api_client.check_health()
    assert health["status"] in ["healthy", "degraded"]
    assert "production_models_loaded" in health
    assert "backend_mode" in health


def test_ui_client_get_forecast_in_process(api_client):
    """Verify ForecastAPIClient executes forecast queries properly."""
    payload = {
        "target": "Both",
        "scope": "All",
        "horizon_days": 30,
        "start_date": "2014-09-30",
        "enable_demo_mode": False,
    }
    res = api_client.get_forecast(payload)
    assert res["status"] == "success"
    assert "Sales" in res["targets"]
    assert "Quantity" in res["targets"]
    assert len(res["targets"]["Sales"]["forecasts"]) == 30


def test_ui_components_build_forecast_plot():
    """Verify Plotly figure builder creates 3 traces with calibrated 80% ribbons."""
    dummy_points = [
        {"date": "2014-09-30", "predicted_value": 1400.0, "lower_bound_80": 1100.0, "upper_bound_80": 1700.0, "day_of_week": "Tuesday"},
        {"date": "2014-10-01", "predicted_value": 1550.0, "lower_bound_80": 1200.0, "upper_bound_80": 1900.0, "day_of_week": "Wednesday"},
        {"date": "2014-10-02", "predicted_value": 1300.0, "lower_bound_80": 1050.0, "upper_bound_80": 1600.0, "day_of_week": "Thursday"},
    ]
    fig = UIComponents.build_forecast_plot(dummy_points, target_name="Sales")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # Upper bound, filled ribbon, point forecast
    # Verify ribbon fill
    assert fig.data[1].fill == "tonexty"
    # Verify point line
    assert "Point Forecast" in fig.data[2].name


def test_ui_components_rfc4180_csv_exporter():
    """Verify RFC-4180 CSV export generation with metadata headers."""
    dummy_response = {
        "horizon_days": 30,
        "start_date": "2014-09-30",
        "start_date_mode": "DEFAULT",
        "targets": {
            "Sales": {
                "forecasts": [
                    {"date": "2014-09-30", "predicted_value": 1420.5, "lower_bound_80": 1150.0, "upper_bound_80": 1700.0},
                    {"date": "2014-10-01", "predicted_value": 1510.2, "lower_bound_80": 1210.0, "upper_bound_80": 1820.0},
                ]
            },
            "Quantity": {
                "forecasts": [
                    {"date": "2014-09-30", "predicted_value": 35.0, "lower_bound_80": 25.0, "upper_bound_80": 45.0},
                    {"date": "2014-10-01", "predicted_value": 38.0, "lower_bound_80": 28.0, "upper_bound_80": 48.0},
                ]
            },
        },
    }

    csv_text = UIComponents.generate_rfc4180_csv(dummy_response, scope="Category", scope_id="Beverages")
    assert "SalesSight AI — Daily Sales Forecast Export" in csv_text
    assert "Forecast Scope: Category (Beverages)" in csv_text
    assert "Date,Predicted_Revenue_USD,Revenue_Lower_80,Revenue_Upper_80,Predicted_Quantity_Units" in csv_text

    # Parse into pandas to verify RFC-4180 structure
    data_lines = [line for line in csv_text.splitlines() if not line.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    assert len(df) == 2
    assert "Predicted_Revenue_USD" in df.columns
    assert "Predicted_Quantity_Units" in df.columns
    assert df.loc[0, "Predicted_Revenue_USD"] == 1420.5
    assert df.loc[0, "Predicted_Quantity_Units"] == 35.0


def test_ui_client_beyond_limit_error_remediation(api_client):
    """Verify error response structure when beyond-limit date is queried without demo mode."""
    payload = {
        "target": "Sales",
        "horizon_days": 30,
        "start_date": "2026-10-24",
        "enable_demo_mode": False,
    }
    res = api_client.get_forecast(payload)
    assert "detail" in res
    assert res["detail"]["error"] == "DateBeyondLimitError"
