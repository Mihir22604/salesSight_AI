"""
SalesSight AI - Smart Sales Forecasting & Commercial Demand Intelligence Dashboard
==================================================================================
Main Streamlit application entry point implementing Section 16 of the Master Blueprint.
Full interactive controls, Plotly charts with 80% confidence envelopes, and demand intelligence.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import inspect
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.ui.client import ForecastAPIClient
from src.ui.components import UIComponents

# Forward and backward compatible width parameter to support Streamlit 1.66+ without deprecation warnings
_WIDTH_KWARG = (
    {"width": "stretch"}
    if hasattr(st, "button") and "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="SalesSight AI — Demand Forecasting & Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Dark Theme CSS
UIComponents.inject_custom_css()

# Initialize API / Inference Client
client = ForecastAPIClient()

# ==============================================================================
# SIDEBAR CONTROLS (Blueprint Section 16.2)
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Forecast Configuration")

    # 1. Forecast Start Date & Demonstration Mode
    enable_demo = st.toggle("Enable Demonstration Mode (2026)", value=True, help="Maps current 2026 calendar dates onto historical seasonality.")
    
    if enable_demo:
        default_date = date(2026, 10, 24)
        min_date = date(2025, 1, 1)
        max_date = date(2028, 12, 31)
    else:
        default_date = date(2014, 9, 30)
        min_date = date(2011, 1, 21)
        max_date = date(2014, 12, 28)

    start_date_val = st.date_input(
        "Forecast Start Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help="Latest historical sales data ends on 2014-09-29.",
    )

    # 2. Prediction Horizon (30, 60, 90 Days)
    horizon_choice = st.radio(
        "Prediction Horizon",
        options=[30, 60, 90],
        index=0,
        horizontal=True,
        format_func=lambda x: f"{x} Days",
    )

    # Fixed settings: Macro Store Sales Revenue Forecasting with Champion Model
    selected_scope = "All"
    scope_id = None
    selected_target = "Sales"
    model_mode = "automatic"
    manual_model = None

    generate_btn = st.button("🚀 Generate Forecast", type="primary", **_WIDTH_KWARG)

    # Detect if user modified parameters from the currently active forecast
    current_params = {
        "start_date": start_date_val.strftime("%Y-%m-%d"),
        "horizon_days": int(horizon_choice),
        "enable_demo_mode": enable_demo,
    }
    last_params = st.session_state.get("last_executed_params")
    if last_params is not None and last_params != current_params:
        st.info("💡 Settings modified. Click **🚀 Generate Forecast** to calculate predictions.")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #8B949E; line-height: 1.4;">
            <strong>System Telemetry:</strong><br>
            • Target: Sales Revenue ($)<br>
            • API Endpoint: <code>/forecast</code><br>
            • Model: Ridge Regression (Champion)<br>
            • Latency: &lt; 2ms / forecast<br>
            • Calibration: 80% Empirical Quantiles
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# MAIN VIEWPORT DISPLAY
# ==============================================================================

# Build Request Payload
payload = {
    "target": "Sales",
    "scope": selected_scope,
    "scope_identifier": scope_id,
    "horizon_days": int(horizon_choice),
    "start_date": start_date_val.strftime("%Y-%m-%d"),
    "model_selection_mode": "automatic",
    "manual_model_name": None,
    "enable_demo_mode": enable_demo,
}

# Execution Policy: Execute forecasting ONLY on initial app load OR when Generate button is clicked
is_first_load = "forecast_data" not in st.session_state
should_execute = bool(generate_btn or is_first_load)

if should_execute:
    with st.spinner("Executing calibrated inference pipeline..."):
        st.session_state["forecast_data"] = client.get_forecast(payload)
        st.session_state["intel_data"] = client.get_intelligence()
        st.session_state["models_meta"] = client.get_models()
        st.session_state["last_executed_params"] = current_params

forecast_data = st.session_state["forecast_data"]
intel_data = st.session_state["intel_data"]
models_meta = st.session_state["models_meta"]

# Check for Error Responses
if "error" in forecast_data.get("detail", {}) if isinstance(forecast_data.get("detail"), dict) else False:
    err_detail = forecast_data["detail"]
    st.error(f"⚠️ {err_detail.get('error', 'Forecasting Error')}: {err_detail.get('message')}")
    st.stop()

# Header and Dynamic Mode Notice Banner
banner_text = forecast_data.get("banner_message", "Production forecast pipeline active.")
active_demo = forecast_data.get("enable_demo_mode", enable_demo)
active_horizon = forecast_data.get("horizon_days", horizon_choice)
UIComponents.render_header(banner_text, is_demo_mode=active_demo)

# ROW 1: Executive KPI Metric Cards (Sales Revenue Focus)
targets = forecast_data.get("targets", {})
sales_summary = targets.get("Sales", {}).get("metrics_summary")

UIComponents.render_kpi_cards(sales_summary, qty_summary=None)

st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

# ROW 2: Interactive Plotly Timeline Chart (Sales Revenue Forecast)
st.markdown("### Interactive Forecast Trajectory & Calibrated 80% Uncertainty Envelopes")

if "Sales" in targets:
    fig_sales = UIComponents.build_forecast_plot(targets["Sales"]["forecasts"], target_name="Sales")
    st.plotly_chart(fig_sales, **_WIDTH_KWARG)

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# ROW 3: Two-Column Layout (Demand Intelligence vs Daily Prediction Table)
col_digest, col_table = st.columns([1.1, 0.9])

with col_digest:
    # Row 3A: Demand Intelligence Digest
    UIComponents.render_demand_intelligence(intel_data)

with col_table:
    # Row 3B: Daily Prediction Table & RFC-4180 CSV Export
    st.markdown("### Daily Prediction Forecast Table")

    forecast_list = targets.get("Sales", {}).get("forecasts", [])
    table_df = pd.DataFrame(forecast_list)

    display_cols = ["date", "day_of_week", "predicted_value", "lower_bound_80", "upper_bound_80"]
    rename_cols = {
        "date": "Date",
        "day_of_week": "Day",
        "predicted_value": "Forecast (Sales $)",
        "lower_bound_80": "80% Lower",
        "upper_bound_80": "80% Upper",
    }
    st.dataframe(
        table_df[display_cols].rename(columns=rename_cols),
        height=380,
        hide_index=True,
        **_WIDTH_KWARG,
    )

    # RFC-4180 CSV Download Button (Blueprint Section 17.1)
    csv_payload = UIComponents.generate_rfc4180_csv(forecast_data, scope="All", scope_id=None)
    now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="📥 Download RFC-4180 Forecast CSV",
        data=csv_payload,
        file_name=f"sales_sight_forecast_{active_horizon}d_{now_ts}.csv",
        mime="text/csv",
        **_WIDTH_KWARG,
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ROW 4: Model Governance & Accuracy Audit Panel
with st.expander("🛡️ Production Model Governance, Validation Scorecard & Integrity Audit"):
    sel_models = models_meta.get("production_selected_models", {})
    st.markdown("##### Sales Revenue Champion Model")
    s_meta = sel_models.get("Sales", {})
    st.markdown(f"- **Algorithm:** `{s_meta.get('algorithm', 'Ridge')}` ({s_meta.get('display_name', 'Ridge Regression')})")
    st.markdown(f"- **Out-of-Sample Holdout WAPE:** **{s_meta.get('validation_metrics', {}).get('out_of_sample_wape', 92.78)}%**")
    st.markdown(f"- **Holdout RMSE:** ${s_meta.get('validation_metrics', {}).get('out_of_sample_rmse', 3902.15):,.2f}")
    st.markdown(f"- **Inference Latency:** {s_meta.get('validation_metrics', {}).get('inference_latency_ms', 1.76)} ms")
    st.markdown(f"- **Selection Category:** `{s_meta.get('selection_decision', 'TIE_BROKEN_BY_SIMPLICITY')}`")
    st.caption(f"Rationale: {s_meta.get('selection_rationale')}")
