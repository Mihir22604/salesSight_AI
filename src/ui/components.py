"""
SalesSight AI - Streamlit Visual Components & Styling
=====================================================
Implements Section 16 & 17 of the Master Blueprint:
- Custom Dark Theme CSS styling (Slate #0E1117, Electric Blue #2563EB, Emerald #10B981).
- Interactive Plotly timeline charts with calibrated 80% shaded prediction bounds.
- Executive KPI summary scorecards.
- Demand Intelligence Digest (Top-5, Momentum, Attention List, Disclaimer).
- RFC-4180 compliant CSV forecast export generator (Section 17.1).
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import inspect
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Forward and backward compatible width parameter to support Streamlit 1.66+ without deprecation warnings
_WIDTH_KWARG = (
    {"width": "stretch"}
    if hasattr(st, "button") and "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)


class UIComponents:
    """Reusable visual presentation components for SalesSight AI Streamlit Dashboard."""

    @staticmethod
    def inject_custom_css() -> None:
        """Injects production-grade Dark Theme typography and component styling."""
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            /* Main container padding */
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 95rem;
            }
            
            /* Executive KPI Card Container */
            .kpi-card {
                background: linear-gradient(135deg, #161B22 0%, #0D1117 100%);
                border: 1px solid #30363D;
                border-radius: 10px;
                padding: 1.1rem 1.25rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            .kpi-card:hover {
                border-color: #2563EB;
                transform: translateY(-2px);
            }
            .kpi-title {
                font-size: 0.82rem;
                font-weight: 500;
                color: #8B949E;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.35rem;
            }
            .kpi-value {
                font-size: 1.75rem;
                font-weight: 700;
                color: #FFFFFF;
                line-height: 1.2;
            }
            .kpi-sub {
                font-size: 0.8rem;
                color: #10B981;
                font-weight: 500;
                margin-top: 0.35rem;
            }
            .kpi-sub-amber {
                font-size: 0.8rem;
                color: #F59E0B;
                font-weight: 500;
                margin-top: 0.35rem;
            }
            
            /* Alert Banners */
            .banner-notice {
                background: rgba(37, 99, 235, 0.12);
                border-left: 4px solid #2563EB;
                border-radius: 6px;
                padding: 0.85rem 1.2rem;
                margin-bottom: 1.25rem;
                font-size: 0.92rem;
                color: #E6EDF3;
            }
            .banner-demo {
                background: rgba(245, 158, 11, 0.12);
                border-left: 4px solid #F59E0B;
                border-radius: 6px;
                padding: 0.85rem 1.2rem;
                margin-bottom: 1.25rem;
                font-size: 0.92rem;
                color: #FEEBC8;
            }
            .disclaimer-box {
                background: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 1rem;
                font-size: 0.82rem;
                color: #8B949E;
                font-style: italic;
                margin-top: 1rem;
            }
            
            /* Attention list badge styles */
            .badge-high {
                background: #DC2626;
                color: white;
                padding: 0.2rem 0.6rem;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.75rem;
            }
            .badge-review {
                background: #2563EB;
                color: white;
                padding: 0.2rem 0.6rem;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.75rem;
            }
            .badge-overstock {
                background: #F59E0B;
                color: black;
                padding: 0.2rem 0.6rem;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.75rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_header(banner_text: str, is_demo_mode: bool = False) -> None:
        """Renders top navigation header and historical limit alert banner."""
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700; color: #FFFFFF;">
                    SalesSight <span style="color: #2563EB;">AI</span>
                </h1>
                <span style="background: #21262D; color: #58A6FF; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">
                    Commercial Forecasting Engine v1.0.0
                </span>
            </div>
            <p style="color: #8B949E; margin-top: 0; font-size: 0.95rem; margin-bottom: 1rem;">
                Precision Demand Intelligence & Calibrated 80% Uncertainty Range Forecasting
            </p>
            """,
            unsafe_allow_html=True,
        )

        banner_class = "banner-demo" if is_demo_mode else "banner-notice"
        st.markdown(f'<div class="{banner_class}">{banner_text}</div>', unsafe_allow_html=True)

    @staticmethod
    def render_kpi_cards(
        sales_summary: Optional[Dict[str, Any]],
        qty_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Renders Row 1: Executive KPI metric cards."""
        if qty_summary is None:
            # Clean 4-card layout focused purely on Sales Revenue
            cols = st.columns(4)

            # 1. Total Projected Revenue
            with cols[0]:
                if sales_summary:
                    rev_val = f"${sales_summary['total_forecast']:,.0f}"
                    growth = sales_summary.get("growth_rate_pct", 0.0)
                    sub_text = f"+{growth:.1f}% vs baseline" if growth >= 0 else f"{growth:.1f}% vs baseline"
                else:
                    rev_val = "N/A"
                    sub_text = "Metric omitted"
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Projected Sales Revenue</div>
                        <div class="kpi-value">{rev_val}</div>
                        <div class="kpi-sub">{sub_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 2. Daily Average Run Rate
            with cols[1]:
                if sales_summary:
                    avg_text = f"${sales_summary['daily_average']:,.0f} / day"
                else:
                    avg_text = "N/A"
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Daily Run Rate</div>
                        <div class="kpi-value" style="font-size: 1.45rem;">{avg_text}</div>
                        <div class="kpi-sub" style="color: #8B949E;">Expected daily revenue</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3. Projected Peak Day
            with cols[2]:
                peak_date = sales_summary["peak_day"] if sales_summary else "N/A"
                peak_mag = f"${sales_summary['peak_value']:,.0f}" if sales_summary else "N/A"
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Projected Peak Sales Day</div>
                        <div class="kpi-value" style="font-size: 1.35rem; color: #58A6FF;">{peak_date}</div>
                        <div class="kpi-sub" style="color: #58A6FF;">Peak volume: {peak_mag}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 4. Trough / Lowest Day
            with cols[3]:
                low_date = sales_summary["lowest_day"] if sales_summary else "N/A"
                low_mag = f"${sales_summary['lowest_value']:,.0f}" if sales_summary else "N/A"
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Trough / Lowest Sales Day</div>
                        <div class="kpi-value" style="font-size: 1.35rem; color: #F59E0B;">{low_date}</div>
                        <div class="kpi-sub-amber">Trough: {low_mag}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            return

        # 5-card layout if dual summary provided
        cols = st.columns(5)
        # 1. Total Projected Revenue
        with cols[0]:
            if sales_summary:
                rev_val = f"${sales_summary['total_forecast']:,.0f}"
                growth = sales_summary.get("growth_rate_pct", 0.0)
                sub_text = f"+{growth:.1f}% vs baseline" if growth >= 0 else f"{growth:.1f}% vs baseline"
            else:
                rev_val = "N/A"
                sub_text = "Metric omitted"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Projected Revenue</div>
                    <div class="kpi-value">{rev_val}</div>
                    <div class="kpi-sub">{sub_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Total Projected Demand Units
        with cols[1]:
            if qty_summary:
                qty_val = f"{qty_summary['total_forecast']:,.0f} u"
                growth = qty_summary.get("growth_rate_pct", 0.0)
                sub_text = f"+{growth:.1f}% vs baseline" if growth >= 0 else f"{growth:.1f}% vs baseline"
            else:
                qty_val = "N/A"
                sub_text = "Metric omitted"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Projected Demand</div>
                    <div class="kpi-value">{qty_val}</div>
                    <div class="kpi-sub">{sub_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Daily Average Velocity
        with cols[2]:
            if sales_summary:
                avg_text = f"${sales_summary['daily_average']:,.0f} / day"
                sub_units = f"~{qty_summary['daily_average']:,.0f} units/day" if qty_summary else "Daily mean"
            else:
                avg_text = f"{qty_summary['daily_average']:,.0f} units/day"
                sub_units = "Daily mean"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Daily Run Rate</div>
                    <div class="kpi-value" style="font-size: 1.45rem;">{avg_text}</div>
                    <div class="kpi-sub" style="color: #8B949E;">{sub_units}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 4. Projected Peak Day
        with cols[3]:
            primary_sum = sales_summary or qty_summary
            peak_date = primary_sum["peak_day"] if primary_sum else "N/A"
            peak_mag = f"${primary_sum['peak_value']:,.0f}" if sales_summary else f"{primary_sum['peak_value']:,.0f} units"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Projected Peak Day</div>
                    <div class="kpi-value" style="font-size: 1.35rem; color: #58A6FF;">{peak_date}</div>
                    <div class="kpi-sub" style="color: #58A6FF;">Peak volume: {peak_mag}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 5. Lowest Demand Day
        with cols[4]:
            primary_sum = sales_summary or qty_summary
            low_date = primary_sum["lowest_day"] if primary_sum else "N/A"
            low_mag = f"${primary_sum['lowest_value']:,.0f}" if sales_summary else f"{primary_sum['lowest_value']:,.0f} units"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Trough / Lowest Day</div>
                    <div class="kpi-value" style="font-size: 1.35rem; color: #F59E0B;">{low_date}</div>
                    <div class="kpi-sub-amber">Trough: {low_mag}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    @staticmethod
    def build_forecast_plot(
        forecast_pts: List[Dict[str, Any]],
        target_name: str,
        color_hex: str = "#2563EB",
    ) -> go.Figure:
        """
        Constructs interactive Plotly timeline chart with shaded 80% confidence ribbons
        and high-contrast tooltips adhering to Blueprint §16.1.
        """
        df = pd.DataFrame(forecast_pts)
        df["date"] = pd.to_datetime(df["date"])

        fig = go.Figure()

        # Upper bound (invisible line, basis for fill)
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["upper_bound_80"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="80% Upper Bound",
            )
        )

        # Lower bound with translucent fill to upper bound
        fill_color = "rgba(37, 99, 235, 0.18)" if target_name == "Sales" else "rgba(16, 185, 129, 0.18)"
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["lower_bound_80"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                name="80% Calibrated Prediction Range",
                hovertemplate="<b>80% Range</b>: %{customdata[0]:$,.1f} – %{customdata[1]:$,.1f}<extra></extra>"
                if target_name == "Sales"
                else "<b>80% Range</b>: %{customdata[0]:,.0f} – %{customdata[1]:,.0f} units<extra></extra>",
                customdata=df[["lower_bound_80", "upper_bound_80"]].values,
            )
        )

        # Point Forecast Line
        line_color = "#3B82F6" if target_name == "Sales" else "#10B981"
        unit_prefix = "$" if target_name == "Sales" else ""
        unit_suffix = "" if target_name == "Sales" else " units"

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["predicted_value"],
                mode="lines+markers",
                line=dict(color=line_color, width=3),
                marker=dict(size=5, color=line_color),
                name=f"Point Forecast ({target_name})",
                hovertemplate=f"<b>Date</b>: %{{x|%Y-%m-%d}} (%{{text}})<br><b>Forecast</b>: {unit_prefix}%{{y:,.2f}}{unit_suffix}<extra></extra>",
                text=df["day_of_week"],
            )
        )

        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#C9D1D9", size=11),
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="#21262D",
                tickfont=dict(color="#8B949E"),
                showline=True,
                linecolor="#30363D",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#21262D",
                tickfont=dict(color="#8B949E"),
                tickprefix=unit_prefix,
                ticksuffix=unit_suffix,
                showline=True,
                linecolor="#30363D",
            ),
            hovermode="x unified",
            height=420,
        )

        return fig

    @staticmethod
    def render_demand_intelligence(intel: Dict[str, Any]) -> None:
        """Renders Row 3A: Demand Intelligence Digest."""
        st.markdown("### Commercial Demand Intelligence")

        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            st.markdown("##### Top-5 Product Demand Ranking (Next 30 Days)")
            top5 = intel.get("top_5_demand_ranking", [])
            if top5:
                top5_rows = []
                for p in top5:
                    top5_rows.append({
                        "Rank": f"#{p['rank']}",
                        "Product Name": p["product_name"],
                        "Category": p["category_name"],
                        "Projected Units (30d)": f"{p['projected_units_30d']:,.0f} u",
                    })
                st.dataframe(pd.DataFrame(top5_rows), hide_index=True, **_WIDTH_KWARG)

            st.markdown("##### Strategic Planning Attention List")
            att = intel.get("planning_attention_list", [])
            for item in att[:4]:
                p_level = item["priority_level"]
                badge_class = (
                    "badge-high"
                    if p_level == "High Priority"
                    else ("badge-overstock" if p_level == "Overstock Watch" else "badge-review")
                )
                st.markdown(
                    f"""
                    <div style="background: #161B22; border: 1px solid #30363D; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                            <strong style="color: #FFFFFF; font-size: 0.88rem;">{item['product_name']}</strong>
                            <span class="{badge_class}">{p_level}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #8B949E;">{item['reason']}</div>
                        <div style="font-size: 0.8rem; color: #58A6FF; margin-top: 0.2rem;">Directive: {item['recommended_action']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_right:
            st.markdown("##### Category Share Distribution")
            cat_rank = intel.get("category_rankings", [])
            if cat_rank:
                cat_df = pd.DataFrame(cat_rank)
                fig_cat = go.Figure(
                    go.Bar(
                        x=cat_df["total_revenue_usd"],
                        y=cat_df["category_name"],
                        orientation="h",
                        marker=dict(color="#2563EB"),
                        hovertemplate="<b>%{y}</b>: $%{x:,.2f}<extra></extra>",
                    )
                )
                fig_cat.update_layout(
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161B22",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=360,
                    xaxis=dict(showgrid=True, gridcolor="#21262D", tickfont=dict(color="#8B949E"), tickprefix="$"),
                    yaxis=dict(autorange="reversed", tickfont=dict(color="#C9D1D9")),
                )
                st.plotly_chart(fig_cat, **_WIDTH_KWARG)

        # Mandatory Inventory Disclaimer (§15.3)
        disclaimer = intel.get(
            "mandatory_inventory_disclaimer",
            "Planning recommendations reflect relative demand velocity derived strictly from historical patterns.",
        )
        st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Operational Notice:</strong> {disclaimer}</div>', unsafe_allow_html=True)

    @staticmethod
    def generate_rfc4180_csv(
        forecast_response: Dict[str, Any],
        scope: str,
        scope_id: Optional[str] = None,
    ) -> str:
        """
        Generates RFC-4180 compliant CSV export with standardized metadata headers (Blueprint §17.1).
        """
        targets = forecast_response.get("targets", {})
        h_days = forecast_response.get("horizon_days", 30)
        s_date = forecast_response.get("start_date", "2014-09-30")
        mode = forecast_response.get("start_date_mode", "DEFAULT")

        sales_pts = targets.get("Sales", {}).get("forecasts", [])
        qty_pts = targets.get("Quantity", {}).get("forecasts", [])

        # Align rows by step
        n = max(len(sales_pts), len(qty_pts))
        rows = []
        for i in range(n):
            s_pt = sales_pts[i] if i < len(sales_pts) else {}
            q_pt = qty_pts[i] if i < len(qty_pts) else {}

            date_str = s_pt.get("date", q_pt.get("date", ""))
            rev = s_pt.get("predicted_value", 0.0)
            rev_l = s_pt.get("lower_bound_80", 0.0)
            rev_u = s_pt.get("upper_bound_80", 0.0)

            qty = q_pt.get("predicted_value", 0.0)
            qty_l = q_pt.get("lower_bound_80", 0.0)
            qty_u = q_pt.get("upper_bound_80", 0.0)

            rows.append({
                "Date": date_str,
                "Predicted_Revenue_USD": rev,
                "Revenue_Lower_80": rev_l,
                "Revenue_Upper_80": rev_u,
                "Predicted_Quantity_Units": qty,
                "Quantity_Lower_80": qty_l,
                "Quantity_Upper_80": qty_u,
                "Scope": scope,
                "Scope_Identifier": scope_id or "All",
                "Model_Used": "Ridge Regression",
                "Date_Mode": mode,
            })

        df = pd.DataFrame(rows)
        csv_buffer = io.StringIO()

        # Metadata Header (Section 17.1)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        csv_buffer.write("# ==============================================================================\n")
        csv_buffer.write("# SalesSight AI — Daily Sales Forecast Export\n")
        csv_buffer.write(f"# Export Timestamp: {now_utc}\n")
        csv_buffer.write(f"# Forecast Scope: {scope} ({scope_id or 'All'})\n")
        csv_buffer.write(f"# Horizon: {h_days} Days | Start Date: {s_date} ({mode})\n")
        csv_buffer.write("# Selected Model: Ridge Regression | Historical Baseline WAPE: 138.56% | Model WAPE: 92.78%\n")
        csv_buffer.write("# Historical Data Limitation: Underlying dataset ends 2014-09-29.\n")
        csv_buffer.write("# ==============================================================================\n")

        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
