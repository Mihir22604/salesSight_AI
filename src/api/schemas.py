"""
Pydantic Request & Response Validation Schemas for SalesSight AI REST API
========================================================================
Implements strict validation models adhering to Sections 13, 15, and 16 of the Master Blueprint.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TargetType(str, Enum):
    SALES = "Sales"
    QUANTITY = "Quantity"
    BOTH = "Both"


class ScopeType(str, Enum):
    ALL = "All"
    CATEGORY = "Category"
    SKU = "SKU"


class ModelSelectionMode(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class StartDateMode(str, Enum):
    DEFAULT = "DEFAULT"
    BACKTEST = "BACKTEST"
    BRIDGE = "BRIDGE"
    BEYOND_LIMIT = "BEYOND_LIMIT"
    DEMONSTRATION = "DEMONSTRATION"


class DailyForecastPoint(BaseModel):
    date: str = Field(..., description="Calendar forecast date (YYYY-MM-DD)")
    predicted_value: float = Field(..., description="Calibrated point prediction")
    lower_bound_80: float = Field(..., description="Calibrated 80% prediction lower bound (non-negative)")
    upper_bound_80: float = Field(..., description="Calibrated 80% prediction upper bound")
    day_of_week: str = Field(..., description="Day name (e.g., Monday)")
    step_horizon: int = Field(..., description="Forecast horizon step h in [1, H]")


class ForecastMetricsSummary(BaseModel):
    total_forecast: float = Field(..., description="Sum of predictions across the horizon")
    daily_average: float = Field(..., description="Average daily forecasted magnitude")
    peak_day: str = Field(..., description="Date of highest projected demand")
    peak_value: float = Field(..., description="Magnitude of peak demand")
    lowest_day: str = Field(..., description="Date of lowest projected demand")
    lowest_value: float = Field(..., description="Magnitude of lowest demand")
    growth_rate_pct: Optional[float] = Field(None, description="Projected period-over-period growth %")


class ForecastRequest(BaseModel):
    target: TargetType = Field(default=TargetType.SALES, description="Forecast target (Sales, Quantity, Both)")
    scope: ScopeType = Field(default=ScopeType.ALL, description="Forecast aggregation scope")
    scope_identifier: Optional[str] = Field(default=None, description="Category name or ProductID when scope is Category or SKU")
    horizon_days: int = Field(default=30, description="Prediction horizon in days (30, 60, 90)")
    start_date: Optional[str] = Field(default="2014-09-30", description="Forecast starting date (YYYY-MM-DD)")
    model_selection_mode: ModelSelectionMode = Field(default=ModelSelectionMode.AUTOMATIC, description="automatic or manual")
    manual_model_name: Optional[str] = Field(default=None, description="Model key when mode is manual")
    enable_demo_mode: bool = Field(default=False, description="Enable Demonstration Mode for 2026/current dates")

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in [30, 60, 90]:
            raise ValueError("Horizon must be exactly 30, 60, or 90 days.")
        return v


class SingleTargetForecastResponse(BaseModel):
    target: str
    model_used: str
    model_version: str
    metrics_summary: ForecastMetricsSummary
    forecasts: List[DailyForecastPoint]


class ForecastResponse(BaseModel):
    status: str = "success"
    scope: str
    scope_identifier: Optional[str] = None
    horizon_days: int
    start_date: str
    start_date_mode: str
    banner_message: str
    targets: Dict[str, SingleTargetForecastResponse]


class HealthResponse(BaseModel):
    status: str
    service_name: str
    version: str
    uptime_seconds: float
    production_models_loaded: Dict[str, Any]
    artifacts_verified: bool
    last_historical_date: str
    timestamp: str


class Top5ProductDemand(BaseModel):
    rank: int
    product_id: int
    product_name: str
    category_name: str
    projected_units_30d: float
    is_baseline_fallback: bool


class DemandMomentumItem(BaseModel):
    product_id: int
    product_name: str
    category_name: str
    projected_units_30d: float
    reference_units_30d: float
    delta_pct: float
    delta_abs: float
    momentum_class: str = Field(..., description="RISING, FALLING, or STABLE")


class PlanningAttentionItem(BaseModel):
    priority_level: str = Field(..., description="High Priority, Review, or Overstock Watch")
    product_id: int
    product_name: str
    category_name: str
    reason: str
    recommended_action: str


class CategoryShareRanking(BaseModel):
    category_name: str
    total_volume_units: float
    volume_share_pct: float
    total_revenue_usd: float
    revenue_share_pct: float


class IntelligenceResponse(BaseModel):
    status: str = "success"
    evaluation_window_days: int = 30
    top_5_demand_ranking: List[Top5ProductDemand]
    demand_momentum: List[DemandMomentumItem]
    planning_attention_list: List[PlanningAttentionItem]
    category_rankings: List[CategoryShareRanking]
    narrative_insights: List[str]
    mandatory_inventory_disclaimer: str


class ModelsResponse(BaseModel):
    status: str = "success"
    manifest_version: str
    data_provenance: Dict[str, Any]
    production_selected_models: Dict[str, Any]
    residual_quantiles_calibration: Dict[str, Any]
    total_artifacts_audited: int
