"""
SalesSight AI - Production FastAPI REST Server
==============================================
Standalone headless REST API built with FastAPI in src/api/main.py.
Implements endpoints:
- GET /health: Liveness and readiness probe, returning HTTP 200 OK and model status.
- POST /forecast: Multi-step demand predictions with 80% confidence bands across 5 date modes.
- GET /models: Complete scorecards, parameters, out-of-sample metrics, and selection rationale.
- GET /intelligence: Downstream Demand Intelligence (Top-5, momentum, attention list, category shares).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .forecast_service import ProductionForecastEngine
from .intelligence_service import DemandIntelligenceEngine
from .schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    IntelligenceResponse,
    ModelsResponse,
)

logger = logging.getLogger("SalesSight.API")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SERVER_START_TIME = time.time()
forecast_engine: Optional[ProductionForecastEngine] = None
intelligence_engine: Optional[DemandIntelligenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler preloading model packages and catalog metadata."""
    global forecast_engine, intelligence_engine
    logger.info("Starting SalesSight AI REST Server...")
    forecast_engine = ProductionForecastEngine()
    intelligence_engine = DemandIntelligenceEngine()
    logger.info("Preloaded production forecasting & intelligence engines.")
    yield
    logger.info("Shutting down SalesSight AI REST Server.")


app = FastAPI(
    title="SalesSight AI - Demand Forecasting & Intelligence REST API",
    description=(
        "Production ML inference API delivering daily sales revenue and unit demand forecasts "
        "with calibrated 80% empirical prediction intervals and commercial demand intelligence."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for dashboard and UI consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check() -> HealthResponse:
    """
    Standardized GET /health endpoint returning HTTP 200 OK, uptime, and model status.
    """
    global forecast_engine
    engine = forecast_engine or ProductionForecastEngine()

    models_loaded: Dict[str, Any] = {}
    for tgt, pkg in engine.models.items():
        models_loaded[tgt] = {
            "model_version": pkg.model_version,
            "target_name": pkg.target_name,
            "hyperparameters": pkg.hyperparameters,
            "features_count": len(pkg.feature_columns),
            "packaged_at": pkg.packaged_at,
        }

    uptime = round(time.time() - SERVER_START_TIME, 2)
    return HealthResponse(
        status="healthy" if engine.is_healthy() else "degraded",
        service_name="SalesSight AI ML Inference Service",
        version="1.0.0",
        uptime_seconds=uptime,
        production_models_loaded=models_loaded,
        artifacts_verified=engine.is_healthy(),
        last_historical_date="2014-09-29",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Inference"])
def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Generates multi-step point predictions and calibrated 80% prediction intervals
    for Sales Revenue ($), Order Quantity (units), or Both, observing the 5 start-date modes.
    """
    global forecast_engine
    engine = forecast_engine or ProductionForecastEngine()
    return engine.generate_forecast(request)


@app.get("/models", response_model=ModelsResponse, tags=["Governance"])
def get_model_scorecards() -> ModelsResponse:
    """
    Returns the comprehensive production model manifest containing validation scorecards,
    hyperparameters, SHA-256 checksums, and selection rationale.
    """
    manifest_path = Path("models/artifacts/model_manifest.json")
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model manifest not found at models/artifacts/model_manifest.json.",
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ModelsResponse(
        status="success",
        manifest_version=data.get("manifest_version", "2.0.0"),
        data_provenance=data.get("data_provenance", {}),
        production_selected_models=data.get("production_selected_models", {}),
        residual_quantiles_calibration=data.get("residual_quantiles_calibration", {}),
        total_artifacts_audited=len(data.get("artifact_inventory", [])),
    )


@app.get("/intelligence", response_model=IntelligenceResponse, tags=["Commercial Intelligence"])
def get_demand_intelligence() -> IntelligenceResponse:
    """
    Computes commercial demand intelligence:
    - Top-5 Product Demand Ranking (strictly Days 1-30)
    - Demand Momentum Classification (Rising, Falling, Stable)
    - Planning Attention List (High Priority, Review, Overstock Watch)
    - Category Share Ranking (Volume vs. Revenue)
    - Deterministic Plain-Language Narrative Insights
    """
    global intelligence_engine
    engine = intelligence_engine or DemandIntelligenceEngine()
    return engine.generate_intelligence()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Defensive error boundary suppressing raw stack traces."""
    logger.error("Unhandled exception processing request %s: %s", request.url, str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": "Internal inference engine error. Please review input parameters.",
        },
    )
