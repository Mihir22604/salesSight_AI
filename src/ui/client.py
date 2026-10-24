"""
SalesSight AI - Streamlit API Client
====================================
Communicates with the FastAPI backend at http://127.0.0.1:8000.
Falls back seamlessly to direct in-process inference engines if the remote API daemon is offline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

from src.api.forecast_service import ProductionForecastEngine
from src.api.intelligence_service import DemandIntelligenceEngine
from src.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    IntelligenceResponse,
    ModelsResponse,
    ScopeType,
    TargetType,
)

logger = logging.getLogger("SalesSight.UIClient")


class ForecastAPIClient:
    """
    Robust API client querying FastAPI endpoints with automatic in-process fallback.
    Supports remote Render backend via API_URL or BACKEND_URL environment variable.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url or os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"
        ).rstrip("/")
        self._local_forecast_engine: Optional[ProductionForecastEngine] = None
        self._local_intel_engine: Optional[DemandIntelligenceEngine] = None

    def _get_local_forecast_engine(self) -> ProductionForecastEngine:
        if self._local_forecast_engine is None:
            self._local_forecast_engine = ProductionForecastEngine()
        return self._local_forecast_engine

    def _get_local_intel_engine(self) -> DemandIntelligenceEngine:
        if self._local_intel_engine is None:
            self._local_intel_engine = DemandIntelligenceEngine()
        return self._local_intel_engine

    def check_health(self) -> Dict[str, Any]:
        """Query GET /health endpoint or in-process health status."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                data["backend_mode"] = "FASTAPI_REST_DAEMON"
                return data
        except Exception:
            pass

        # In-process fallback
        engine = self._get_local_forecast_engine()
        return {
            "status": "healthy" if engine.is_healthy() else "degraded",
            "service_name": "SalesSight AI In-Process ML Engine",
            "version": "1.0.0",
            "uptime_seconds": 120.0,
            "production_models_loaded": {
                k: {"model_version": v.model_version, "features": len(v.feature_columns)}
                for k, v in engine.models.items()
            },
            "artifacts_verified": engine.is_healthy(),
            "last_historical_date": "2014-09-29",
            "backend_mode": "DIRECT_IN_PROCESS_FALLBACK",
        }

    def get_forecast(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Query POST /forecast endpoint or execute in-process prediction."""
        try:
            resp = requests.post(f"{self.base_url}/forecast", json=request_payload, timeout=3.0)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 400:
                return resp.json()
        except Exception:
            pass

        # In-process fallback execution
        from fastapi import HTTPException
        try:
            engine = self._get_local_forecast_engine()
            req_obj = ForecastRequest(**request_payload)
            resp_obj = engine.generate_forecast(req_obj)
            return resp_obj.model_dump()
        except HTTPException as he:
            return {"detail": he.detail, "status_code": he.status_code}

    def get_models(self) -> Dict[str, Any]:
        """Query GET /models endpoint or read local manifest."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=1.5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        import json
        from pathlib import Path
        manifest_path = Path("models/artifacts/model_manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "unavailable", "message": "Manifest not found"}

    def get_intelligence(self) -> Dict[str, Any]:
        """Query GET /intelligence endpoint or compute in-process."""
        try:
            resp = requests.get(f"{self.base_url}/intelligence", timeout=2.5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        intel_engine = self._get_local_intel_engine()
        resp_obj = intel_engine.generate_intelligence()
        return resp_obj.model_dump()
