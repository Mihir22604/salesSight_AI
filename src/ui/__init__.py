"""
SalesSight AI - Streamlit Web UI Package
=======================================
Modular interactive dashboard components adhering to Section 16 of the Master Blueprint.
"""

from .client import ForecastAPIClient
from .components import UIComponents

__all__ = ["ForecastAPIClient", "UIComponents"]
