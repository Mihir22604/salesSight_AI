"""
Unit tests for Step 1 (Data Understanding) and Step 2 (Data Cleaning & Aggregation).
Verifies schema contracts, missing values, leakage elimination, and daily continuity.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data_pipeline import (
    DataLoader,
    DataValidator,
    DataUnderstandingProfiler,
    DataCleaner,
    SalesAggregator,
)


@pytest.fixture(scope="module")
def raw_dataframe():
    loader = DataLoader("data/raw/New_Query_2026_10_22_10_01_37.csv")
    return loader.load_raw_data()


def test_step1_loader_and_schema(raw_dataframe):
    assert len(raw_dataframe) == 2172, "Raw dataset must contain exactly 2,172 rows"
    assert len(raw_dataframe.columns) == 33, "Raw dataset must contain exactly 33 columns"
    validator = DataValidator()
    is_valid, missing = validator.validate_raw_schema(raw_dataframe)
    assert is_valid is True, f"Schema validation failed. Missing: {missing}"
    assert len(missing) == 0


def test_step1_understanding_profiler(raw_dataframe):
    profiler = DataUnderstandingProfiler(raw_dataframe)
    audit = profiler.profile_dataset()
    assert audit["dimensions"]["rows"] == 2172
    assert audit["dimensions"]["columns"] == 33
    assert audit["date_profile"]["calendar_span_days"] == 1348
    assert audit["date_profile"]["active_dates_count"] == 533
    assert audit["date_profile"]["inactive_dates_count"] == 815
    assert audit["mathematical_identity"]["holds_strictly"] is True
    assert "Sales" in audit["target_analysis"]
    assert "Quantity" in audit["target_analysis"]
    assert audit["entity_counts"]["unique_orders"] == 836
    assert audit["entity_counts"]["unique_products"] == 77
    assert audit["entity_counts"]["unique_categories"] == 8


def test_step2_cleaner_leakage_and_types(raw_dataframe):
    cleaner = DataCleaner(drop_leakage=True)
    cleaned_df, audit = cleaner.clean_dataset(raw_dataframe)
    assert len(cleaned_df) == 2172
    assert "COS" not in cleaned_df.columns, "COS must be purged to prevent target leakage"
    assert "GP" not in cleaned_df.columns, "GP must be purged to prevent target leakage"
    assert "Freight" not in cleaned_df.columns
    assert "Order_Date" in cleaned_df.columns
    assert "OrderID" in cleaned_df.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["Order_Date"])
    assert "UnitPrice" in cleaned_df.columns
    assert (cleaned_df["Quantity"] > 0).all()
    assert (cleaned_df["Sales"] > 0).all()


def test_step2_aggregator_macro_continuity(raw_dataframe):
    cleaner = DataCleaner(drop_leakage=True)
    cleaned_df, _ = cleaner.clean_dataset(raw_dataframe)
    aggregator = SalesAggregator(processed_dir="data/processed")
    macro_series = aggregator.aggregate_macro_series(cleaned_df)

    assert len(macro_series) == 1348, "Macro continuous timeline must contain exactly 1,348 days"
    assert pytest.approx(macro_series["Sales"].sum(), 0.01) == 1783147.70
    assert macro_series["Quantity"].sum() == 52057
    assert (macro_series["Sales"] == 0.0).sum() == 815, "Zero-sales days must equal 815 (60.46%)"


def test_step2_aggregator_category_grid(raw_dataframe):
    cleaner = DataCleaner(drop_leakage=True)
    cleaned_df, _ = cleaner.clean_dataset(raw_dataframe)
    aggregator = SalesAggregator(processed_dir="data/processed")
    cat_series = aggregator.aggregate_category_series(cleaned_df)

    assert len(cat_series) == 10784, "Category grid must equal 1,348 days x 8 categories = 10,784 rows"
    assert set(cat_series["CategoryName"].unique()) == set(cleaned_df["CategoryName"].unique())
