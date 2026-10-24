"""
Consolidated Data Pipeline for SalesSight AI (Step 1 & Step 2).
Combines:
1. DataLoader: CSV ingestion and memory-bounded streaming
2. DataValidator: Schema contract assertions for the 33 raw attributes
3. DataUnderstanding: Statistical profiling, moment auditing, chunk deconstruction, leakage proof
4. DataCleaner: Normalization, UnitPrice/DiscountRate derivation, IQR auditing, target leakage purging
5. SalesAggregator: Continuous daily grid generation across Macro, Meso, and Micro hierarchies
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Generator, Optional
import json
import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger, set_seed, SchemaValidationError, DataIntegrityError

logger = get_logger("DataPipeline")

# Canonical 33 Raw Dataset Columns
EXPECTED_RAW_COLUMNS = [
    "Order", "ProductID", "Quantity", "Sales", "Discount", "COS", "GP",
    "EmployeeID", "CustomerID", "OrderDate", "ShipperID", "Freight",
    "EmployeeName", "Extension", "HireDate", "Office", "ReportsTo", "Title",
    "YearSalary", "Customer", "ContactName", "Fax", "Phone", "PostalCode",
    "CountryCode", "Latitude", "Longitude", "City", "Country", "ProductName",
    "SupplierID", "CategoryName", "CategoryDescription"
]

LEAKAGE_AND_ADMIN_COLUMNS = [
    "COS", "GP", "Freight", "ShipperID", "EmployeeName", "Extension",
    "HireDate", "Office", "ReportsTo", "Title", "YearSalary", "ContactName",
    "Fax", "Phone", "PostalCode", "CountryCode", "Latitude", "Longitude",
    "CategoryDescription"
]


# ==============================================================================
# 1. DATA LOADER
# ==============================================================================
class DataLoader:
    """Handles raw data ingestion from CSV files and batch streaming."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Source dataset not found at {self.data_path}")

    def load_full(self) -> pd.DataFrame:
        """Loads complete raw CSV dataset into a pandas DataFrame."""
        logger.info(f"Loading raw dataset from {self.data_path}")
        df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns successfully.")
        return df

    def load_raw_data(self) -> pd.DataFrame:
        """Alias for load_full()."""
        return self.load_full()

    def load_stream(self, chunksize: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """Streams dataset in bounded memory chunks for scalable processing."""
        logger.info(f"Streaming dataset in chunks of size {chunksize} from {self.data_path}")
        for chunk in pd.read_csv(self.data_path, chunksize=chunksize, low_memory=False):
            yield chunk


# ==============================================================================
# 2. DATA VALIDATOR
# ==============================================================================
class DataValidator:
    """Enforces data contracts and schema integrity on raw data."""

    def __init__(self, expected_columns: Optional[List[str]] = None):
        self.expected_columns = expected_columns or EXPECTED_RAW_COLUMNS

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates that all expected columns exist in the DataFrame."""
        missing_cols = [c for c in self.expected_columns if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in self.expected_columns]

        report = {
            "is_valid": len(missing_cols) == 0,
            "total_columns": len(df.columns),
            "expected_count": len(self.expected_columns),
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
        }

        if not report["is_valid"]:
            logger.error(f"Schema validation FAILED. Missing columns: {missing_cols}")
            raise SchemaValidationError(f"Missing required columns in dataset: {missing_cols}")
        else:
            logger.info("Schema validation PASSED. All 33 expected columns present in raw CSV.")

        return report

    def validate_raw_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates raw schema and returns (is_valid, missing_columns)."""
        res = self.validate_schema(df)
        return res["is_valid"], res["missing_columns"]

    def validate_data_integrity(self, df: pd.DataFrame) -> Any:
        """Returns schema validation report with attribute-access support."""
        res = self.validate_schema(df)
        class ValidationResult:
            def __init__(self, d):
                self.is_valid = d["is_valid"]
                self.missing_columns = d["missing_columns"]
                self.extra_columns = d["extra_columns"]
        return ValidationResult(res)


# ==============================================================================
# 3. DATA UNDERSTANDING
# ==============================================================================
class DataUnderstanding:
    """Profiles raw schema, statistical moments, target distributions, and leakage identities."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def profile_dataset(self) -> Dict[str, Any]:
        """Runs the complete Step 1 Data Understanding audit with full dimensions and entities."""
        logger.info("Executing Step 1: Comprehensive Data Understanding profiling...")
        n_rows, n_cols = self.df.shape
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in self.df.columns if c not in numeric_cols]
        datetime_cols = [c for c in self.df.columns if "date" in c.lower()]

        chunk_fields = {
            "order_chunk": ["Order", "OrderDate"],
            "customer_chunk": ["CustomerID", "Customer", "ContactName", "Phone", "Fax", "City", "Country", "CountryCode", "PostalCode", "Latitude", "Longitude"],
            "fulfillment_and_staff_chunk": ["EmployeeID", "EmployeeName", "Title", "HireDate", "Office", "Extension", "ReportsTo", "YearSalary", "ShipperID", "Freight"]
        }
        non_chunk_fields = {
            "merchandise_item": ["ProductID", "ProductName", "CategoryName", "CategoryDescription", "SupplierID"],
            "line_economics": ["Quantity", "Sales", "Discount"],
            "accounting_outcomes": ["COS", "GP"]
        }

        # Target analysis
        target_analysis = {}
        for target in ["Sales", "Quantity"]:
            if target in self.df.columns:
                t = self.df[target].astype(float)
                target_analysis[target] = {
                    "total_sum": float(t.sum()),
                    "mean": float(t.mean()),
                    "median": float(t.median()),
                    "std": float(t.std()),
                    "min": float(t.min()),
                    "max": float(t.max()),
                    "skewness": float(t.skew()),
                    "kurtosis": float(t.kurtosis()),
                    "zero_count": int((t == 0).sum()),
                    "iqr": float(t.quantile(0.75) - t.quantile(0.25)),
                    "p25": float(t.quantile(0.25)),
                    "p75": float(t.quantile(0.75)),
                    "p90": float(t.quantile(0.90)),
                    "p95": float(t.quantile(0.95)),
                    "p99": float(t.quantile(0.99))
                }

        # Date profile
        date_col = "OrderDate" if "OrderDate" in self.df.columns else "Order_Date"
        date_series = pd.to_datetime(self.df[date_col], errors="coerce")
        min_date = date_series.min()
        max_date = date_series.max()
        unique_dates = date_series.dt.normalize().nunique()
        total_calendar_days = (max_date.normalize() - min_date.normalize()).days + 1 if pd.notnull(min_date) else 0

        date_profile = {
            "start_date": str(min_date.date()) if pd.notnull(min_date) else "",
            "end_date": str(max_date.date()) if pd.notnull(max_date) else "",
            "calendar_span_days": int(total_calendar_days),
            "active_dates_count": int(unique_dates),
            "inactive_dates_count": int(total_calendar_days - unique_dates),
            "sparsity_pct": float((total_calendar_days - unique_dates) / total_calendar_days * 100) if total_calendar_days > 0 else 0.0
        }

        order_col = "Order" if "Order" in self.df.columns else "OrderID"
        customer_col = "CustomerID" if "CustomerID" in self.df.columns else "Customer"
        entity_counts = {
            "unique_orders": int(self.df[order_col].nunique()) if order_col in self.df.columns else 0,
            "unique_products": int(self.df["ProductID"].nunique()) if "ProductID" in self.df.columns else 0,
            "unique_categories": int(self.df["CategoryName"].nunique()) if "CategoryName" in self.df.columns else 0,
            "unique_customers": int(self.df[customer_col].nunique()) if customer_col in self.df.columns else 0,
            "unique_employees": int(self.df["EmployeeID"].nunique()) if "EmployeeID" in self.df.columns else 0,
            "lines_per_order_avg": float(n_rows / self.df[order_col].nunique()) if order_col in self.df.columns else 0.0
        }

        # Mathematical identity
        identity_check = self.verify_target_leakage_identity()

        return {
            "dimensions": {"rows": n_rows, "columns": n_cols},
            "column_classification": {
                "numerical_count": len(numeric_cols),
                "categorical_count": len(categorical_cols),
                "numerical_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": datetime_cols
            },
            "chunk_vs_non_chunk": {
                "chunk_fields": chunk_fields,
                "non_chunk_fields": non_chunk_fields
            },
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "target_analysis": target_analysis,
            "date_profile": date_profile,
            "entity_counts": entity_counts,
            "mathematical_identity": identity_check,
        }

    def run_full_understanding_audit(self) -> Dict[str, Any]:
        """Alias for profile_dataset()."""
        return self.profile_dataset()

    def verify_target_leakage_identity(self) -> Dict[str, Any]:
        """Empirically tests the identity: Sales = COS + GP."""
        if "Sales" in self.df.columns and "COS" in self.df.columns and "GP" in self.df.columns:
            computed_sales = self.df["COS"] + self.df["GP"]
            diff = np.abs(self.df["Sales"] - computed_sales)
            max_diff = float(np.max(diff))
            holds_strictly = bool(max_diff < 1e-4)

            return {
                "equation": "Sales = COS + GP",
                "holds_strictly": holds_strictly,
                "max_residual": max_diff,
                "mean_residual": float(np.mean(diff)),
                "design_directive": "COS and GP are computed post-sale. They must be purged to prevent 100% target leakage."
            }
        return {"error": "Columns Sales, COS, or GP not present"}


# Backward-compatible alias
DataUnderstandingProfiler = DataUnderstanding


# ==============================================================================
# 4. DATA CLEANER
# ==============================================================================
class DataCleaner:
    """Cleans raw transactional data, derives prices, and purges target leakage."""

    def __init__(self, df: Optional[pd.DataFrame] = None, drop_leakage: bool = True):
        self.raw_df = df.copy() if df is not None else None
        self.drop_leakage = drop_leakage
        self.cleaning_log: List[Dict[str, Any]] = []

    def clean_dataset(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Alias for clean()."""
        if df is not None:
            self.raw_df = df.copy()
        return self.clean()

    def clean(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Executes complete cleaning pipeline and returns (cleaned_df, cleaning_metrics)."""
        logger.info(f"Initiating Step 2 Data Cleaning on raw DataFrame with shape {self.raw_df.shape}")
        cleaned_df = self.raw_df.copy()

        # 1. Deduplication
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        exact_dups = initial_rows - len(cleaned_df)
        self.cleaning_log.append({
            "operation": "Deduplicate exact matching rows",
            "affected_rows": exact_dups,
            "method": "drop_duplicates()",
            "rationale": "Exact matching order-line records artificially duplicate transactions and distort true product demand."
        })

        # 2. Date Formatting & UTC Conversion
        cleaned_df["Order_Date"] = pd.to_datetime(cleaned_df["OrderDate"], utc=True)
        self.cleaning_log.append({
            "operation": "Standardize OrderDate to UTC timestamp (Order_Date)",
            "affected_rows": len(cleaned_df),
            "method": "pd.to_datetime(utc=True)",
            "rationale": "Uniform datetime formatting enables chronological sorting, expanding-window validation, and daily aggregation."
        })

        # 3. Standardize Order to OrderID
        if "Order" in cleaned_df.columns and "OrderID" not in cleaned_df.columns:
            cleaned_df.rename(columns={"Order": "OrderID"}, inplace=True)
            self.cleaning_log.append({
                "operation": "Rename Order to OrderID",
                "affected_rows": len(cleaned_df),
                "method": "Column rename",
                "rationale": "Aligns primary transaction identifier with enterprise relational schema standards."
            })

        # 4. Numerical Casting
        for num_col in ["Sales", "Quantity", "Discount"]:
            if num_col in cleaned_df.columns:
                if cleaned_df[num_col].dtype == object:
                    cleaned_df[num_col] = cleaned_df[num_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
                cleaned_df[num_col] = pd.to_numeric(cleaned_df[num_col], errors="coerce")

        # 5. Derive UnitPrice and DiscountRate
        effective_qty = cleaned_df["Quantity"].replace(0, np.nan)
        dollar_discount = cleaned_df["Discount"].fillna(0.0)
        gross_sales = cleaned_df["Sales"] + dollar_discount
        cleaned_df["UnitPrice"] = (gross_sales / effective_qty).round(2)
        cleaned_df["DiscountRate"] = np.where(gross_sales > 0, (dollar_discount / gross_sales).round(4), 0.0)
        self.cleaning_log.append({
            "operation": "Derive UnitPrice and DiscountRate from Gross Sales and Quantity",
            "affected_rows": len(cleaned_df),
            "method": "UnitPrice = (Sales + Discount) / Quantity; DiscountRate = Discount / (Sales + Discount)",
            "rationale": "Discount is dollar concession; back-calculating Gross Sales recovers exact positive catalog UnitPrice and fractional DiscountRate."
        })

        # 6. Filter Non-Positive Records
        invalid_qty_mask = cleaned_df["Quantity"] <= 0
        invalid_sales_mask = cleaned_df["Sales"] <= 0
        invalid_rows_count = int((invalid_qty_mask | invalid_sales_mask).sum())
        if invalid_rows_count > 0:
            cleaned_df = cleaned_df[~(invalid_qty_mask | invalid_sales_mask)].copy()

        self.cleaning_log.append({
            "operation": "Filter non-positive quantities and non-positive sales",
            "affected_rows": invalid_rows_count,
            "method": "Filter: Quantity > 0 and Sales > 0",
            "rationale": "Forward sales demand must be strictly positive; non-positive entries reflect cancellations or corrupted records."
        })

        # 7. Text Normalization
        text_cols = ["ProductName", "CategoryName", "City", "Country"]
        for col in text_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.title()

        self.cleaning_log.append({
            "operation": "Normalize categorical text casing and strip whitespace",
            "affected_rows": len(cleaned_df),
            "method": "str.strip().str.title()",
            "rationale": "Inconsistent casing fragments SKU sales across duplicate product name variants."
        })

        # 8. Outlier Auditing via IQR Fencing
        outlier_audit = {}
        for metric in ["UnitPrice", "Quantity", "Sales"]:
            if metric in cleaned_df.columns:
                q25 = float(cleaned_df[metric].quantile(0.25))
                q75 = float(cleaned_df[metric].quantile(0.75))
                iqr = q75 - q25
                lower_fence = max(0.0, q25 - 1.5 * iqr)
                upper_fence = q75 + 1.5 * iqr
                outliers = cleaned_df[(cleaned_df[metric] < lower_fence) | (cleaned_df[metric] > upper_fence)]
                outlier_audit[metric] = {
                    "q25": q25, "q75": q75, "iqr": iqr,
                    "lower_fence": lower_fence, "upper_fence": upper_fence,
                    "outlier_count": int(len(outliers)),
                    "outlier_pct": float(len(outliers) / len(cleaned_df) * 100),
                }

        self.cleaning_log.append({
            "operation": "Perform IQR outlier detection across financial metrics",
            "affected_rows": sum(v["outlier_count"] for v in outlier_audit.values()),
            "method": "1.5 * IQR Fencing (Audited, not deleted)",
            "rationale": "Large commercial wholesale orders represent authentic high-demand spikes; auditing flags them for tree models without discarding legitimate sales."
        })

        # 9. Purge Target Leakage and Administrative Fields
        cols_to_drop = [c for c in LEAKAGE_AND_ADMIN_COLUMNS if c in cleaned_df.columns]
        if "OrderDate" in cleaned_df.columns:
            cols_to_drop.append("OrderDate")
        cleaned_df.drop(columns=cols_to_drop, inplace=True)

        self.cleaning_log.append({
            "operation": "Purge post-transaction accounting leakage (COS, GP) and administrative attributes",
            "affected_rows": len(cleaned_df),
            "method": f"Drop columns: {cols_to_drop}",
            "rationale": "COS and GP are computed post-sale (Sales = COS + GP); keeping them causes 100% target leakage. Admin fields contain non-generalizable noise."
        })

        # Sort chronologically
        cleaned_df.sort_values(by=["Order_Date", "OrderID"], inplace=True)
        cleaned_df.reset_index(drop=True, inplace=True)

        metrics = {
            "initial_null_counts": {c: int(self.raw_df[c].isnull().sum()) for c in self.raw_df.columns if self.raw_df[c].isnull().sum() > 0},
            "exact_duplicates_removed": exact_dups,
            "invalid_records_removed": invalid_rows_count,
            "outlier_detection_iqr": outlier_audit,
            "columns_purged_for_leakage": cols_to_drop,
            "final_shape": {"rows": int(cleaned_df.shape[0]), "columns": int(cleaned_df.shape[1])},
            "cleaning_operations": self.cleaning_log,
        }

        logger.info(f"Step 2 Data Cleaning finished. Output shape: {cleaned_df.shape}")
        return cleaned_df, metrics


# ==============================================================================
# 5. SALES AGGREGATOR
# ==============================================================================
class SalesAggregator:
    """Aggregates transactional lines into continuous daily series across Macro, Meso, and Micro hierarchies."""

    def __init__(self, cleaned_transactions: Optional[pd.DataFrame] = None, processed_dir: Optional[str] = None):
        self.processed_dir = Path(processed_dir) if processed_dir else Path("data/processed")
        if cleaned_transactions is not None:
            self._set_data(cleaned_transactions)
        else:
            self.df = None

    def _set_data(self, cleaned_transactions: pd.DataFrame):
        self.df = cleaned_transactions.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["Order_Date"]):
            self.df["Order_Date"] = pd.to_datetime(self.df["Order_Date"], utc=True)
        self.df["Date"] = self.df["Order_Date"].dt.floor("D").dt.tz_localize(None)
        self.min_date = self.df["Date"].min()
        self.max_date = self.df["Date"].max()
        self.full_calendar_idx = pd.date_range(start=self.min_date, end=self.max_date, freq="D", name="Date")

    def aggregate_macro(self) -> pd.DataFrame:
        """Aggregates all products daily over the continuous 1,348 calendar days."""
        daily_agg = self.df.groupby("Date").agg(
            Sales=("Sales", "sum"),
            Quantity=("Quantity", "sum"),
            OrdersCount=("OrderID", "nunique"),
        )
        macro_df = daily_agg.reindex(self.full_calendar_idx, fill_value=0.0).reset_index()
        macro_df["Quantity"] = macro_df["Quantity"].astype(int)
        macro_df["OrdersCount"] = macro_df["OrdersCount"].astype(int)
        return macro_df

    def aggregate_macro_series(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is not None:
            self._set_data(df)
        return self.aggregate_macro()

    def aggregate_category(self) -> pd.DataFrame:
        """Aggregates merchandise category daily series (1,348 days × 8 categories = 10,784 rows)."""
        categories = self.df["CategoryName"].unique().tolist()
        cartesian_idx = pd.MultiIndex.from_product([self.full_calendar_idx, categories], names=["Date", "CategoryName"])

        cat_agg = self.df.groupby(["Date", "CategoryName"]).agg(
            Sales=("Sales", "sum"),
            Quantity=("Quantity", "sum"),
        )
        cat_df = cat_agg.reindex(cartesian_idx, fill_value=0.0).reset_index()
        cat_df["Quantity"] = cat_df["Quantity"].astype(int)
        return cat_df

    def aggregate_category_series(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is not None:
            self._set_data(df)
        return self.aggregate_category()

    def aggregate_sku(self) -> pd.DataFrame:
        """Aggregates daily series for each catalog SKU."""
        sku_agg = self.df.groupby(["Date", "ProductID", "ProductName", "CategoryName"]).agg(
            Sales=("Sales", "sum"),
            Quantity=("Quantity", "sum"),
        ).reset_index()
        sku_agg["Quantity"] = sku_agg["Quantity"].astype(int)
        return sku_agg

    def aggregate_sku_series(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is not None:
            self._set_data(df)
        return self.aggregate_sku()

    def save_all_hierarchies(self, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Saves Macro, Category, and SKU datasets in Parquet and CSV formats."""
        out_dir = Path(output_dir) if output_dir else self.processed_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        macro_df = self.aggregate_macro()
        cat_df = self.aggregate_category()
        sku_df = self.aggregate_sku()

        paths = {
            "macro_parquet": str(out_dir / "all_products_daily.parquet"),
            "macro_csv": str(out_dir / "all_products_daily.csv"),
            "category_parquet": str(out_dir / "category_daily.parquet"),
            "category_csv": str(out_dir / "category_daily.csv"),
            "sku_parquet": str(out_dir / "sku_daily.parquet"),
            "sku_csv": str(out_dir / "sku_daily.csv"),
        }

        macro_df.to_parquet(paths["macro_parquet"], index=False)
        macro_df.to_csv(paths["macro_csv"], index=False)
        cat_df.to_parquet(paths["category_parquet"], index=False)
        cat_df.to_csv(paths["category_csv"], index=False)
        sku_df.to_parquet(paths["sku_parquet"], index=False)
        sku_df.to_csv(paths["sku_csv"], index=False)

        logger.info(f"All processed datasets saved successfully into {out_dir}")
        return paths

    def save_processed_datasets(self, macro_df: pd.DataFrame, cat_df: pd.DataFrame, sku_df: pd.DataFrame) -> Dict[str, str]:
        """Saves pre-aggregated DataFrames to the output directory."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "macro_parquet": str(self.processed_dir / "all_products_daily.parquet"),
            "macro_csv": str(self.processed_dir / "all_products_daily.csv"),
            "category_parquet": str(self.processed_dir / "category_daily.parquet"),
            "category_csv": str(self.processed_dir / "category_daily.csv"),
            "sku_parquet": str(self.processed_dir / "sku_daily.parquet"),
            "sku_csv": str(self.processed_dir / "sku_daily.csv"),
        }
        macro_df.to_parquet(paths["macro_parquet"], index=False)
        macro_df.to_csv(paths["macro_csv"], index=False)
        cat_df.to_parquet(paths["category_parquet"], index=False)
        cat_df.to_csv(paths["category_csv"], index=False)
        sku_df.to_parquet(paths["sku_parquet"], index=False)
        sku_df.to_csv(paths["sku_csv"], index=False)
        return paths


def run_data_pipeline(
    raw_csv_path: str = "data/raw/New_Query_2026_10_22_10_01_37.csv",
    output_dir: str = "report_stage",
    processed_dir: str = "data/processed",
) -> Dict[str, Any]:
    """
    Executes the end-to-end Step 1 (Data Understanding) and Step 2 (Data Cleaning) pipeline.
    Runs data ingestion, schema validation, understanding profiling, cleaning,
    and multi-tier continuous daily aggregation.
    """
    logger.info("=" * 80)
    logger.info("SALESSIGHT AI: STEP 1 (DATA UNDERSTANDING) & STEP 2 (DATA CLEANING) PIPELINE")
    logger.info("=" * 80)
    set_seed(42)

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = Path(processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Data
    loader = DataLoader(raw_csv_path)
    raw_df = loader.load_raw_data()

    # 2. Schema Validation
    validator = DataValidator()
    validation_result = validator.validate_data_integrity(raw_df)
    logger.info(f"Schema Validation Result: is_valid={validation_result.is_valid}")

    # 3. STEP 1: Data Understanding Profiling
    logger.info("--- Executing Step 1: Data Understanding Profiler ---")
    profiler = DataUnderstandingProfiler(raw_df)
    understanding_audit = profiler.profile_dataset()

    step1_json_path = report_dir / "step1_understanding_metrics.json"
    with open(step1_json_path, "w", encoding="utf-8") as f:
        json.dump(understanding_audit, f, indent=2, default=str)
    logger.info(f"Saved Step 1 audit metrics to: {step1_json_path}")

    # 4. STEP 2: Data Cleaning
    logger.info("--- Executing Step 2: Data Cleaning & Leakage Elimination ---")
    cleaner = DataCleaner(drop_leakage=True)
    cleaned_df, cleaning_audit = cleaner.clean_dataset(raw_df)

    cleaned_tx_pq = proc_dir / "cleaned_transactions.parquet"
    cleaned_df.to_parquet(cleaned_tx_pq, index=False)
    cleaned_df.to_csv(proc_dir / "cleaned_transactions.csv", index=False)
    cleaning_audit["cleaned_transactions_parquet"] = str(cleaned_tx_pq)

    # 5. Continuous Daily Aggregation
    logger.info("--- Executing Step 2: Continuous Daily Grid Aggregation ---")
    aggregator = SalesAggregator(processed_dir=str(proc_dir))
    macro_series = aggregator.aggregate_macro_series(cleaned_df)
    cat_series = aggregator.aggregate_category_series(cleaned_df)
    sku_series = aggregator.aggregate_sku_series(cleaned_df)

    saved_files = aggregator.save_processed_datasets(macro_series, cat_series, sku_series)
    cleaning_audit["aggregated_datasets"] = saved_files
    cleaning_audit["macro_summary"] = {
        "calendar_days": len(macro_series),
        "total_revenue": float(macro_series["Sales"].sum()),
        "total_quantity": int(macro_series["Quantity"].sum()),
        "total_orders": int(macro_series["OrdersCount"].sum()),
        "zero_revenue_days": int((macro_series["Sales"] == 0).sum()),
        "zero_revenue_pct": float((macro_series["Sales"] == 0).mean() * 100),
    }

    step2_json_path = report_dir / "step2_cleaning_metrics.json"
    with open(step2_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_audit, f, indent=2, default=str)
    logger.info(f"Saved Step 2 audit metrics to: {step2_json_path}")

    logger.info("=" * 80)
    logger.info("STEP 1 & STEP 2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    return {
        "understanding_metrics": understanding_audit,
        "cleaning_metrics": cleaning_audit,
        "cleaned_df": cleaned_df,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    run_data_pipeline()

