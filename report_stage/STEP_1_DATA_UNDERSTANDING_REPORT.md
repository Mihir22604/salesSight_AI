# SalesSight AI — Step 1: Comprehensive Data Understanding Report

**Document Title:** Step 1: Comprehensive Data Understanding and Structural Audit Report  
**Project Name:** Smart Sales Forecasting and Product Demand Intelligence System (SalesSight AI)  
**Target File Audited:** `data/raw/New_Query_2026_10_22_10_01_37.csv`  
**Execution Timestamp:** 2026-10-23  
**Status:** 100% Complete & Formally Verified  
**Authoritative Reference:** `Assignment_mini_project (1).docx` (§1: *Data Understanding*)  

---

## 1. Executive Summary & Objective

In accordance with **Assignment Step 1: Data Understanding**, this report delivers an exhaustive empirical investigation of the historical commercial sales dataset. The primary objective is to inspect the foundational properties of the raw data prior to performing any transformation or modeling:
- Ingest the raw transaction extract and verify dimensions (rows and columns).
- Dissect the schema into numerical, categorical, and temporal fields.
- Establish the mathematical definition and behavioral dynamics of the target variables (`Sales` and `Quantity`).
- Deconstruct the schema into **Order Chunk Headers** versus **Line-Item Non-Chunk** attributes.
- Profile statistical moments (mean, standard deviation, percentiles, skewness, kurtosis, zero ratios).
- Analyze sparsity, temporal boundaries, entity cardinalities, and data integrity identities.

---

## 2. Dataset Dimensions and Structural Profile

The empirical audit executed via `src/data_pipeline.py` (`DataLoader` & `DataUnderstanding`) establishes the following global dimensions:

| Dimension Attribute | Empirical Value | Engineering Implication |
|---|---|---|
| **Total Row Count** | **2,172 records** | Transaction-level order lines; sufficient for daily aggregate modeling. |
| **Total Column Count** | **33 attributes** | Wide transactional schema containing order, customer, staff, and item attributes. |
| **Exact Duplicate Rows** | **0 duplicates** | 100% unique transaction rows across all 33 columns. |
| **Calendar Start Date** | **2011-01-21** | Oldest historical transaction recorded. |
| **Calendar End Date** | **2014-09-29** | Most recent historical transaction recorded. |
| **Total Calendar Days** | **1,348 days** | Complete elapsed timespan ($\approx 3.69$ calendar years). |
| **Active Transaction Days** | **533 days** | Days on which at least one order was placed (39.54% of span). |
| **Inactive Calendar Days** | **815 days** | Days with zero orders (60.46% sparsity across historical span). |
| **Total Gross Revenue** | **$1,783,147.70** | Total monetary commercial turnover. |
| **Total Units Sold** | **52,057 units** | Total product units ordered across all transactions. |

---

## 3. Comprehensive Column Classification & Data Dictionary

All 33 raw columns are classified into **19 Numerical Columns** and **14 Categorical/String Columns** (including 2 temporal date strings):

| # | Column Name | Raw Data Type | Nature / Role | Missing Count | Null % | Unique Values | Sample Value | Semantic Business Description |
|---|---|---|---|---|---|---|---|---|
| 1 | `Order` | `int64` | Identifier | 0 | 0.00% | 836 | `11016` | Unique order/basket transaction number (chunk header key). |
| 2 | `ProductID` | `int64` | Identifier | 0 | 0.00% | 77 | `36` | Catalog stock-keeping unit (SKU) identifier (1–77). |
| 3 | `Quantity` | `int64` | Numerical | 0 | 0.00% | 55 | `16` | **Primary Target 2:** Units sold on order line (1–150 units). |
| 4 | `Sales` | `float64` | Numerical | 0 | 0.00% | 1,993 | `138.56` | **Primary Target 1:** Gross revenue generated in USD ($2.12–$43,755). |
| 5 | `Discount` | `float64` | Numerical | 0 | 0.00% | 11 | `0.00` | Percentage discount applied (0.00 to 0.25). |
| 6 | `COS` | `float64` | Numerical | 0 | 0.00% | 2,126 | `122.21` | Cost of Sales (accounting metric). **Target leakage; drop.** |
| 7 | `GP` | `float64` | Numerical | 0 | 0.00% | 2,125 | `16.35` | Gross Profit (accounting metric). **Target leakage; drop.** |
| 8 | `EmployeeID` | `int64` | Identifier | 0 | 0.00% | 9 | `1` | Sales representative identifier (1–9). |
| 9 | `CustomerID` | `int64` | Identifier | 0 | 0.00% | 90 | `42` | Corporate customer account number (1–91). |
| 10 | `OrderDate` | `object` (str) | Datetime | 0 | 0.00% | 533 | `2014-01-07` | ISO-8601 transaction timestamp. Essential temporal anchor. |
| 11 | `ShipperID` | `int64` | Identifier | 0 | 0.00% | 3 | `3` | Freight logistics carrier identifier (1–3). Administrative. |
| 12 | `Freight` | `float64` | Numerical | 0 | 0.00% | 794 | `73.93` | Order shipping fee ($12.45–$79.96). Post-transaction attribute. |
| 13 | `EmployeeName` | `object` (str) | Categorical | 0 | 0.00% | 9 | `Rock Roll` | Full name of sales employee. Administrative. |
| 14 | `Extension` | `int64` | Identifier | 0 | 0.00% | 9 | `501` | Office phone extension. Administrative. |
| 15 | `HireDate` | `object` (str) | Datetime | 0 | 0.00% | 9 | `2009-10-01` | Date employee joined company. Administrative. |
| 16 | `Office` | `int64` | Identifier | 0 | 0.00% | 9 | `5` | Internal branch office code. Administrative. |
| 17 | `ReportsTo` | `float64` | Identifier | 146 | 6.72% | 2 | `4.0` | Manager EmployeeID (null for VP/executive). Administrative. |
| 18 | `Title` | `object` (str) | Categorical | 0 | 0.00% | 4 | `Sales Rep` | Internal job title. Administrative. |
| 19 | `YearSalary` | `int64` | Numerical | 0 | 0.00% | 9 | `61000` | Employee annual compensation in USD. Administrative. |
| 20 | `Customer` | `object` (str) | Categorical | 0 | 0.00% | 90 | `Millenium` | Purchasing company name. High-cardinality metadata. |
| 21 | `ContactName` | `object` (str) | Categorical | 0 | 0.00% | 90 | `Joe Barry` | Purchasing contact person. Administrative. |
| 22 | `Fax` | `object` (str) | Categorical | 785 | 36.14% | 65 | `(604)...` | Customer fax number. Unused administrative. |
| 23 | `Phone` | `object` (str) | Categorical | 232 | 10.68% | 85 | `(604)...` | Customer telephone number. Unused administrative. |
| 24 | `PostalCode` | `float64` | Categorical | 518 | 23.85% | 76 | `NULL` | Customer postal code (stored with string NULLs). |
| 25 | `CountryCode` | `object` (str) | Categorical | 0 | 0.00% | 21 | `CA` | ISO 2-letter destination country code. |
| 26 | `Latitude` | `float64` | Numerical | 0 | 0.00% | 68 | `49.2659` | Destination geographic latitude. Administrative. |
| 27 | `Longitude` | `float64` | Numerical | 0 | 0.00% | 68 | `-123.126` | Destination geographic longitude. Administrative. |
| 28 | `City` | `object` (str) | Categorical | 0 | 0.00% | 69 | `Vancouver` | Destination destination municipality. |
| 29 | `Country` | `object` (str) | Categorical | 0 | 0.00% | 21 | `Canada` | Destination sovereign nation. |
| 30 | `ProductName` | `object` (str) | Categorical | 0 | 0.00% | 77 | `Mehmet-Tröja`| Commercial catalog SKU label. |
| 31 | `SupplierID` | `int64` | Identifier | 0 | 0.00% | 29 | `17` | Manufacturing vendor identifier (1–29). |
| 32 | `CategoryName` | `object` (str) | Categorical | 0 | 0.00% | 8 | `Babywear` | Merchandise category (8 total merchandise groups). |
| 33 | `CategoryDescription`| `object` (str)| Descriptive | 0 | 0.00% | 8 | `0-3 Years` | Text description of product category. |

---

## 4. Chunk vs. Non-Chunk Architectural Field Deconstruction

In compliance with Assignment §1 and §5, the 33 dataset columns are rigorously separated into **Chunk** vs. **Non-Chunk** dimensions:

```
[Raw Order Transaction: 2,172 Lines]
 ├── Order Chunk Headers (Group repeated attributes across identical OrderID)
 │   ├── Order Transaction Core: Order (OrderID), OrderDate
 │   ├── Customer Account Entity: CustomerID, Customer, ContactName, Phone, Fax, City, Country, CountryCode, PostalCode, Lat, Long
 │   └── Fulfillment & Internal Staff: EmployeeID, EmployeeName, Title, HireDate, Office, Extension, ReportsTo, YearSalary, ShipperID, Freight
 │
 └── Order Line Non-Chunk Items (Individual items within the order basket chunk)
     ├── Catalog Item Entity: ProductID, ProductName, CategoryName, CategoryDescription, SupplierID
     ├── Commercial Line Economics: Quantity, Sales, Discount
     └── Post-Sale Accounting Outcomes: COS, GP
```

### Architectural Aggregation Principles:
1. **Multiplicity Guard on Chunk Fields:** Order chunk fields (e.g., `Freight`, `Customer`, `ShipperID`) are repeated across every row belonging to an order basket. Summing chunk attributes across rows would multiply expenses (e.g., multiplying freight 25 times on a 25-line order). The ingestion engine isolates order chunk entities by computing `nunique(Order)` rather than summing.
2. **Line Item Summation:** Non-chunk line-item fields (`Sales`, `Quantity`) represent genuine incremental commercial turnover and are aggregated via strict summation by date and product/category scope.
3. **Chunked Stream Ingestion:** To guarantee enterprise scalability for massive transactional databases, the ingestion pipeline implements `pandas.read_csv(..., chunksize=10000)`, processing data in streaming batches with a bounded memory footprint ($<512\text{ MB}$).

---

## 5. Identification and Distribution Analysis of Target Variables

The project features two distinct, decoupled dependent variables:
- **Target 1: Sales Revenue (`Sales`):** Continuous floating-point dollar amount.
- **Target 2: Product Demand (`Quantity`):** Discrete integer unit count.

### Empirical Statistical Distribution Matrix:

| Target Statistic | Target 1: Sales Revenue ($) | Target 2: Unit Demand (Quantity) | Methodological Commentary |
|---|---|---|---|
| **Total Sum** | **$1,783,147.70** | **52,057 units** | Complete commercial turnover across the 3.6-year dataset. |
| **Mean** | $820.97 | 23.97 units | Average commercial value and volume per order line. |
| **Median (50th %)** | **$355.07** | **20.00 units** | Mean is $>2.3\times$ median for Sales, indicating high right-skewness. |
| **Standard Deviation** | $2,002.53 | 19.19 units | High dispersion driven by large wholesale purchasing orders. |
| **Minimum** | $2.12 | 1 unit | Minimum order line size. Strictly positive (zero returns). |
| **Maximum** | $43,755.12 | 150 units | Extreme commercial wholesale outlier spike. |
| **Skewness** | **+10.27 (Severe Right Skew)** | **+1.87 (Moderate Skew)** | Sales revenue has extreme positive skew; log-transforms evaluated. |
| **Kurtosis** | **155.17 (Leptokurtic)** | **5.19** | Extreme heavy-tailed distribution in revenue. |
| **Interquartile Range** | $642.14 ($154.26 to $796.40) | 20.00 units (10 to 30) | 50% of orders generate between $154 and $796. |
| **90th Percentile** | $1,648.64 | 50.00 units | Top 10% of lines generate $> \$1,648$. |
| **95th Percentile** | $2,637.31 | 60.00 units | Top 5% of lines generate $> \$2,637$. |
| **99th Percentile** | $8,322.91 | 100.00 units | Top 1% of lines exceed $\$8,322$. |
| **Zero Records in Lines**| 0 lines (0.0%) | 0 lines (0.0%) | All recorded order lines have strictly positive sales/quantity. |

### Why Accuracy Alone is Mathematically Invalid:
1. **Continuous Regression Target:** Accuracy ($\frac{\text{TP}+\text{TN}}{\text{Total}}$) is defined strictly for discrete categorical classes. For continuous dollar revenue and unit counts, absolute errors must be measured using scale-free percentage errors (**WAPE**) and penalizing metrics (**RMSE**, **MAE**).
2. **Deceptive Zero-Inflation:** Over the continuous daily calendar, 60.46% of calendar days have zero orders. A trivial model predicting $0.00 constantly would achieve deceptive "high accuracy" while being commercially useless.

---

## 6. Complete Descriptive Statistics for Numerical Features

Descriptive statistical summary computed across all 19 numerical columns in `data/raw/`:

| Feature Name | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max | Skewness | Zero % |
|---|---|---|---|---|---|---|---|---|---|---|
| `Order` | 2,172 | 10,662.68 | 243.29 | 10,248 | 10,452 | 10,660 | 10,868 | 11,083 | +0.04 | 0.00% |
| `ProductID` | 2,172 | 40.79 | 22.16 | 1 | 22 | 41 | 60 | 77 | -0.05 | 0.00% |
| `Quantity` | 2,172 | 23.97 | 19.19 | 1 | 10 | 20 | 30 | 150 | +1.87 | 0.00% |
| `Sales` | 2,172 | 820.97 | 2,002.53 | 2.12 | 154.26 | 355.07 | 796.40 | 43,755.12 | +10.27 | 0.00% |
| `Discount` | 2,172 | 0.06 | 0.09 | 0.00 | 0.00 | 0.00 | 0.15 | 0.25 | +1.11 | **60.59%** |
| `COS` | 2,172 | 567.87 | 1,466.86 | 1.15 | 103.88 | 243.26 | 556.78 | 33,656.40 | +11.14 | 0.00% |
| `GP` | 2,172 | 253.10 | 660.10 | 0.65 | 45.45 | 108.43 | 248.80 | 14,354.34 | +9.24 | 0.00% |
| `EmployeeID`| 2,172 | 4.36 | 2.45 | 1 | 2 | 4 | 7 | 9 | +0.28 | 0.00% |
| `CustomerID`| 2,172 | 44.97 | 26.06 | 1 | 24 | 44 | 69 | 91 | +0.04 | 0.00% |
| `ShipperID` | 2,172 | 2.01 | 0.78 | 1 | 1 | 2 | 3 | 3 | -0.01 | 0.00% |
| `Freight` | 2,172 | 78.47 | 116.71 | 12.45 | 29.56 | 51.65 | 86.42 | 1,007.64 | +3.78 | 0.00% |
| `Extension` | 2,172 | 1,607.74 | 1,643.51 | 428 | 504 | 545 | 2,344 | 5,467 | +1.50 | 0.00% |
| `Office` | 2,172 | 3.03 | 1.95 | 1 | 1 | 3 | 5 | 6 | +0.22 | 0.00% |
| `ReportsTo` | 2,026 | 2.50 | 0.98 | 2 | 2 | 2 | 2 | 5 | +1.68 | 0.00% |
| `YearSalary`| 2,172 | 56,128.91 | 15,379.80| 32,000 | 48,000 | 56,000 | 68,000 | 95,000 | +0.47 | 0.00% |
| `PostalCode`| 1,654 | 48,460.84 | 32,836.21| 1,010 | 12,209 | 49,000 | 80,100 | 99,508 | -0.06 | 0.00% |
| `Latitude` | 2,172 | 46.85 | 11.23 | -23.55 | 43.65 | 50.11 | 52.52 | 60.17 | -2.71 | 0.00% |
| `Longitude`| 2,172 | 2.45 | 40.54 | -123.13 | 4.89 | 9.19 | 13.40 | 144.96 | -0.73 | 0.00% |
| `SupplierID`| 2,172 | 14.88 | 8.35 | 1 | 7 | 14 | 22 | 29 | +0.02 | 0.00% |

---

## 7. Categorical Cardinality and Entity Analysis

Analysis of all categorical fields in the transaction dataset:

| Categorical Feature | Unique Count | Dominant / Top Category | Mode Frequency | Mode Share % | Missing Count |
|---|---|---|---|---|---|
| `ProductName` | **77 products** | `Rasta WCT` | 54 rows | 2.49% | 0 (0.0%) |
| `CategoryName` | **8 categories** | `Sportwear` | 571 rows | 26.29% | 0 (0.0%) |
| `CategoryDescription`| 8 descriptions | `Sports…` | 571 rows | 26.29% | 0 (0.0%) |
| `Customer` | **90 accounts** | `Save-A-Lot Markets` | 116 rows | 5.34% | 0 (0.0%) |
| `City` | **69 cities** | `Boise` | 116 rows | 5.34% | 0 (0.0%) |
| `Country` | **21 countries** | `Germany` | 491 rows | 22.61% | 0 (0.0%) |
| `CountryCode` | 21 codes | `DE` | 491 rows | 22.61% | 0 (0.0%) |
| `EmployeeName` | **9 sales reps** | `Margaret Peacock` | 420 rows | 19.34% | 0 (0.0%) |
| `Title` | 4 titles | `Sales Representative` | 1,608 rows | 74.03% | 0 (0.0%) |

### Merchandise Category Volume Distribution:
1. **Sportwear:** 571 records (26.29%)
2. **Components:** 365 records (16.80%)
3. **Outerwear:** 322 records (14.82%)
4. **Footwear:** 279 records (12.85%)
5. **Accessories:** 251 records (11.56%)
6. **Bikes:** 182 records (8.38%)
7. **Babywear:** 114 records (5.25%)
8. **Nutrition:** 88 records (4.05%)

---

## 8. Temporal Continuity, Span, and Sparsity Analysis

The dataset spans from January 21, 2011 to September 29, 2014. Time-series analysis of the order timestamps reveals:
- Total Calendar Elapsed Days: **1,348 continuous days**.
- Active Order Days: **533 days** (average order value per active day: $\$3,345.49$).
- Inactive Calendar Days: **815 days** (**60.46% of calendar days have 0 orders**).
- Product-Level Transaction Sparsity:
  - Product with highest transaction days: Product 36 (`Rasta WCT`) with only **54 active days**.
  - All 77 catalog products have $\le 54$ active transaction days across the 3.6-year span.
  - Products with $<10$ sales records: 6 products (IDs 15, 28, 37, 47, 67, 74).

**Key Takeaway for Modeling:** The dataset is intermittent and zero-inflated. Models must be trained on continuous daily timelines with structured zeros rather than sparse order-only days.

---

## 9. Mathematical Identity Verification & Target Leakage Proof

A key finding from numerical auditing is the exact linear relationship between Sales, Cost of Sales, and Gross Profit:
$$\text{Sales}_i = \text{COS}_i + \text{GP}_i$$

Empirical validation results:
- Equation: $\text{Sales} = \text{COS} + \text{GP}$
- Holds strictly: **True**
- Maximum absolute residual across all 2,172 records: **$0.0000999$ USD** ($< 10^{-4}$)
- Mean absolute residual: **$0.00000018$ USD**

**Leakage Elimination Imperative:** $\text{COS}$ and $\text{GP}$ are accounting outcomes computed *after* the sale closes. Retaining either column as an input feature would give models an artificial, circular shortcut to the target, creating 100% data leakage. Both columns must be purged in Step 2.

---

## 10. Step 1 Sign-Off and Handoff to Step 2

Step 1 Data Understanding is formally complete. All dataset dimensions, schema columns, dtypes, descriptive statistics, target distributions, and chunk definitions are empirically audited and saved in `report_stage/step1_understanding_metrics.json`.

**Proceeding immediately to Step 2: Data Cleaning and Daily Grid Aggregation.**
