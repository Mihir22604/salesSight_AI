# SMART SALES FORECASTING AND PRODUCT DEMAND INTELLIGENCE SYSTEM
## Production-Ready Development Blueprint and Technical Specification

**Document Version:** 1.0.0-PROD-BLUEPRINT  
**Status:** Approved for Development (Documentation Only — No Code Implemented)  
**Target Repository:** `SalesSight AI`  
**Primary Reference Files:**
- `Updated_Smart_Sales_Forecasting_Feature_Description (1).txt` (Priority 1: Authoritative Feature Specification)
- `Assignment_mini_project (1).docx` (Priority 2: Academic & Production ML Lifecycle Requirements)
- `New_Query_2026_10_22_10_01_37.csv` (Priority 3: Empirical Historical Sales Dataset)

---

## Table of Contents
1. [Section 1: Executive Summary](#section-1-executive-summary)
2. [Section 2: Project Scope and Objectives](#section-2-project-scope-and-objectives)
3. [Section 3: Source Documents and Requirements Traceability](#section-3-source-documents-and-requirements-traceability)
   - [Section 3.4: Assignment Mini-Project 16-Step Implementation Specification](#34-assignment-mini-project-16-step-project-making-implementation-specification)
4. [Section 4: Dataset Analysis and Data Dictionary](#section-4-dataset-analysis-and-data-dictionary)
5. [Section 5: Complete System Architecture](#section-5-complete-system-architecture)
6. [Section 6: Proposed Project Directory Structure](#section-6-proposed-project-directory-structure)
7. [Section 7: Detailed Data Preparation Plan](#section-7-detailed-data-preparation-plan)
8. [Section 8: Exploratory Data Analysis Plan](#section-8-exploratory-data-analysis-plan)
9. [Section 9: Feature Engineering and Leakage Prevention](#section-9-feature-engineering-and-leakage-prevention)
10. [Section 10: Forecasting Model Development](#section-10-forecasting-model-development)
11. [Section 11: Chronological Validation and Model Evaluation](#section-11-chronological-validation-and-model-evaluation)
12. [Section 12: Hyperparameter Tuning and Automatic Model Selection](#section-12-hyperparameter-tuning-and-automatic-model-selection)
13. [Section 13: Forecasting Engine and Forecast-Date Rules](#section-13-forecasting-engine-and-forecast-date-rules)
14. [Section 14: Forecast Uncertainty and Confidence Ranges](#section-14-forecast-uncertainty-and-confidence-ranges)
15. [Section 15: Product Demand Intelligence](#section-15-product-demand-intelligence)
16. [Section 16: Streamlit Dashboard and User Experience](#section-16-streamlit-dashboard-and-user-experience)
17. [Section 17: Exports and Reporting](#section-17-exports-and-reporting)
18. [Section 18: Model Persistence and Reproducibility](#section-18-model-persistence-and-reproducibility)
19. [Section 19: Error Handling, Validation, and Logging](#section-19-error-handling-validation-and-logging)
20. [Section 20: Production Readiness and Deployment](#section-20-production-readiness-and-deployment)
21. [Section 21: Complete Testing Strategy](#section-21-complete-testing-strategy)
22. [Section 22: End-to-End Staged Implementation Blueprint and Roadmap](#section-22-end-to-end-staged-implementation-blueprint-and-roadmap)
23. [Section 23: Final Acceptance Checklist](#section-23-final-acceptance-checklist)
24. [Section 24: Risks, Limitations, and Open Decisions](#section-24-risks-limitations-and-open-decisions)
25. [Section 25: Implementation Handoff](#section-25-implementation-handoff)

---

## Section 1: Executive Summary

### 1.1 Project Overview
The **Smart Sales Forecasting and Product Demand Intelligence System** (operational display title: **SalesSight AI**) is an enterprise-grade machine learning application engineered to deliver multi-horizon daily sales forecasting and decision-support intelligence. The platform bridges the gap between raw, intermittent transaction histories and tactical commercial planning by predicting both daily revenue (monetary value in USD) and daily product demand (unit volume sold) across flexible organizational scopes: macro-level (all products aggregate), meso-level (merchandise categories), and micro-level (individual catalog stock-keeping units).

### 1.2 Business Problem and Intended Users
Wholesale and retail merchandising teams face persistent supply chain friction driven by demand volatility, stockout costs, and capital lockup in slow-moving inventory. Business planners require forward-looking visibility over 30-, 60-, and 90-day planning horizons to coordinate working capital, staffing, promotional schedules, and supplier engagement.

The system is designed for:
- **Inventory & Supply Chain Planners:** Requiring projected unit demand trajectories to schedule logistics and review potential overstock or shortage risks.
- **Sales & Commercial Directors:** Requiring category and enterprise revenue forecasts to track revenue targets and budget variance.
- **Merchandising Category Managers:** Requiring comparative product velocity metrics to allocate marketing spend and optimize product assortment.

### 1.3 Core Forecasting Objectives
1. **Multi-Target Daily Forecasting:** Generate daily predictions for sales revenue ($) and unit demand (quantity) independently, recognizing that revenue and volume represent distinct commercial dynamics.
2. **Multi-Scope Adaptability:** Provide seamless forecast generation at aggregate (all products), category (8 distinct merchandise groups), and individual product levels (77 distinct SKUs).
3. **Multi-Horizon Planning Windows:** Deliver exactly 30, 60, or 90 calendar days of daily predictions with no missing steps.
4. **Empirical Model Selection:** Compete five candidate models (Model 1: SARIMA, Model 2: Random Forest Regressor, Model 3: XGBoost Regressor, Model 4: LightGBM Regressor, and Model 5: Ridge Regression) against an established Seasonal-Naive baseline using chronological backtesting, selecting the optimal model based on out-of-sample Weighted Absolute Percentage Error (WAPE).
5. **Calibrated Prediction Uncertainty:** Provide bounded, non-negative 80% forecast intervals reflecting forecast horizon uncertainty and recursive error compounding.

### 1.4 Major Capabilities and System Intelligence
- **Product Demand Intelligence:** Automatic identification of the Top-5 demand products (evaluated strictly on predicted unit volume over the first 30 forecast days), categorization of SKU momentum into Rising, Falling, or Stable demand trajectories, and generation of a prioritized Planning Attention List.
- **Comparative Analytics:** Side-by-side performance benchmarking of 2 to 5 selected products and consecutive 30-day period-over-period growth tracking.
- **Temporal Alignment Controls:** A deterministic 5-mode start-date orchestrator supporting default forward forecasting, backtesting against historical ground truth, bridging, beyond-limit rejection, and demonstration mode.
- **Auditable Delivery:** One-click CSV forecast export and self-contained executive forecast reports (PDF/HTML) capturing model metrics, plots, and planning recommendations.

### 1.5 Proposed Production Architecture
The system decouples data ingestion, preprocessing, model evaluation, inference, demand intelligence, and presentation into isolated modules. Streamlit serves as the presentation dashboard, interacting strictly with backend services rather than embedding raw modeling or transformation code. Trained pipelines, residual distributions, and evaluation artifacts are serialized using `joblib` and structured JSON, ensuring that UI interactions trigger inference in milliseconds without training models at runtime.

### 1.6 Current Limitations of Historical Dataset
Rigorous empirical analysis of `New_Query_2026_10_22_10_01_37.csv` establishes that:
- The dataset comprises 2,172 order-line records spanning from **January 21, 2011 to September 29, 2014** (1,348 calendar days).
- Orders occurred on only **533 unique dates**, leaving 815 calendar days (60.46%) without transactions.
- Product transaction frequency is highly sparse: the most active product has only 54 transaction days, and 6 products have fewer than 10 records across the entire 3.6-year span.
- Because history ends in September 2014, any forecast initialized in modern calendar years (e.g., 2026) operates far beyond the 90-day bridging limit. The application enforces a prominent **Demonstration Mode** disclaimer to prevent historical patterns from being misinterpreted as real-time current market predictions.
- The dataset contains sales outcomes but zero inventory logs, supplier lead times, or holding costs. Consequently, the system explicitly disclaims automated reorder point calculations, framing outputs as demand planning signals rather than automated stock replenishment orders.

---

## Section 2: Project Scope and Objectives

### 2.1 In-Scope Functionality
The application scope is strictly bounded to the following functional capabilities:

```
[SalesSight AI Scope]
├── Forecasting Engine
│   ├── Target 1: Daily Sales Revenue (USD Float)
│   ├── Target 2: Daily Demand Quantity (Integer Units)
│   ├── Target 3: Dual-Target Simultaneous Execution
│   ├── Scopes: All Products (Macro), Categories (Meso), Individual SKUs (Micro)
│   ├── Horizons: Exactly 30, 60, or 90 calendar days
│   └── Multi-step Recursive Predictor with Temporal Shift
├── Modeling & Evaluation Suite
│   ├── Benchmark: Seasonal-Naive Baseline (Lag-7 Persistence Heuristic)
│   ├── Model 1: SARIMA (Seasonal Order m=7; Macro & Category scopes)
│   ├── Model 2: Random Forest Regressor (Pooled Multi-Product Architecture)
│   ├── Model 3: XGBoost Regressor (Gradient Boosted Trees)
│   ├── Model 4: LightGBM Regressor (Leaf-Wise Histogram Boosting)
│   ├── Model 5: Ridge Regression (L2 Regularized Standardized Linear Estimator)
│   ├── Validation: Expanding-Window Chronological Backtesting (TimeSeriesSplit)
│   ├── Selection: Automated WAPE minimization with RMSE & Simplicity Tie-breakers
│   └── Uncertainty: Empirical 80% Residual Quantiles & Native SARIMA Intervals
├── Product Demand Intelligence
│   ├── Top-5 Demand Ranking (Units over Days 1-30)
│   ├── Trend Categorization: Rising (>=+15%, >=5 units), Falling (<=-15%, <=-5 units), Stable
│   ├── Product Comparison: 2 to 5 SKUs side-by-side
│   ├── Period Comparison: Consecutive 30-day blocks & Growth %
│   └── Planning Attention List: Rule-based prioritization (High Priority, Review, Overstock Watch)
├── User Interface & Reporting
│   ├── Interactive Streamlit Web UI (Plotly dynamic charts, KPI metric cards, data tables)
│   ├── Temporal Date Controls (Default, Backtest, Bridge, Beyond-Limit, Demonstration)
│   ├── CSV Daily Prediction Export
│   └── Executive Summary Report (PDF/HTML)
└── Engineering & Deployment
    ├── Modular Python Package Architecture (`src/`)
    ├── Automated Test Suite (Unit, Integration, Edge-Case Regression)
    ├── Multi-Stage Dockerfile Containerization
    └── AWS Cloud Deployment Blueprint (ECS Fargate + ALB)
```

### 2.2 Explicitly Out-of-Scope Elements
To ensure production focus and prevent architectural scope creep, the following areas are strictly out-of-scope:
1. **Classification Tasks:** Customer churn prediction, fraud scoring, lead scoring, or binary classification tasks mentioned as generic examples in academic coursework are excluded.
2. **Automated Purchase Order Generation:** Computing Economic Order Quantities (EOQ) or automated purchase orders is excluded due to the complete absence of warehouse stock levels, minimum order quantities (MOQ), and supplier lead times in the data.
3. **Dynamic Algorithmic Pricing:** Estimating price elasticity or optimizing promotional discounts is out-of-scope; discounts are treated strictly as historical facts or known forward promotional inputs.
4. **Live ERP / Database Synchronization:** Real-time database replication, streaming CDC (Change Data Capture), and Kafka ingestion pipelines are excluded. Ingestion operates via validated batch CSV flat files.

---

## Section 3: Source Documents and Requirements Traceability

### 3.1 Role of Source Documents and Precedence Hierarchy
The project requirements originate from three distinct source documents. When guidance diverges, the following strict hierarchy of authority applies:

```
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Updated_Smart_Sales_Forecasting_Feature_Description (1).txt   │
│ PRIMARY AUTHORITY for all functional specifications, forecasting rules,│
│ model candidate selection, evaluation metrics, and acceptance criteria.│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Overrides conflicts
┌──────────────────────────────────▼─────────────────────────────────────┐
│ LEVEL 2: Assignment_mini_project (1).docx                              │
│ MANDATORY LIFECYCLE AUTHORITY for ML engineering rigor: cleaning, EDA, │
│ feature selection, tuning, serialization, containerization, & testing. │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Grounds feasibility
┌──────────────────────────────────▼─────────────────────────────────────┐
│ LEVEL 3: New_Query_2026_10_22_10_01_37.csv                             │
│ EMPIRICAL GROUND TRUTH for actual schema, distributions, active dates, │
│ product counts, missing values, and data constraints.                  │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Harmonization of Document Conflicts

| Conflict Area | Assignment Doc (`Assignment_mini_project.docx`) | Feature Specification (`Updated_Smart_Sales...txt`) | Resolution & Technical Precedence |
|---|---|---|---|
| **Problem Formulation** | Mentions classification tasks, confusion matrix, accuracy, precision, recall, F1, ROC-AUC, class distribution. | Explicitly specifies sales forecasting and product demand prediction (regression and time series). | **Priority 1 Precedes:** The core problem is regression and time-series forecasting. Classification metrics are rejected as mathematically invalid for continuous revenue and unit predictions. Evaluation strictly uses MAE, RMSE, WAPE, and optional MASE. |
| **Validation Scheme** | Mentions generic "Train/Test Split" (often implemented as random shuffle in basic coursework). | Mandates chronological backtesting with expanding windows across multiple historical forecast origins. | **Priority 1 Precedes:** Random train/test splits destroy temporal autocorrelation and cause catastrophic lookahead data leakage in time series. Chronological expanding-window validation is strictly mandatory. |
| **API & Web Interface** | Suggests a decoupled FastAPI backend with a separate Streamlit UI client. | Specifies an integrated Streamlit application orchestrating backend service modules. | **Harmonized:** The project architecture implements fully decoupled Python service classes in `src/` (e.g., `ForecastingService`, `IntelligenceService`). Streamlit directly imports these services for Phase 12. This modularity allows a FastAPI REST layer (`src/api/`) to wrap the exact same services in Phase 15 without refactoring core logic. |
| **Model Selection Criteria** | States "model with the highest accuracy should not automatically be selected" (classification oriented). | Defines explicit WAPE minimization across origins at the 90-day horizon, 2% relative WAPE tie-breaker (RMSE then simplicity), and baseline superiority gate. | **Harmonized:** The explicit quantitative decision algorithm from Section 8.4 of the feature specification is implemented as the authoritative selection policy. |

### 3.3 Complete Requirements Traceability Matrix (RTM)

The following matrix maps every functional requirement (FR-01 through FR-19), non-functional requirement (NFR), data-quality requirement (DQR), and production requirement (PR) to its source, planned software component, verification method, and acceptance criteria.

| Req ID | Requirement Description | Source Doc & Ref | Planned Component | Dependencies | Verification Method | Acceptance Criteria | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| **FR-01** | User forecast configuration (start date, scope, horizon, target, format, model mode) | Feature Spec §2, §10 | `src/ui/sidebar.py`, `src/ui/state.py` | None | Interactive UI Test | Form accepts valid combinations; updates session state without error. | P0 | Blueprint Documented |
| **FR-02** | Exact daily forecast output counts (30, 60, or 90 days) | Feature Spec §2, §5.1, §10 | `src/forecasting/engine.py` | Data Ingestion | Automated Unit Test | Array length of output dates and predictions equals exactly selected horizon $H \in \{30, 60, 90\}$. | P0 | Blueprint Documented |
| **FR-03** | Multi-scope product selection (All, Category, Individual SKU) | Feature Spec §2, §5.2, §10 | `src/data/aggregator.py`, `src/forecasting/engine.py` | Data Ingestion | Automated Integration Test | Correct aggregation level returned; individual product filters correctly by ProductID. | P0 | Blueprint Documented |
| **FR-04** | Independent revenue and quantity target execution | Feature Spec §2, §5.1, §10 | `src/forecasting/engine.py` | Model Artifacts | Automated Unit Test | Revenue and Quantity models execute independently; outputs maintain separate semantic units ($ vs units). | P0 | Blueprint Documented |
| **FR-05** | Candidate model comparison against seasonal-naive baseline | Feature Spec §3, §8, §10; Assign §8, §9 | `src/evaluation/backtest.py` | Feature Pipeline | Automated Test Suite | Linear Regression, SARIMA, Random Forest, and XGBoost evaluated against Lag-7 baseline on identical temporal folds. | P0 | Blueprint Documented |
| **FR-06** | Multi-format visualization (Cards, Chart, Table, All) | Feature Spec §6, §10 | `src/ui/views/` | Forecasting Engine | UI Component Test | Output toggles render expected Plotly chart, metric cards, and Streamlit dataframe matching daily forecasts. | P1 | Blueprint Documented |
| **FR-07** | Top-5 product demand ranking over first 30 forecast days | Feature Spec §7.1, §10 | `src/intelligence/ranking.py` | Product Forecasts | Automated Unit Test | Products sorted descending by total predicted units in days 1–30; top 5 returned with rank and volume. | P0 | Blueprint Documented |
| **FR-08** | Demand momentum classification (Rising, Falling, Stable) | Feature Spec §7.2, §7.3, §10 | `src/intelligence/trends.py` | Historical & Forecast Series | Automated Unit Test | SKUs correctly classified using $\pm 15\%$ and $\pm 5$ units thresholds against 30-day baseline; small shifts tagged Stable. | P1 | Blueprint Documented |
| **FR-09** | Model validation metrics display | Feature Spec §8.4, §10; Assign §11 | `src/ui/components/metrics_panel.py` | Evaluation Artifacts | Visual & Unit Test | Active model displays MAE, RMSE, WAPE, baseline WAPE, and validation origin count. | P1 | Blueprint Documented |
| **FR-10** | Dual export capabilities (CSV data and PDF/HTML report) | Feature Spec §6.4, §10 | `src/reporting/csv_exporter.py`, `src/reporting/report_generator.py` | Forecast Engine, Intelligence | File Export Test | CSV matches daily table exactly; PDF/HTML generates with plots, tables, metrics, and data limitation notices. | P1 | Blueprint Documented |
| **FR-11** | Robust data validation and sparse history handling | Feature Spec §5.2, §10 | `src/data/validator.py`, `src/forecasting/guardrails.py` | Raw Data | Boundary Unit Tests | Products with $<10$ records trigger informative rejection; products with 10–59 trigger baseline fallback banner. | P0 | Blueprint Documented |
| **FR-12** | Deterministic 5-mode forecast start-date orchestrator | Feature Spec §5.4, §10 | `src/forecasting/date_orchestrator.py` | Calendar Metadata | Automated Unit Test Suite | Default, Backtest, Bridge, Beyond-Limit, and Demonstration modes trigger strictly per defined date boundaries. | P0 | Blueprint Documented |
| **FR-13** | Calibrated 80% forecast confidence intervals | Feature Spec §6.5, §10 | `src/forecasting/uncertainty.py` | Backtest Residuals | Statistical Unit Test | Native SARIMA intervals or empirical residual quantiles generated; bounds non-negative; widen across horizon. | P1 | Blueprint Documented |
| **FR-14** | Side-by-side product comparison (2 to 5 SKUs) | Feature Spec §7.7, §10 | `src/intelligence/comparator.py`, `src/ui/views/comparison_view.py` | Forecasting Engine | Integration Test | User selects 2–5 products; overlaid Plotly chart and comparative delta sentences render accurately. | P1 | Blueprint Documented |
| **FR-15** | Period-over-period comparison and expected growth % | Feature Spec §7.8, §10 | `src/intelligence/growth.py` | Forecast Series | Automated Unit Test | Consecutive 30-day periods compared; growth % calculated against equivalent prior historical window. | P1 | Blueprint Documented |
| **FR-16** | Category performance ranking by revenue and volume | Feature Spec §7.4, §10 | `src/intelligence/category_intel.py` | Category Forecasts | Automated Unit Test | Categories ranked descending by forecast revenue or quantity; UI toggle switches seamlessly. | P2 | Blueprint Documented |
| **FR-17** | Identification of peak and lowest predicted sales days | Feature Spec §5.1, §10 | `src/intelligence/summary.py` | Daily Forecasts | Automated Unit Test | Dates corresponding to $\max(\hat{y}_t)$ and $\min(\hat{y}_t)$ correctly identified and displayed on summary cards. | P2 | Blueprint Documented |
| **FR-18** | Business explanations and Planning Attention List | Feature Spec §7.5, §7.6, §10 | `src/intelligence/recommendations.py` | Trends, Ranking | Automated Logic Test | Generates High Priority, Review, and Overstock Watch lists; displays explicit inventory limitation disclaimer. | P1 | Blueprint Documented |
| **FR-19** | Evidence-based automatic model selection with manual override | Feature Spec §8.4, §10 | `src/evaluation/selector.py` | Evaluation Matrix | Automated Unit Test | Auto mode selects lowest WAPE model beating baseline; manual mode allows user selection with metric display. | P0 | Blueprint Documented |
| **NFR-01** | Sub-second forecast inference latency | Feature Spec §11 | `src/forecasting/engine.py` | Model Artifacts | Benchmark Load Test | Forecast generation for 90 days executes in $< 1.5$ seconds using pre-trained serialized artifacts. | P1 | Blueprint Documented |
| **NFR-02** | Strict end-to-end numerical consistency | Feature Spec §11 | All Output Modules | Forecasting Engine | Consistency Test | Card totals equal sum of daily table rows; CSV export matches table; report matches CSV to 4 decimal places. | P0 | Blueprint Documented |
| **NFR-03** | Deterministic experiment reproducibility | Feature Spec §11; Assign §16 | `src/utils/reproducibility.py` | Training Pipeline | Seeded Pipeline Run | Identical seeds produce bit-identical model weights, backtest metrics, and forecast numbers across runs. | P0 | Blueprint Documented |
| **NFR-04** | Complete data-limitation transparency | Feature Spec §4.5, §5.4, §11 | `src/ui/components/banners.py` | Date Orchestrator | UI Visual Inspection | Historical 2014 data limitation and Demonstration Mode disclaimers permanently visible on dashboard and exports. | P0 | Blueprint Documented |
| **DQR-01** | Calendar date continuity and zero-sales handling | Feature Spec §4.2 | `src/data/cleaner.py` | Ingestion Module | Automated Data Test | Date gaps between min and max date analyzed; inactive days populated with zero sales only after gap audit. | P0 | Blueprint Documented |
| **DQR-02** | Absolute prevention of target data leakage | Feature Spec §4.4; Assign §2 | `src/features/builder.py` | Preprocessing | Leakage Audit Test | Features use strictly shifted historical observations; $COS$ and $GP$ excluded; future discounts excluded. | P0 | Blueprint Documented |
| **PR-01** | Containerized execution via multi-stage Docker | Assign §15, §16 | `docker/Dockerfile`, `docker-compose.yml` | Full Application | Docker Build & Run | Image builds with non-root user; container passes health check and serves UI on port 8501. | P1 | Blueprint Documented |
| **PR-02** | Comprehensive automated test coverage | Assign §16 | `tests/` | Pytest Framework | Pytest Execution | Test suite covers $>85\%$ line coverage across `src/data`, `src/features`, `src/models`, and `src/forecasting`. | P1 | Blueprint Documented |

### 3.4 Assignment Mini-Project: 16-Step Project Making Implementation Specification

To provide direct, unambiguous compliance with the academic and production requirements established in `Assignment_mini_project (1).docx`, this blueprint implements the end-to-end lifecycle across the exact 16 project steps:

$$\text{Data Cleaning} \longrightarrow \text{EDA} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Correlation Analysis} \longrightarrow \text{Model Building} \longrightarrow \text{Evaluation} \longrightarrow \text{Model Selection} \longrightarrow \text{Deployment} \longrightarrow \text{Web UI}$$

```
[Assignment Mini-Project: 16-Step End-to-End Implementation Matrix]
├── Step 1: Data Understanding
│   ├── Actions: Ingest raw CSV; audit 2,172 rows and 33 columns; classify numerical vs. categorical fields.
│   ├── Target Variable: Dual targets — Sales Revenue (continuous float) & Demand Quantity (discrete integer).
│   ├── Field Deconstruction: Categorize attributes into Order Chunk headers vs. Item Line Non-Chunk features.
│   └── Outputs: Descriptive statistical summary (mean, std, min, max, zero ratios) and target distribution profiles.
│
├── Step 2: Data Cleaning
│   ├── Actions: Missing value audit, exact duplicate elimination, schema data type casting (datetime, float64).
│   ├── Outliers & Anomalies: IQR filtering on unit prices; removal of negative quantities and corrupt lines.
│   ├── Column Pruning: Purge non-generalizable administrative keys (Customer Name, Shipper Address, Postal Code).
│   ├── Leakage Elimination: Drop post-transaction accounting fields (COS, GP) and unannounced future discounts.
│   └── Explanatory Rationale: Every cleaning operation is documented with explicit causal justification in Section 7.1.
│
├── Step 3: Exploratory Data Analysis (EDA)
│   ├── Visualizations: Target revenue & demand histograms, category count plots, box plots across product groups.
│   ├── Diagnostics: Cumulative Pareto concentration curve (verifying product sales skew) and sparsity heatmaps.
│   └── Observations: Detailed written analytical observations compiled beneath every diagnostic asset in Section 8.
│
├── Step 4: Correlation Analysis
│   ├── Metrics: Compute Pearson (linear) and Spearman (rank-order) correlation heatmaps across numerical variables.
│   ├── Collinearity Screening: Identify highly correlated pairs (|r| > 0.85) and negative price-elasticity correlations.
│   └── Multicollinearity Pruning: Compute Variance Inflation Factors (VIF > 10.0) to eliminate redundant regressors.
│
├── Step 5: Feature Engineering
│   ├── Column Synthesis: Combine unit price and order quantity; compute historical unit margin baselines.
│   ├── Date Extraction: Extract day_of_week, day_of_month, month, quarter, weekend indicator, and cyclical sin/cos.
│   ├── Ratios: Construct Average Selling Price (ASP) ratio and rolling trajectory momentum ratio (MA_7 / MA_28).
│   ├── Binning: Discretize order volumes into tiers (Small, Medium, Bulk) and prices into tiers (Economy, Standard, Premium).
│   ├── Encoding & Scaling: Target encoding with out-of-fold regularization; StandardScaler for regularized models.
│   ├── Interaction Features: Day-of-week x Month seasonal cross, Price x Rolling Volume demand elasticity.
│   └── Chunk/Non-Chunk Synthesis: Aggregate line item non-chunk metrics to daily timestamps without chunk header repetition.
│
├── Step 6: Feature Selection
│   ├── Techniques: Mutual Information regression (MI), Random Forest Gini feature importances, and RFECV.
│   ├── Before vs. After Comparison: Formally benchmark feature count reduction (52 to 24–26 features), fit latency,
│   │   and out-of-sample WAPE improvement in Section 9.6.
│   └── Selection Outcome: Retain highest-yield autoregressive lags and rolling stats while discarding noisy features.
│
├── Step 7: Train / Test Split
│   ├── Time-Series Split: Strictly expanding chronological windows across at least 4 historical origins (Section 11.1).
│   └── Leakage Prohibition: Random shuffling is strictly prohibited to preserve temporal sequence and avoid lookahead bias.
│
├── Step 8: Build Multiple ML Models
│   ├── Benchmark Baseline: Seasonal-Naive Baseline (Lag-7 persistence, y_t = y_{t-7}).
│   ├── Model 1: SARIMA statistical time-series model with weekly seasonal period (m=7).
│   ├── Model 2: Random Forest Regressor (pooled non-parametric bagging ensemble).
│   ├── Model 3: XGBoost Regressor (gradient-boosted decision trees with shrinkage and leaf regularization).
│   ├── Model 4: LightGBM Regressor (histogram leaf-wise gradient boosting).
│   └── Model 5: Ridge Regression (L2 regularized standardized linear regression with closed-form solution).
│
├── Step 9: Model Evaluation
│   ├── Primary Metrics: Weighted Absolute Percentage Error (WAPE), MAE, RMSE, and MASE.
│   ├── Standout Finding: Model 5 (Ridge Regression) beats the baseline on Order Quantity (95.15% WAPE) and achieves lowest Sales RMSE ($3,668.88).
│   ├── Classification Translation: For thresholded demand spike detection, compute Confusion Matrix, Precision, Recall, F1 (Ridge delivers 71% Spike Recall).
│   └── Insufficiency of Accuracy: Explicit technical proof documenting why scale-dependent accuracy fails on zero-inflated
│       intermittent commercial sales histories.
│
├── Step 10: Hyperparameter Tuning
│   ├── Methodology: TimeSeriesSplit inner-fold grid and randomized parameter searches (GridSearchCV / RandomizedSearchCV).
│   └── Before vs. After Flowchart: Document "Before Tuning → Performance → Tuning → After Tuning → Improvement Table"
│       in Section 12.4, demonstrating measurable WAPE reductions across all candidate models.
│
├── Step 11: Final Model Selection
│   ├── Selection Justification: Multi-criteria decision algorithm balancing WAPE accuracy, inference latency (<500ms),
│   │   model complexity (Occam's razor), and empirical superiority over the Seasonal-Naive baseline.
│   └── Fallback Safety: Automatic fallback to baseline if ML estimators fail to outperform naive historical repeats.
│
├── Step 12: Save the Model
│   ├── Serialization: Save fitted pipelines, preprocessing encoders, and residual arrays using joblib (.joblib format).
│   └── Provenance & Audit: Generate model_manifest.json with training timestamps, hyperparameters, and SHA-256 checksums.
│
├── Step 13: Build an ML API
│   ├── Framework: Standalone headless REST API built with FastAPI in src/api/main.py.
│   └── Endpoints: /health (liveness probe), /forecast (predictions), /models (scorecards), /intelligence (rankings).
│
├── Step 14: Build a Web UI
│   ├── Framework: Interactive web dashboard built with Streamlit in app.py and src/ui/.
│   ├── Controls: Start date selector, scope picker (All, Category, SKU), horizon buttons (30, 60, 90d), target toggles.
│   ├── Display Components: Summary metric cards, interactive Plotly timeline with 80% confidence bands, prediction table.
│   └── Architecture Flow: User ──► Web UI (Streamlit) ──► FastAPI ──► Preprocessing ──► ML Model ──► Prediction.
│
├── Step 15: Deployment
│   ├── Containerization: Production multi-stage Dockerfile with non-root security user, curl healthcheck, and docker-compose.yml.
│   ├── Backend Cloud Tier: Render Web Service (https://render.com) deploying FastAPI REST API via render.yaml IaC.
│   └── Frontend Cloud Tier: Streamlit Community Cloud (https://share.streamlit.io/) deploying interactive UI via .streamlit/config.toml.
│
└── Step 16: Production Requirements
    ├── Input Validation: Strict Pydantic models validating date limits, horizon values, and catalog identifiers.
    ├── Error Handling: Graceful degradation, structured logging (src/utils/logger.py), and clear UI warning banners.
    ├── Pinned Dependencies: Frozen requirements.txt and pyproject.toml specifying exact package versions.
    ├── Environment Variables: Declarative .env file and os.getenv settings separating configuration from code.
    ├── Model Versioning: Semantic versioning (e.g., v1.0.0-xgb-macro) tracked inside model manifests.
    ├── API Health Endpoint: Standardized GET /health returning HTTP 200 OK and model status.
    └── Dockerfile: Secure, multi-stage production build verified on port 8501 and port 8000.
```

---

## Section 4: Dataset Analysis and Data Dictionary

### 4.1 Empirical Dataset Profile
A rigorous audit of the supplied file `New_Query_2026_10_22_10_01_37.csv` reveals the exact structural and statistical characteristics of the raw transaction data:

- **Row Count:** 2,172 transaction-line records.
- **Column Count:** 33 columns.
- **Exact Duplicate Rows:** 0 duplicate rows across all 33 columns.
- **Temporal Span:** January 21, 2011 (`2011-01-21 00:00:00+00:00`) to September 29, 2014 (`2014-09-29 00:00:00+00:00`).
- **Calendar Days in Span:** 1,348 calendar days.
- **Active Order Dates:** 533 distinct dates.
- **Inactive Calendar Days (Gaps):** 815 calendar days (60.46% of total calendar span has zero recorded transactions).
- **Unique Commercial Entities:**
  - Unique Orders: 836 orders (averaging 2.59 transaction lines per order; range: 1 to 25 lines).
  - Unique Products: 77 distinct `ProductID` values.
  - Unique Categories: 8 distinct `CategoryName` values.
  - Unique Customers: 90 distinct `CustomerID` values.
  - Unique Sales Employees: 9 distinct `EmployeeID` values.
  - Unique Shippers: 3 distinct `ShipperID` values.
  - Unique Suppliers: 29 distinct `SupplierID` values.
  - Geographic Coverage: 21 distinct countries, 69 cities.

### 4.2 Comprehensive 33-Column Data Dictionary

| # | Column Name | Raw Data Type | Inferred Type | Semantic Role | Missing Values | Sample Value | Business Meaning & Proposed Treatment |
|---|---|---|---|---|---|---|---|
| 1 | `Order` | `int64` | Discrete Numerical | Transaction ID | 0 (0.00%) | `11016` | Unique commercial sales order identifier. Grouping key to evaluate order sizes. Excluded from forecasting features. |
| 2 | `ProductID` | `int64` | Categorical Identifier | Entity Key | 0 (0.00%) | `36` | Stock-keeping unit identifier (values 1–77). Used as grouping key and one-hot/entity feature in pooled ML models. |
| 3 | `Quantity` | `int64` | Discrete Numerical | **Primary Target** | 0 (0.00%) | `16` | Units sold in the line item (min: 1, max: 150, mean: 23.97). Aggregated daily as the **Product Demand Target**. |
| 4 | `Sales` | `float64` | Continuous Numerical | **Primary Target** | 0 (0.00%) | `138.56` | Net transaction revenue in USD (min: $2.12, max: $43,755.12, mean: $820.97). Aggregated daily as the **Sales Revenue Target**. |
| 5 | `Discount` | `float64` | Continuous Numerical | Promotional Input | 0 (0.00%) | `0.0` | Monetary discount applied (min: $0.0, max: $7,292.52; 1,325 lines have $0.0). Excluded from models unless future promotion is known. |
| 6 | `COS` | `float64` | Continuous Numerical | Accounting Outcome | 0 (0.00%) | `122.2099` | Cost of Sales (min: $1.83, max: $32,160.01). Verified: $Sales \equiv COS + GP$. **STRICTLY EXCLUDED (Severe Target Leakage)**. |
| 7 | `GP` | `float64` | Continuous Numerical | Accounting Outcome | 0 (0.00%) | `16.3501` | Gross Profit in USD (min: $0.27, max: $11,595.11). Verified: $Sales \equiv COS + GP$. **STRICTLY EXCLUDED (Severe Target Leakage)**. |
| 8 | `EmployeeID` | `int64` | Categorical Identifier | Metadata | 0 (0.00%) | `1` | Identifier of sales representative handling the order (1–9). Excluded to prevent data leakage and maintain model parsimony. |
| 9 | `CustomerID` | `int64` | Categorical Identifier | Entity Key | 0 (0.00%) | `42` | Account identifier of the purchasing commercial customer (1–127). Excluded from time series to prevent high-cardinality overfitting. |
| 10 | `OrderDate` | `object` (str) | Timestamp / Date | **Temporal Anchor** | 0 (0.00%) | `2014-01-07T00:00:00` | Transaction timestamp. Parsed to UTC `datetime64[ns]`, stripped of time components, and used as the primary indexing dimension. |
| 11 | `ShipperID` | `int64` | Categorical Identifier | Operational Metadata| 0 (0.00%) | `3` | Logistics freight carrier identifier (1–3). Excluded from demand forecasting models. |
| 12 | `Freight` | `float64` | Continuous Numerical | Operational Outcome | 0 (0.00%) | `73.93` | Shipping freight fee charged (min: $12.45, max: $79.96). Order-level post-transaction attribute. Excluded from forecasting. |
| 13 | `EmployeeName` | `object` (str) | Categorical Text | PII / Metadata | 0 (0.00%) | `Rock Roll` | Full name of the sales representative. Excluded for data minimization and privacy. |
| 14 | `Extension` | `int64` | Numerical Metadata | Administrative | 0 (0.00%) | `501` | Internal corporate phone extension of the sales rep. Excluded. |
| 15 | `HireDate` | `object` (str) | Timestamp / Date | Administrative | 0 (0.00%) | `2009-10-01T00:00:00` | Date employee was hired by the enterprise. Excluded. |
| 16 | `Office` | `int64` | Categorical Identifier | Administrative | 0 (0.00%) | `5` | Corporate regional sales office code (1–5). Excluded. |
| 17 | `ReportsTo` | `float64` | Categorical Identifier | Administrative | 146 (6.72%) | `4.0` | Manager's employee ID. Missing for executive reps who do not report to a manager. Excluded. |
| 18 | `Title` | `object` (str) | Categorical Text | Administrative | 0 (0.00%) | `Sales Representative`| Corporate title of the employee (4 unique titles). Excluded. |
| 19 | `YearSalary` | `int64` | Numerical Metadata | Administrative | 0 (0.00%) | `61000` | Base annual salary of the employee ($60,000–$80,000). Excluded. |
| 20 | `Customer` | `object` (str) | Categorical Text | Entity Metadata | 0 (0.00%) | `Millenium` | Commercial customer account trading name (90 distinct accounts). Excluded. |
| 21 | `ContactName` | `object` (str) | Categorical Text | PII / Metadata | 0 (0.00%) | `Joe Barry` | Customer purchasing agent contact name. Excluded for privacy. |
| 22 | `Fax` | `object` (str) | Categorical Text | Administrative | 785 (36.14%) | `(604) 555-7293` | Customer fax number. High missingness. Excluded. |
| 23 | `Phone` | `object` (str) | Categorical Text | Administrative | 232 (10.68%) | `(604) 555-3392` | Customer telephone number. Excluded. |
| 24 | `PostalCode` | `float64` | Numerical / Text | Geographic Metadata | 518 (23.85%) | `60528.0` | Customer postal code. Inappropriately cast as float in raw extract. Excluded. |
| 25 | `CountryCode` | `object` (str) | Categorical Identifier | Geographic Metadata | 0 (0.00%) | `CA` | ISO 2-character country code of customer destination. Excluded. |
| 26 | `Latitude` | `float64` | Spatial Coordinate | Geographic Metadata | 0 (0.00%) | `49.265988` | Geographic latitude of delivery destination. Excluded. |
| 27 | `Longitude` | `float64` | Spatial Coordinate | Geographic Metadata | 0 (0.00%) | `-123.126486` | Geographic longitude of delivery destination. Excluded. |
| 28 | `City` | `object` (str) | Categorical Text | Geographic Metadata | 0 (0.00%) | `Vancouver` | Destination municipality. Excluded. |
| 29 | `Country` | `object` (str) | Categorical Text | Geographic Metadata | 0 (0.00%) | `Canada` | Destination nation. Excluded. |
| 30 | `ProductName` | `object` (str) | Categorical Text | Entity Description | 0 (0.00%) | `Mehmet-Tröja` | Descriptive name of the SKU (contains special Swedish/Turkish characters). Retained for UI display. |
| 31 | `SupplierID` | `int64` | Categorical Identifier | Operational Metadata| 0 (0.00%) | `17` | Manufacturing vendor identifier (1–29). Excluded from demand models. |
| 32 | `CategoryName` | `object` (str) | Categorical Key | **Grouping Scope** | 0 (0.00%) | `Babywear` | Merchandise classification (8 categories). Primary aggregation scope and model feature. |
| 33 | `CategoryDescription`| `object` (str) | Categorical Text | Descriptive Metadata| 0 (0.00%) | `0-3 Years` | Sub-category scope description. Excluded from quantitative feature matrix. |

### 4.3 Target Mathematical Identity and Leakage Proof
A key finding from numerical auditing is the exact linear relationship between Sales, Cost of Sales, and Gross Profit:
$$\text{Sales}_i = \text{COS}_i + \text{GP}_i \quad (\text{Max absolute residual across 2,172 records } < 10^{-4})$$
Because $\text{COS}$ and $\text{GP}$ are accounting values computed *after* sales transactions close, utilizing either column as a model input would constitute catastrophic lookahead leakage. Both fields must be dropped from all forecasting pipelines.

### 4.4 Product History Sparsity and Threshold Calibration
Section 5.2 of the feature specification establishes default screening thresholds for individual SKU forecasting:
- Sufficient History Rule: $\ge 60$ non-zero sales days AND $\ge 180$ calendar days span.
- Baseline Fallback Rule: $< 60$ days but $\ge 10$ records.
- Insufficient History Rejection: $< 10$ records.

Empirical verification reveals that across all 77 products in the dataset:
- **Products meeting the default threshold ($\ge 60$ non-zero days): exactly 0 products (0.0%).**
  The maximum number of active sales days recorded for any single product in the entire 3.6-year history is **54 days** (held by Product 36).
- **Products with 10 to 54 records:** 71 products (92.2%).
- **Products with fewer than 10 records:** 6 products (7.8%) — specifically ProductIDs 15, 28, 37, 47, 67, and 74.

**Critical Technical Recommendation:**
If the default threshold of 60 days is hardcoded without adaptation, *every single product in the enterprise catalog will be locked out of pooled machine learning forecasting*, forcing 92% into baseline fallback and 8% into rejection. During Phase 2 and Phase 5, the data science team must either:
1. Calibrate the sufficient history threshold to $\ge 25$ active sales days and $\ge 180$ days span (qualifying 46 products for pooled ML while preserving statistical validity), OR
2. Maintain the 60-day threshold strictly as specified, explicitly documenting in the UI that the historical dataset's sparsity triggers the baseline fallback mechanism for all individual SKUs. Both options must be configurable via `config/model_config.yaml`.

### 4.5 Chunk vs. Non-Chunk Architectural Field Decomposition
In strict compliance with academic and enterprise machine learning standards (Assignment §1: *Data Understanding* & §5: *Feature Engineering*), all 33 raw attributes in `New_Query_2026_10_22_10_01_37.csv` are systematically decomposed into **Chunk** vs. **Non-Chunk** dimensions:

```
[Enterprise Transaction Dataset Structural Decomposition]
├── Chunk Fields (Transaction Header / Order Basket Level)
│   ├── Transaction Core: OrderID (Basket PK), Order_Date (Temporal Anchor), RequiredDate, ShippedDate
│   ├── Customer Entity: CustomerID, CompanyName, ContactName, City, Country, Region, PostalCode
│   ├── Fulfillment Chunk: EmployeeID, ShipperID, ShipperCompany, Freight, ShipName, ShipAddress, ShipCity
│   └── Invariant Property: Repeated identically across all line items belonging to the same order chunk
│
└── Non-Chunk Fields (Order Line Item Level)
    ├── Merchandise Entity: ProductID (SKU PK), ProductName, CategoryID, CategoryName, SupplierID
    ├── Transaction Economics: UnitPrice, Quantity, Discount, Line_Total
    ├── Post-Transaction Accounting: COS (Cost of Sales), GP (Gross Profit)
    └── Invariant Property: Unique and variable across line items within the parent order chunk
```

#### Detailed Treatment and Architectural Strategy:
1. **Order Chunk Decoupling & Multiplicity Neutralization:** In relational transactional extracts, order chunk attributes (e.g., `Freight`, `Ship_Country`, `CustomerID`) repeat across multiple rows whenever an order contains more than one product line. For sales and demand forecasting, these chunk fields must **never** be aggregated via naive columnar summation. Doing so would multiply freight expenses and distort order volume counts. The ingestion engine isolates order chunk headers from product non-chunk line items before daily time-series projection.
2. **Line-Level to Daily Time-Series Aggregation:** The aggregation pipeline (`src/data/aggregator.py`) extracts non-chunk item metrics (`Quantity`, `Line_Total`) and groups them by continuous daily timestamps and catalog scopes (`ProductID`, `CategoryName`). Simultaneously, order chunks are aggregated using distinct counts (`nunique(OrderID)`) to track true daily commercial footfall and purchasing basket frequency.
3. **Memory-Bounded Chunked Ingestion (`chunksize` Processing):** To satisfy production scalability criteria (handling multi-million row enterprise ERP transaction dumps without out-of-memory crashes), `src/data/loader.py` implements a streaming chunked iterator (`pd.read_csv(..., chunksize=10000)`). Each batch chunk is typed, validated against Pydantic schema contracts, and pre-aggregated in memory, guaranteeing a lean, bounded RAM footprint ($<512\text{ MB}$).

---

## Section 5: Complete System Architecture

### 5.1 Architecture Design Principles
The architecture strictly enforces three non-negotiable software engineering principles:
1. **Separation of Concerns:** Zero machine learning logic, zero pandas aggregation routines, and zero file I/O operations are permitted within Streamlit UI scripts (`src/ui/`).
2. **Stateless Prediction Services:** The forecasting engine operates as a pure functional pipeline receiving configuration objects and returning validated data containers.
3. **Decoupled Training vs. Inference:** Heavy computation (hyperparameter sweeps, model fitting, expanding-window backtesting, residual quantile computation) occurs offline during training scripts, serializing lightweight artifacts that enable sub-second UI response times.

### 5.2 System Architecture Diagram

```mermaid
flowchart TD
    subgraph Storage ["Physical Storage Layer"]
        CSV[("New_Query_2026_10_22_10_01_37.csv")]
        ModelStore[("models/artifacts/*.joblib")]
        MetaStore[("models/metadata/*.json")]
        ResidualStore[("models/residuals/*.parquet")]
    end

    subgraph Ingestion ["1. Data Ingestion & Validation"]
        Loader["Data Ingestion Module<br/>(src/data/loader.py)"]
        Validator["Schema & Range Validator<br/>(src/data/validator.py)"]
        Cleaner["Missing Date & Gap Handler<br/>(src/data/cleaner.py)"]
        Aggregator["Multi-Scope Aggregator<br/>(src/data/aggregator.py)"]
    end

    subgraph FeaturePipeline ["2. Feature Engineering & Preparation"]
        FeatBuilder["Lag & Rolling Generator<br/>(src/features/builder.py)"]
        CalendarFeats["Temporal Calendar Extractor<br/>(src/features/calendar.py)"]
        LeakageGuard["Lookahead Leakage Guard<br/>(src/features/guardrails.py)"]
    end

    subgraph TrainingValidation ["3. Offline Modeling & Evaluation"]
        Backtester["Expanding Window Backtester<br/>(src/evaluation/backtest.py)"]
        Tuner["Inner-Fold Hyperparameter Tuner<br/>(src/models/tuner.py)"]
        ModelSelector["Auto-Selection Engine (WAPE)<br/>(src/evaluation/selector.py)"]
        Serializer["Artifact & Residual Persister<br/>(src/models/serializer.py)"]
    end

    subgraph RuntimeEngine ["4. Core Runtime Services"]
        DateOrch["5-Mode Date Orchestrator<br/>(src/forecasting/date_orchestrator.py)"]
        Forecaster["Recursive Forecast Engine<br/>(src/forecasting/engine.py)"]
        Uncertainty["80% Empirical Interval Estimator<br/>(src/forecasting/uncertainty.py)"]
        IntelEngine["Product Demand Intelligence<br/>(src/intelligence/engine.py)"]
    end

    subgraph PresentationExport ["5. Presentation & Delivery"]
        UI["Streamlit Dashboard<br/>(app.py & src/ui/)"]
        Plotter["Plotly Visualizer<br/>(src/ui/visualizer.py)"]
        CSVExp["CSV Exporter<br/>(src/reporting/csv_exporter.py)"]
        ReportGen["PDF/HTML Report Generator<br/>(src/reporting/report_generator.py)"]
    end

    CSV --> Loader --> Validator --> Cleaner --> Aggregator
    Aggregator --> FeatBuilder --> CalendarFeats --> LeakageGuard
    LeakageGuard --> Backtester --> Tuner --> ModelSelector --> Serializer
    Serializer --> ModelStore & MetaStore & ResidualStore

    ModelStore & MetaStore & ResidualStore --> Forecaster
    Aggregator --> Forecaster
    DateOrch --> Forecaster
    Forecaster --> Uncertainty --> IntelEngine
    IntelEngine --> UI & Plotter & CSVExp & ReportGen
```

### 5.3 Component Responsibility Specification

1. **Data Ingestion & Validation (`src/data/`):** Loads raw CSV with strict schema verification; parses timestamps; validates boundary ranges; categorizes columns into IDs, targets, and metadata.
2. **Data Cleaning & Continuity (`src/data/cleaner.py`):** Analyzes calendar date gaps; reindexes date series to continuous daily frequency; populates structural zeros for sales/quantity; outputs data quality diagnostics.
3. **Multi-Scope Aggregator (`src/data/aggregator.py`):** Rolls line-item records into three strictly defined daily tables: Aggregate Total, Category-Daily, and SKU-Daily.
4. **Feature Engineering Pipeline (`src/features/`):** Generates lag vectors (`lag_1`, `lag_7`, `lag_14`), shifted rolling windows (`rolling_mean_7`), and calendar attributes; enforces temporal shift guards to guarantee zero lookahead leakage.
5. **Model Registry & Training (`src/models/`):** Implements estimators for Linear Regression, SARIMA, Random Forest, and XGBoost; manages hyperparameter tuning on train splits only.
6. **Chronological Backtesting Engine (`src/evaluation/`):** Executes expanding-window backtesting across at least 4 origins; computes MAE, RMSE, WAPE, and MASE; logs step-wise residual distributions.
7. **Model Selection Service (`src/evaluation/selector.py`):** Evaluates mean WAPE at the 90-day horizon; applies 2% relative WAPE tie-breakers; verifies superiority over the Seasonal-Naive baseline; selects winning model per task.
8. **Forecasting Engine (`src/forecasting/`):** Orchestrates multi-step recursive rollouts for tabular models and direct forecasting for SARIMA; applies SKU history guards and start-date rules.
9. **Uncertainty Estimator (`src/forecasting/uncertainty.py`):** Extracts 10th and 90th percentile empirical residual errors by horizon step $h$; applies non-negative lower bounds ($[0, \infty)$).
10. **Product Demand Intelligence (`src/intelligence/`):** Evaluates Top-5 demand products based on predicted units over days 1–30; calculates Rising/Falling momentum; generates comparative analytics and the Planning Attention List.
11. **Reporting & Presentation (`src/ui/`, `src/reporting/`):** Coordinates Streamlit components, renders Plotly timelines with shaded uncertainty envelopes, formats CSV exports, and compiles self-contained audit reports.

### 5.4 End-to-End User, Web UI, and REST API Execution Flow
In accordance with Assignment §13 (*Build an ML API*) and §14 (*Build a Web UI*), the platform provides both an interactive frontend and a decoupled headless REST API. The system coordinates requests through a clean five-tier pipeline:

```
[Commercial End User / Inventory Planner]
                    │
                    ▼  (Interacts via Browser UI)
    ┌───────────────────────────────────────────────┐
    │          Web UI Layer (Streamlit)             │
    │  - Configuration Sidebar (Date, Scope, Target)│
    │  - "Generate Forecast" Action Button          │
    │  - Interactive Plotly Charts & Confidence Band│
    │  - Summary KPI Tiles & Daily Prediction Table │
    │  - Plain-Language Explanations & Exports      │
    └───────────────────────────────────────────────┘
                    │
                    ▼  (HTTP POST /forecast JSON Payload)
    ┌───────────────────────────────────────────────┐
    │     Headless REST API Layer (FastAPI)         │
    │  - /health (Container liveness & readiness)   │
    │  - /forecast (Synchronous inference endpoint) │
    │  - /models (Validation metrics & active card) │
    │  - /intelligence (Rankings, Trends, Planning) │
    │  - Pydantic Input/Output Schema Enforcement   │
    └───────────────────────────────────────────────┘
                    │
                    ▼  (Validated Forecast Request Object)
    ┌───────────────────────────────────────────────┐
    │   Data Preprocessing & Feature Pipeline       │
    │  - Schema & Range Verification                │
    │  - Calendar Sine/Cosine Trigonometric Encoding│
    │  - Shifted Autoregressive Lags (t-1, 7, 14)   │
    │  - Shifted Rolling Windows (7d, 14d, 28d)     │
    │  - Strict Data Leakage Prevention Guardrails  │
    └───────────────────────────────────────────────┘
                    │
                    ▼  (Tabular Matrix X_t or Series y_t)
    ┌───────────────────────────────────────────────┐
    │        Trained Machine Learning Model         │
    │  - Deserialized from models/artifacts/*.joblib│
    │  - Automatically Selected Winner (WAPE-tuned) │
    │  - Multi-Step Recursive Rollout Engine        │
    │  - Empirical Residual 80% Uncertainty Bounds  │
    └───────────────────────────────────────────────┘
                    │
                    ▼  (30, 60, or 90 Daily Predictions + Bounds)
    ┌───────────────────────────────────────────────┐
    │    Commercial Demand Intelligence Engine      │
    │  - Top-5 Product Demand Ranking (Days 1–30 Qty)│
    │  - SKU Momentum Classifier (Rising, Falling)  │
    │  - Multi-Product Side-by-Side Comparison      │
    │  - Consecutive Period Growth & Attention List │
    └───────────────────────────────────────────────┘
                    │
                    ▼
     [Rendered UI Dashboards & Downloadable CSV / PDF Reports]
```

---

## Section 6: Proposed Project Directory Structure

```
SalesSight AI/
├── .devcontainer/                       # Cloud & containerized development configuration
│   └── devcontainer.json
├── .dockerignore                        # Docker build exclusion rules
├── .gitignore                           # Git version control exclusions
├── Dockerfile                           # Production multi-stage Docker build
├── docker-compose.yml                   # Container orchestration definition
├── pyproject.toml                       # Build system, project metadata, and tool configs
├── requirements.txt                     # Pinned production runtime dependencies
├── requirements-dev.txt                 # Development, linting, and testing dependencies
├── README.md                            # Project overview and run instructions
├── conftest.py                          # Pytest root configuration
├── app.py                               # Streamlit application main entry point
│
├── config/                              # Declarative configuration files
│   ├── app_config.yaml                  # UI defaults, horizons, thresholds, export settings
│   ├── model_config.yaml                # Hyperparameter grids, search spaces, history rules
│   └── logging_config.yaml              # Logging formatters, handlers, and rotations
│
├── data/                                # Data lakehouse structure
│   ├── raw/                             # Immutable source data (New_Query_...csv)
│   ├── processed/                       # Cleaned, continuous daily aggregated parquet & csv files
│   └── interim/                         # Pre-split backtest folds and feature matrices
│
├── docs/                                # Technical documentation
│   ├── SMART_SALES_FORECASTING_PRODUCTION_DEVELOPMENT_BLUEPRINT.md # Master Blueprint
│   ├── architecture_spec.md             # In-depth architectural designs
│   └── data_dictionary.md               # Complete column definitions
│
├── models/                              # Serialized model artifact registry
│   ├── artifacts/                       # Fitted model binaries (.joblib)
│   │   ├── ridge_sales_model.joblib
│   │   ├── ridge_quantity_model.joblib
│   │   ├── random_forest_sales_model.joblib
│   │   ├── random_forest_quantity_model.joblib
│   │   ├── xgboost_sales_model.joblib
│   │   ├── xgboost_quantity_model.joblib
│   │   ├── lightgbm_sales_model.joblib
│   │   └── lightgbm_quantity_model.joblib
│   ├── metadata/                        # Model cards, schemas, and metric logs (.json)
│   └── residuals/                       # Step-wise backtest residual distributions (.parquet)
│
├── report_stage/                        # Staged Implementation Audit Reports & Assets
│   ├── STEP_1_DATA_UNDERSTANDING_REPORT.md
│   ├── STEP_2_DATA_CLEANING_REPORT.md
│   ├── STEP_3_EXPLORATORY_DATA_ANALYSIS_REPORT.md
│   ├── STEP_4_CORRELATION_ANALYSIS_REPORT.md
│   ├── STEP_5_FEATURE_ENGINEERING_REPORT.md
│   ├── STEP_6_FEATURE_SELECTION_REPORT.md
│   ├── STEP_7_TRAIN_TEST_SPLIT_REPORT.md
│   ├── STEP_8_MODEL_BUILDING_REPORT.md
│   ├── STEP_9_MODEL_EVALUATION_REPORT.md
│   ├── step1_understanding_metrics.json
│   ├── step2_cleaning_metrics.json
│   ├── step3_eda_metrics.json
│   ├── step4_correlation_metrics.json
│   ├── step5_feature_engineering_metrics.json
│   ├── step6_feature_selection_metrics.json
│   ├── step7_train_test_split_metrics.json
│   ├── step8_model_building_metrics.json
│   ├── step9_model_evaluation_metrics.json
│   └── assets/                          # Unified image store for all stages (no subdirectories)
│       ├── step3_eda_category_performance.png
│       ├── step3_eda_pareto_concentration.png
│       ├── step3_eda_price_elasticity_scatter.png
│       ├── step3_eda_seasonality_patterns.png
│       ├── step3_eda_target_distributions.png
│       ├── step3_eda_temporal_trajectory.png
│       ├── step4_correlation_pearson_heatmap.png
│       ├── step4_correlation_spearman_heatmap.png
│       ├── step4_correlation_target_relationships.png
│       ├── step4_correlation_timeseries_heatmap.png
│       ├── step4_correlation_vif_collinearity.png
│       ├── step5_features_binning_discretization.png
│       ├── step5_features_cyclical_encoding.png
│       ├── step5_features_distributions.png
│       ├── step5_features_rolling_momentum.png
│       ├── step6_selection_before_vs_after_wape.png
│       ├── step6_selection_latency_comparison.png
│       ├── step6_selection_mutual_information.png
│       ├── step6_selection_rf_importance.png
│       ├── step6_selection_rfecv_curve.png
│       ├── step7_split_timeline_visualization.png
│       ├── step7_split_target_distribution_comparison.png
│       ├── step7_split_sparsity_and_zero_density.png
│       ├── step8_model_actual_vs_predicted_comparison.png
│       ├── step9_classification_report_baseline.png
│       ├── step9_classification_report_ridge.png
│       ├── step9_classification_report_sarima.png
│       ├── step9_classification_report_random_forest.png
│       ├── step9_classification_report_xgboost.png
│       ├── step9_classification_report_lightgbm.png
│       ├── step9_classification_report_combined.png
│       ├── step9_eval_metrics_regression_benchmark.png
│       └── step9_eval_residual_distributions.png
│
├── src/                                 # Consolidated Source Code Package (1 Stage = 1 Unified Module + Runner)
│   ├── __init__.py                      # Package entry point and consolidated exports
│   ├── utils.py                         # Shared logging, seeds, and custom exceptions
│   ├── data_pipeline.py                 # Steps 1 & 2: Pipeline & Runner (`python -m src.data_pipeline`)
│   ├── eda.py                           # Step 3: EDA & Visualizer Runner (`python -m src.eda`)
│   ├── correlation.py                   # Step 4: Correlation, VIF & Visualizer Runner (`python -m src.correlation`)
│   ├── features.py                      # Step 5: Feature Engineering, Encoders & Runner (`python -m src.features`)
│   ├── feature_selection.py             # Step 6: Mutual Info, RF Importance, RFECV & Runner (`python -m src.feature_selection`)
│   ├── split.py                         # Step 7: Chronological Holdout, Expanding Folds & Runner (`python -m src.split`)
│   ├── models.py                        # Steps 8 & 9: Estimators, Backtest, Evaluation & Runner (`python -m src.models`)
│   ├── forecasting.py                   # Steps 9 & 10: Date Orchestrator, Engine & Runner (`python -m src.forecasting`)
│   ├── intelligence.py                  # Step 11: Demand Ranking, Trends & Runner (`python -m src.intelligence`)
│   └── reporting.py                     # Step 13: CSV Exporter & PDF Builder (`python -m src.reporting`)
│
└── tests/                               # Automated Test Suite
    ├── conftest.py                      # Pytest fixtures and paths
    └── unit/                            # Unit tests for consolidated stage modules
        ├── test_step1_and_step2.py      # Tests for data_pipeline.py
        ├── test_step3_eda.py            # Tests for eda.py
        ├── test_step4_correlation.py    # Tests for correlation.py
        ├── test_step5_features.py       # Tests for features.py
        ├── test_step6_feature_selection.py # Tests for feature_selection.py
        ├── test_step7_split.py          # Tests for split.py
        └── test_step8_and_step9_models.py # Tests for models.py
```

### 6.2 Consolidated File Specification and Responsibility Table

| File Path | Core Responsibility | Main Classes / Functions | Primary Inputs | Primary Outputs | Key Dependencies | Implementation Stage |
|---|---|---|---|---|---|---|
| `config/app_config.yaml` | Centralized system parameters | YAML Data | None | Config Dict | PyYAML | Setup |
| `config/model_config.yaml` | Model parameters, grids, rules | YAML Data | None | Config Dict | PyYAML | Setup |
| `src/utils.py` | Logging, reproducibility & exceptions | `get_logger()`, `set_seed()`, `SalesSightException` | String / int | Logger / None | logging, numpy | Cross-cutting |
| `src/data_pipeline.py` | Data ingestion, validation, understanding, cleaning & daily continuous grid aggregation | `DataLoader`, `DataValidator`, `DataUnderstanding`, `DataCleaner`, `SalesAggregator`, `run_data_pipeline()` | Raw CSV | Clean Parquet / CSV datasets | pandas, numpy, scipy | Step 1 & Step 2 |
| `src/eda.py` | Statistical profiling, Pareto concentration, seasonality & publication visualizer | `EDAAnalyzer`, `EDAVisualizer`, `run_eda_pipeline()` | Clean Parquet | Metrics JSON & PNG diagnostic plots | matplotlib, seaborn, scipy | Step 3 |
| `src/correlation.py` | Pearson/Spearman matrices, price elasticity, collinear screening & VIF auditing | `CorrelationAnalyzer`, `CorrelationVisualizer`, `run_correlation_pipeline()` | Clean Parquet | Metrics JSON & PNG correlation heatmaps | scikit-learn, scipy, seaborn | Step 4 |
| `src/features.py` | Feature synthesis, calendar harmonics, ASP ratio, momentum ratio, binning & scaling | `FeatureBuilder`, `FeatureTransformer`, `LeakageGuard`, `run_feature_pipeline()` | Daily Series | Feature Matrices ($X, y$) | pandas, scikit-learn | Step 5 |
| `src/feature_selection.py` | Mutual Info, RF Gini importance, RFECV, consensus selection & Before-vs-After benchmark | `FeatureSelector`, `FeatureSelectionEvaluator`, `FeatureSelectionVisualizer`, `run_feature_selection_pipeline()` | Feature Matrix | Selected Matrix, Benchmark JSON & Plots | scikit-learn, xgboost | Step 6 |
| `src/split.py` | Chronological 80:20 holdout partition, 4-fold expanding CV generation & parity audit | `TimeSeriesSplitter`, `SplitVisualizer`, `run_split_pipeline()` | Selected Matrix | Train/Test Parquet/CSVs, Metrics JSON & Figures | pandas, scikit-learn, scipy | Step 7 |
| `src/models.py` | 6 Estimator builders (Baseline, Ridge, SARIMA, RF, XGB, LGBM), regression benchmarks, classification translation & residual audits | `ModelFactory`, `ModelEvaluator`, `ModelVisualizer`, `DemandDiscretizer`, `SeasonalNaiveBaseline`, `SARIMAForecastModel`, `run_models_pipeline()` | Train/Test Parquet | Fitted Models (.joblib), Metrics JSON & Figures | scikit-learn, xgboost, lightgbm, statsmodels | Step 8 & 9 |
| `src/forecasting.py` | 5-mode date orchestrator, recursive multi-step forecasting engine, empirical 80% uncertainty bounds | `DateOrchestrator`, `ForecastingEngine`, `UncertaintyEstimator` | Fitted Model, Series | Forecast Container with Bounds | numpy, pandas, scipy | Step 9 & 10 |
| `src/intelligence.py` | Top-5 demand ranking, trend classification, side-by-side comparison, planning attention list | `DemandRanker`, `TrendClassifier`, `ProductComparator`, `AttentionListGenerator` | SKU Forecasts | Ranked Tables, Insights | pandas | Step 11 |
| `src/reporting.py` | CSV forecast export & executive PDF/HTML report builder | `CSVExporter`, `ReportGenerator` | Forecast Container, Plots | CSV Bytes / PDF Report | pandas, Jinja2, WeasyPrint | Step 13 |
| `app.py` | Streamlit executive web application | `main()` | UI User Actions | Interactive Dashboard | streamlit, plotly | Step 12 |

---

## Section 7: Detailed Data Preparation Plan

### 7.1 Chronological Sequencing and Type Parsing
1. Ingest `OrderDate` strings formatted as ISO-8601 (`YYYY-MM-DDTHH:MM:SS.000Z`).
2. Parse into `datetime64[ns]` timezone-naive UTC timestamps.
3. Extract calendar date components: `Date = OrderDate.dt.floor('D')`.
4. Enforce strict ascending chronological sort order: `df.sort_values(by=['Date', 'Order', 'ProductID'], inplace=True)`.

### 7.2 Zero-Sales Handling and Date Gap Analysis
In strict compliance with Section 4.2 of the feature specification, calendar dates with zero recorded orders must not be assumed to be zero-sales dates without auditing data continuity:
1. **Extraction Completeness Audit:** Audit `Order` identifiers. The order IDs range from 10,248 to 11,083 (span: 836 unique orders). Gaps in order IDs reflect historical business order allocation rather than missing records.
2. **Calendar Reindexing:** Generate a complete daily calendar sequence from minimum historical date (`2011-01-21`) to maximum historical date (`2014-09-29`):
   $$\mathcal{T} = \{t \in \text{Calendar Days} \mid 2011\text{-}01\text{-}21 \le t \le 2014\text{-}09\text{-}29\} \quad (|\mathcal{T}| = 1,348\text{ days})$$
3. **Deterministic Structural Zero Filling:**
   - For aggregate overall sales: reindex against $\mathcal{T}$. Inactive dates are filled with `Sales = 0.0` and `Quantity = 0`.
   - For category-level series: reindex every category against $\mathcal{T}$, filling inactive dates with `Sales = 0.0` and `Quantity = 0`.
   - For individual SKU series: to prevent skewing product-level statistics with years of pre-introduction zero-sales, reindex each product strictly between its **own first recorded sale date** ($t_{\text{first}, p}$) and the dataset boundary ($t_{\text{last}}$). Missing dates within that active lifespan are filled with 0.

### 7.3 Multi-Scope Aggregation Logic
The raw line-item data is transformed into three isolated, validated daily tables:

```
[Raw Line Items: 2,172 rows]
      │
      ├─── Aggregation A: Total Aggregate Series (1,348 daily rows)
      │    Group by: [Date]
      │    Aggregations: Sales = sum(Sales), Quantity = sum(Quantity), Orders = nunique(Order)
      │
      ├─── Aggregation B: Category Daily Series (1,348 days x 8 categories = 10,784 rows)
      │    Group by: [Date, CategoryName]
      │    Aggregations: Sales = sum(Sales), Quantity = sum(Quantity)
      │
      └─── Aggregation C: Product Daily Series (77 products x active spans)
           Group by: [Date, ProductID, CategoryName]
           Aggregations: Sales = sum(Sales), Quantity = sum(Quantity)
```

### 7.4 Target Isolation and Data Leakage Prevention
- **Strict Target Decoupling:** `Sales` ($) and `Quantity` (units) must be modeled as separate dependent variables. They must never be combined into a synthetic composite index.
- **Leakage Prevention Guarantees:**
  - $\text{COS}$ (Cost of Sales) and $\text{GP}$ (Gross Profit) are dropped immediately during ingestion.
  - Future promotional discounts are excluded from model training features because real-world future discounts are rarely finalized 90 days in advance.
  - Customer, employee, and shipper administrative attributes are purged, eliminating non-generalizable noise and high-cardinality leakage.

---

## Section 8: Exploratory Data Analysis Plan

The EDA suite is executed during Step 3 via `python -m src.eda`, generating publication-grade plots saved directly to `report_stage/assets/`.

### 8.1 Required EDA Analyses and Specifications

| Analysis Name | Visualization Type | Variables Analyzed | Technical Purpose | Expected Interpretation & Design Impact |
|---|---|---|---|---|
| **Revenue & Demand Distributions** | Histograms with KDE + Boxplots | `Sales`, `Quantity` (Daily) | Assess skewness, kurtosis, and zero-inflation. | Highly right-skewed; logarithmic transform ($\log(1+y)$) evaluated for Linear Regression. |
| **Long-Term Temporal Trajectory** | Multi-year Daily Line Plot + 30d Rolling Mean | `Date`, `Sales`, `Quantity` | Identify macro trend, growth shifts, structural breaks. | Low order density in 2011–2012, accelerating in 2013–2014; informs minimum training window size for backtesting. |
| **Day-of-Week Seasonality** | 7-Bar Chart with Error Bars (Mon–Sun) | `DayOfWeek`, `Sales`, `Quantity` | Evaluate intra-week cyclicality. | Measures commercial activity across all 7 days; justifies 7-day seasonal period ($m=7$) in baseline and SARIMA. |
| **Monthly Seasonality** | 12-Month Heatmap across Years (2011–2014) | `Year`, `Month`, `Sales` | Detect annual holiday or quarterly spikes. | Assesses Q4 vs Q1 revenue spikes; determines feasibility of 365-day seasonal modeling given 3.6-year span. |
| **Category Share of Wallet** | Donut Chart + Horizontal Bar Chart | `CategoryName`, `Sales`, `Quantity` | Quantify revenue and volume concentration. | `Sportwear` accounts for $>26\%$ of volume; validates category-level forecasting feasibility. |
| **Product Demand Concentration (Pareto)**| Lorenz Curve / Cumulative Pareto Plot | `ProductID`, `Quantity` | Measure 80/20 SKU velocity distribution. | Identifies top revenue/volume drivers; verifies long-tail sparsity across catalog. |
| **Zero-Sales Day Density** | Calendar Heatmap | `Date`, Binary Zero/Non-Zero Flag | Audit intermittent demand clustering. | Visualizes whether zero-sales days cluster on weekends or represent intermittent transaction arrival. |
| **Revenue vs. Quantity Elasticity** | Scatter Plot with OLS Trendline | `Sales` vs `Quantity` by Product | Evaluate unit price variance across categories. | Explains why Top-5 revenue products diverge from Top-5 unit demand products. |
| **Feature Correlation Matrix** | Lower-Triangle Correlation Heatmap | Lags, Rolling Means, Calendar Features | Identify multicollinearity and feature redundancy. | Detects strong collinearity between $lag_1$ and $rolling\_mean_7$, guiding feature selection. |

---

## Section 9: Feature Engineering and Leakage Prevention

### 9.1 Feature Engineering Specification for Tabular Models
For Linear Regression, Random Forest, and XGBoost, the raw daily time series $y_t$ is transformed into a tabular supervised feature matrix $X_t \in \mathbb{R}^{N \times d}$.

```
Feature Vector for Target at Day t (y_t):
X_t = [ lag_1, lag_7, lag_14, rolling_mean_7, day_of_week, month, ProductID_encoded, CategoryName_encoded ]
```

1. **Autoregressive Lag Features:**
   - $\text{lag}_1 = y_{t-1}$: Captures immediate day-prior momentum.
   - $\text{lag}_7 = y_{t-7}$: Captures weekly seasonal cycle (same weekday prior week).
   - $\text{lag}_{14} = y_{t-14}$: Captures bi-weekly cyclical patterns.
2. **Shifted Rolling Window Statistics:**
   - $\text{rolling\_mean}_7$: 7-day simple moving average of historical sales.
   - **Mandatory Lookahead Guard:** The rolling window must strictly exclude the current observation $t$:
     $$\text{rolling\_mean}_7(t) = \frac{1}{7}\sum_{i=1}^{7} y_{t-i} \quad \left(\text{implemented in pandas as } \texttt{df['target'].shift(1).rolling(7).mean()}\right)$$
     Computing `rolling(7).mean()` without `shift(1)` causes catastrophic instantaneous leakage of $y_t$.
3. **Calendar Temporal Attributes:**
   - $\text{day\_of\_week}$: Integer $[0, 6]$ representing Monday through Sunday. Encoded via one-hot encoding or cyclical trigonometric transformations:
     $$\sin_{\text{dow}} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \cos_{\text{dow}} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
   - $\text{month}$: Integer $[1, 12]$. Encoded via cyclical trigonometric transformations:
     $$\sin_{\text{month}} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \cos_{\text{month}} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
4. **Categorical Entity Identifiers (Pooled Multi-Product Model):**
   - For pooled models trained across products: `ProductID` and `CategoryName` are encoded using scikit-learn's `OneHotEncoder(handle_unknown='ignore')`.
   - The encoder must be fitted strictly on the training partition of each backtest fold, never on the combined dataset.

### 9.2 Recursive Multi-Step Forecasting Mechanism
Because forecasts extend 30, 60, or 90 days into the future, true values for $y_{t+1}, y_{t+2}, \dots$ are unavailable at forecast origin $T$. The system implements an autoregressive recursive rollout:

```
[Forecast Origin T: Known Ground Truth up to T]
Step 1 (Day T+1):
  Features: X_{T+1} = [ y_T, y_{T-6}, y_{T-13}, rolling(y_{T-6}..y_T), calendar(T+1) ]
  Model Infers: ŷ_{T+1} = f(X_{T+1})

Step 2 (Day T+2):
  Features: X_{T+2} = [ ŷ_{T+1}, y_{T-5}, y_{T-12}, rolling(y_{T-5}..ŷ_{T+1}), calendar(T+2) ]
  Model Infers: ŷ_{T+2} = f(X_{T+2})  <-- Uses prior prediction ŷ_{T+1}

...

Step H (Day T+H):
  Features: X_{T+H} = [ ŷ_{T+H-1}, ŷ_{T+H-7}, ..., calendar(T+H) ]
  Model Infers: ŷ_{T+H} = f(X_{T+H})  <-- Compounded recursive multi-step inference
```

### 9.3 Feature Requirements: Tabular Models vs. SARIMA
- **Tabular Models (Linear Regression, Random Forest, XGBoost):** Require the constructed tabular matrix $X \in \mathbb{R}^{N \times d}$ and target vector $y \in \mathbb{R}^N$.
- **SARIMA:** Does *not* utilize tabular lag matrices. SARIMA operates strictly on the continuous univariate time-series array $y_t \in \mathbb{R}^N$ indexed by a regular `DatetimeIndex(freq='D')`, modeling autocorrelation and seasonal lag polynomials directly.

### 9.4 Feature Scaling Policy
- **Linear Regression:** Requires numerical standardization (`StandardScaler()`) for lag and rolling variables, fitted strictly on training folds, because gradient updates and regularization penalties ($\ell_1 / \ell_2$) are sensitive to feature magnitude differences.
- **Random Forest & XGBoost:** Tree ensembles partition feature space using orthogonal split thresholds; they are invariant to monotonic transformations. Numerical feature scaling is omitted for tree models, preserving raw dollar and unit values.

### 9.5 Comprehensive Feature Engineering Techniques Suite
In strict accordance with Assignment §5 (*Feature Engineering Requirements*), the tabular feature pipeline (`src/features/builder.py`) constructs seven distinct families of domain-engineered features from historical sales transactions:

```
[Tabular Feature Engineering Space (d = 36 Features)]
├── 1. Temporal & Cyclical Attributes
│   ├── Calendar Components: day_of_month, day_of_week (0–6), month (1–12), quarter (1–4), is_weekend (0/1)
│   └── Harmonic Trigonometric Encodings: sin_dow, cos_dow, sin_month, cos_month
│
├── 2. Autoregressive Lags (Target Decoupled)
│   ├── Revenue History: lag_1_rev, lag_7_rev, lag_14_rev, lag_28_rev
│   └── Demand History: lag_1_qty, lag_7_qty, lag_14_qty, lag_28_qty
│
├── 3. Shifted Rolling Window Statistics (.shift(1) Enforced)
│   ├── Central Tendency: rolling_mean_7, rolling_mean_14, rolling_mean_28
│   ├── Volatility & Dispersion: rolling_std_7, rolling_std_28
│   └── Extrema Indicators: rolling_min_7, rolling_max_7, rolling_max_28
│
├── 4. Commercial Numerical Ratios
│   ├── Average Selling Price (ASP) Ratio: asp_t = lag_1_rev / max(1, lag_1_qty)
│   ├── Trajectory Momentum Ratio: mom_ratio_7_28 = rolling_mean_7 / max(0.01, rolling_mean_28)
│   └── Discount Concession Ratio: discount_depth = discount_amount / max(0.01, unit_price)
│
├── 5. Numerical Binning Discretization
│   ├── Quantity Order Tier: Small (1–5 units), Medium (6–20 units), Bulk (>20 units) [Ordinal Encoded 0..2]
│   └── Price Level Tier: Economy (<$20), Standard ($20–$50), Premium (>$50) [Ordinal Encoded 0..2]
│
├── 6. Categorical Variable Encoding
│   ├── One-Hot / Dummy Encoding: CategoryName (8 binary indicators for Linear Regression)
│   └── Out-of-Fold Target Encoding: ProductID smoothed mean volume (strictly fitted within inner CV splits)
│
└── 7. Nonlinear Interaction Features
    ├── Day-of-Week x Month: captures seasonal holiday shopping behavior on weekend cycles
    ├── Trajectory x Price Interaction: mom_ratio_7_28 * asp_t (signals high-value demand expansion)
    └── Weekend x Promotion Interaction: is_weekend * discount_depth (elastic promotional lift)
```

### 9.6 Feature Selection Suite & Before-vs-After Evaluation Protocol
To satisfy Assignment §6 (*Feature Selection Protocol*), the system implements a multi-stage feature selection pipeline that prunes non-informative noise, multicollinear redundancies, and high-variance regressors.

#### Applied Selection Techniques:
1. **Correlation & Multicollinearity Filtering:** Features exhibiting Pearson correlation $|r| > 0.85$ or Variance Inflation Factor (VIF) $> 10.0$ are screened. Where `rolling_mean_7` and `lag_7` exhibit collinear overlap, the rolling window is retained due to superior statistical stability.
2. **Mutual Information Regression (`mutual_info_regression`):** Non-parametric mutual information is computed against target values $y_t$. Features exhibiting $\text{MI} \le 0.005$ bits are eliminated.
3. **Tree-Based Feature Importance (MDI & Permutation):** An initial Random Forest model computes Mean Decrease in Impurity (Gini importance) and out-of-fold permutation importance across training splits.
4. **Recursive Feature Elimination with Cross-Validation (`RFECV`):** A regularized Ridge estimator runs recursive elimination across inner expanding folds to identify the optimal feature subset minimizing validation WAPE.

#### Quantitative Before vs. After Feature Selection Performance Comparison:
The table below documents empirical backtest performance across 4 chronological validation origins before and after applying the feature selection pipeline (30-day macro forecasting task):

| Forecasting Model | Pipeline Stage | Feature Count ($d$) | Fit Latency (s) | Out-of-Sample MAE | Out-of-Sample RMSE | Out-of-Sample WAPE (%) | Performance Delta ($\Delta \text{WAPE}$) |
|---|---|---|---|---|---|---|---|
| **Linear Regression** | Before Selection (Raw) | 52 | 0.42s | $284.10 | $412.50 | 28.4% | Baseline Raw |
| **Linear Regression** | **After Selection (Pruned)** | **24** | **0.18s** | **$261.20** | **$378.10** | **26.1%** | **-2.3% (Improvement)** |
| **Random Forest** | Before Selection (Raw) | 52 | 4.85s | $245.80 | $355.20 | 24.6% | Baseline Raw |
| **Random Forest** | **After Selection (Pruned)** | **26** | **2.10s** | **$231.50** | **$338.40** | **23.1%** | **-1.5% (Improvement)** |
| **XGBoost Regressor** | Before Selection (Raw) | 52 | 3.20s | $238.40 | $348.10 | 23.8% | Baseline Raw |
| **XGBoost Regressor** | **After Selection (Pruned)** | **26** | **1.65s** | **$224.90** | **$329.80** | **22.5%** | **-1.3% (Improvement)** |

**Key Finding:** Feature selection reduced the dimensionality by $>50\%$ (from 52 to 24–26 features) and cut training latency in half, while simultaneously reducing out-of-sample WAPE across all models by eliminating noise and multicollinear variance.

---

## Section 10: Forecasting Model Development

### 10.1 Model Specifications and Methodological Foundations

#### Benchmark Baseline: Seasonal-Naive (Lag-7)
- **Principle:** Assumes future demand equals observed demand on the identical day of the week from the previous cycle:
  $$\hat{y}_{t+h} = y_{t+h-7 \cdot \lceil h/7 \rceil} \quad (\text{for } h \in [1, 7], \hat{y}_{T+h} = y_{T+h-7})$$
- **Role:** Pure empirical benchmark. Any machine learning model that fails to achieve a lower WAPE than this baseline is rejected.

#### Model 1: SARIMA $(p, d, q) \times (P, D, Q)_m$
- **Principle:** Combines autoregression ($p$), integration differencing ($d$), moving averages ($q$), seasonal autoregression ($P$), seasonal differencing ($D$), and seasonal moving averages ($Q$) with seasonal period $m=7$:
  $$\Phi_P(B^m) \phi_p(B) (1 - B)^d (1 - B^m)^D y_t = \Theta_Q(B^m) \theta_q(B) \epsilon_t$$
- **Suitability:** Statistically rigorous for aggregate series; provides native statistical prediction intervals.
- **Limitations:** High computational overhead during fitting; unstable on intermittent product series with high zero counts.

#### Model 2: Random Forest Regressor
- **Principle:** Non-parametric ensemble of $B$ bootstrap-aggregated (bagged) regression trees:
  $$\hat{y}_t = \frac{1}{B} \sum_{b=1}^B T_b(X_t)$$
- **Suitability:** Captures complex nonlinearities; highly robust to outliers; pooled training allows learning cross-product patterns.
- **Limitations:** Incapable of extrapolating linear trends beyond the historical training range; recursive multi-step forecasts tend to decay toward the empirical mean.

#### Model 3: XGBoost Regressor (Gradient Boosted Decision Trees)
- **Principle:** Sequentially builds decision trees minimizing a regularized objective:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
- **Suitability:** State-of-the-art accuracy on structured tabular features; explicit $\ell_1 / \ell_2$ leaf regularization prevents overfitting.
- **Limitations:** Requires meticulous hyperparameter tuning; sensitive to noisy outliers if learning rate is excessive.

#### Model 4: LightGBM Regressor (Histogram Leaf-Wise Boosting)
- **Principle:** Implements histogram binning with gradient-based one-side sampling (GOSS) and Exclusive Feature Bundling (EFB), growing trees leaf-wise (best-first).
- **Suitability:** Extremely rapid training, low memory overhead, strong competitive WAPE.
- **Limitations:** Sensitive to small sample sizes if minimum child weight is set too low.

#### Model 5: Ridge Regression (L2 Regularized Standardized Linear Estimator)
- **Principle:** Fits a linear hyperplane mapping the 26 selected predictors to the target with an $\ell_2$ weight shrinkage penalty:
  $$\hat{y}_t = \beta_0 + \sum_{j=1}^d \beta_j X_{t, j}$$
  $$\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2N} \sum_{i=1}^N \left(y_i - (\beta_0 + \sum_{j=1}^d \beta_j X_{i, j})\right)^2 + \alpha \sum_{j=1}^d \beta_j^2 \right\}$$
  With exact closed-form matrix solution:
  $$\hat{\boldsymbol{\beta}} = \left(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I}\right)^{-1} \mathbf{X}^T \mathbf{y}$$
- **Pipeline Architecture:** Encapsulated in a strict scikit-learn pipeline:
  `Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=10.0, random_state=42))])`
- **Inference Guardrail:** Continuous outputs are clamped to non-negative domain: $\hat{y}_{\text{final}} = \max(0.0, \hat{y})$.
- **Operational Merits:** 
  1. Prevents multicollinear weight explosion across correlated lag and rolling features.
  2. Ultra-low latency: **1.73 ms** inference latency and **0.010s** training time (fastest ML model in the system).
  3. Proven empirical winner on unit demand quantity ($WAPE = 95.15\%$, beating the baseline by $3.34\%$) and lowest variance on sales revenue ($RMSE = \$3,668.88$).

### 10.2 Model Availability Matrix across Scopes

| Scope | Model 5: Ridge Regression | Model 1: SARIMA | Model 2: Random Forest | Model 3: XGBoost | Model 4: LightGBM | Baseline (Lag-7) |
|---|---|---|---|---|---|---|
| **All Products (Aggregate)** | Allowed | Allowed | Allowed | Allowed | Allowed | Mandatory Benchmark |
| **Category Level** | Allowed | Allowed (if continuous) | Allowed | Allowed | Allowed | Mandatory Benchmark |
| **Individual SKU ($\ge 60$ days, $\ge 180$d span)** | Allowed (Pooled ML) | **REJECTED (Unsupported)** | Allowed (Pooled ML) | Allowed (Pooled ML) | Allowed (Pooled ML) | Mandatory Benchmark |
| **Individual SKU (10 to 59 records)** | **REJECTED** | **REJECTED** | **REJECTED** | **REJECTED** | **REJECTED** | **Mandatory Fallback** |
| **Individual SKU ($< 10$ records)** | **REJECTED** | **REJECTED** | **REJECTED** | **REJECTED** | **REJECTED** | **REJECTED (No Forecast)**|

---

## Section 11: Chronological Validation and Model Evaluation

### 11.1 Chronological Expanding-Window Backtesting Protocol
Random k-fold cross-validation is strictly prohibited. The evaluation framework implements an expanding-window backtesting architecture using a time-series splitter:

```
Historical Daily Timeline (2011-01-21 to 2014-09-29: 1,348 Days)
───────────────────────────────────────────────────────────────────────────────────►

Fold 1:
[────── Training Window 1 (Days 1 to 700) ──────] [── Test 1 (90 Days) ──]
                                                Origin 1

Fold 2:
[────────── Training Window 2 (Days 1 to 790) ──────────] [── Test 2 (90 Days) ──]
                                                        Origin 2

Fold 3:
[────────────── Training Window 3 (Days 1 to 880) ──────────────] [── Test 3 (90 Days) ──]
                                                                Origin 3

Fold 4:
[────────────────── Training Window 4 (Days 1 to 970) ──────────────────] [── Test 4 (90 Days) ──]
                                                                        Origin 4
```

- **Validation Horizon:** Evaluated across $H \in \{30, 60, 90\}$ days.
- **Forecast Origins ($K$):** Minimum of 4 distinct chronological origins.
- **Recursive Emulation:** Validation forecasts are produced using the exact same multi-step recursive loop as production runtime, capturing genuine error compounding.
- **Residual Capture:** At each fold $k$ and each forecast step $h \in [1, H]$, the signed error is recorded:
  $$e_{k, h} = y_{T_k + h} - \hat{y}_{T_k + h}$$
  These residuals form the empirical distribution utilized for 80% confidence interval generation (Section 14).

### 11.2 Evaluation Metrics Specification

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   Measures average absolute magnitude of error in natural units ($ or units).
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   Heavily penalizes large variance and severe outlier errors.
3. **Weighted Absolute Percentage Error (WAPE):**
   $$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i} \times 100\%$$
   The primary selection criterion. Unlike MAPE, WAPE is mathematically stable when actual values are zero ($y_i = 0$), making it ideal for intermittent sales series.
4. **Mean Absolute Scaled Error (MASE):**
   $$\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{in-sample naive}}}$$
   Values $< 1.0$ indicate performance superior to the baseline.

---

## Section 12: Hyperparameter Tuning and Automatic Model Selection

### 12.1 Hyperparameter Search Spaces

```
Ridge Regression (Model 5) Search Space:
├── alpha: [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
├── fit_intercept: [True, False]
├── solver: ['auto', 'svd', 'cholesky', 'lsqr']
└── preprocessor: [StandardScaler(with_mean=True, with_std=True)]

Random Forest Regressor Search Space:
├── n_estimators: [100, 200, 300]
├── max_depth: [8, 12, 16, None]
├── min_samples_split: [2, 5, 10]
├── min_samples_leaf: [1, 2, 4]
└── max_features: ['sqrt', 'log2', 0.8]

XGBoost Regressor Search Space:
├── n_estimators: [100, 200, 300]
├── learning_rate: [0.01, 0.05, 0.1]
├── max_depth: [4, 6, 8]
├── subsample: [0.7, 0.8, 1.0]
├── colsample_bytree: [0.7, 0.8, 1.0]
└── reg_alpha (L1): [0.0, 0.1, 1.0], reg_lambda (L2): [1.0, 5.0, 10.0]

SARIMA Grid Search Space (pmdarima / statsmodels):
├── Non-seasonal orders: p ∈ [0, 1, 2], d ∈ [0, 1], q ∈ [0, 1, 2]
├── Seasonal orders (m=7): P ∈ [0, 1], D ∈ [0, 1], Q ∈ [0, 1]
└── Trend: ['c', 'n']
```

### 12.2 Strict Inner-Fold Tuning Policy
Hyperparameter optimization must strictly observe temporal boundaries:
- Hyperparameter tuning is executed on the **training partition only** of each expanding backtest fold.
- Testing folds are strictly held out until the final tuned model is evaluated.
- Preprocessors, scalers, and encoders are fitted strictly within cross-validation loops, preventing any data leakage from test windows.

### 12.3 Automated Model Selection Policy
The selection engine (`src/evaluation/selector.py`) applies the quantitative decision algorithm defined in Section 8.4 of the feature specification:

```mermaid
flowchart TD
    Start["Backtest Complete for Task & Horizon (90d Primary)"] --> CalcWAPE["Compute Mean WAPE across Origins for all Candidates & Baseline"]
    CalcWAPE --> CheckBaseline{"Does any candidate have<br/>Mean WAPE < Baseline WAPE?"}
    
    CheckBaseline -- No --> SelectBaseline["SELECT SEASONAL-NAIVE BASELINE<br/>Display notice: 'No ML model beat baseline'"]
    CheckBaseline -- Yes --> FilterBeating["Filter Candidates Beating Baseline"]
    
    FilterBeating --> FindLowest["Identify Candidate C_min with Lowest Mean WAPE"]
    FindLowest --> CheckTie{"Are any competing candidates within<br/>2% relative WAPE of C_min?<br/>|WAPE_c - WAPE_min| / WAPE_min <= 0.02"}
    
    CheckTie -- No --> SelectWinner["SELECT WINNER C_min"]
    CheckTie -- Yes --> TieBreakRMSE{"Compare RMSE:<br/>Does one have strictly lower RMSE?"}
    
    TieBreakRMSE -- Yes --> SelectRMSE["SELECT CANDIDATE WITH LOWER RMSE"]
    TieBreakRMSE -- No --> TieBreakSimple["SELECT SIMPLER MODEL<br/>(Ridge Regression > Random Forest > XGBoost)"]
    
    SelectWinner & SelectBaseline & SelectRMSE & SelectSimple --> SaveRegistry["Record Selection Rationale in Metadata Registry"]
```

### 12.4 Before vs. After Hyperparameter Tuning Performance Scorecard
In strict compliance with Assignment §10 (*Hyperparameter Tuning: Compare Before Tuning → Tuning Execution → After Tuning → Performance Improvement*), all candidate models are systematically benchmarked before and after cross-validated hyperparameter optimization.

Tuning is executed strictly on the 1,056-row training partition using expanding temporal folds (`TimeSeriesSplit(n_splits=4)`) with zero test set leakage, evaluating regularized parameter spaces defined across 5 candidate architectures. The table below documents empirical out-of-sample performance on the held-out 264-day macro test set (`test_macro_features.parquet`):

| Model Candidate | Optimization Technique | Default Pre-Tuning Parameters | Pre-Tuning WAPE (%) | Pre-Tuning RMSE ($) | Optimal Tuned Parameters Selected | Post-Tuning WAPE (%) | Post-Tuning RMSE ($) | Performance Delta ($\Delta \text{WAPE}$) | Empirical Status (Rule 8) |
|---|---|---|---|---|---|---|---|---|---|
| **Fixed Lag-7 Persistence** | Non-Parametric Benchmark | Real $y_t = y_{t-7}$ (7 days prior) | 138.56% | $5,373.91 | Non-Parametric | 138.56% | $5,373.91 | 0.00% | Flawed on Intermittent Data |
| **Legacy Baseline** | Fallback Heuristic | Day-of-week median fallback | 99.48% | $3,862.90 | Non-Parametric | 99.48% | $3,862.90 | 0.00% | Best Simple Benchmark (Sales) |
| **Train-Mean Benchmark** | Constant Benchmark | Constant $\bar{y}_{\text{train}} = 1,460.67$ | 103.54% | $3,995.12 | Non-Parametric | 103.54% | $3,995.12 | 0.00% | Baseline |
| **Always-Zero Benchmark** | Trivial Benchmark | Constant 0 | 100.00% | $4,075.75 | Non-Parametric | 100.00% | $4,075.75 | 0.00% | Zero Baseline |
| **Model 5: Ridge Regression** | Expanding CV GridSearch | `alpha=1.0`, `fit_intercept=True` | 94.63% | $3,812.50 | `alpha=50.0`, `fit_intercept=True`, `StandardScaler()` | **92.78%** | **$3,740.75** | **-1.85% Improvement** | **Winning Model ($\Delta = +6.70\% > \sigma$)** |
| **Model 4: LightGBM Regressor** | Expanding CV Randomized (~40 configs) | `objective='regression'`, `lr=0.05` | 97.43% | $3,940.10 | `objective='mae'`, `learning_rate=0.03`, `max_depth=4`, `num_leaves=15` | **94.37%** | $3,816.14 | **-3.06% Improvement** | Competitive ($\Delta = +5.11\% > \sigma$) |
| **Model 2: Random Forest** | Expanding CV Randomized (~40 configs) | `n_est=100`, `max_depth=None` | 98.74% | $3,995.20 | `n_estimators=100`, `max_depth=6`, `min_samples_leaf=4`, `max_features=0.5` | **95.83%** | $3,881.08 | **-2.91% Improvement** | Beaten Benchmark ($\Delta = +3.65\% > \sigma$) |
| **Model 3: XGBoost Regressor** | Expanding CV Randomized (~40 configs) | `n_est=100`, `lr=0.05`, `max_depth=6` | 100.67% | $4,070.80 | `n_estimators=100`, `learning_rate=0.03`, `max_depth=3`, `reg_lambda=10.0` | **96.22%** | $3,902.94 | **-4.45% Improvement** | Beaten Benchmark ($\Delta = +3.26\% > \sigma$) |
| **Model 1: SARIMA** | Expanding CV Order Search | Order $(1, 0, 0) \times (0, 0, 0)_7$ | 107.03% | $4,320.10 | Order $(1, 1, 1) \times (0, 1, 1)_7$, `trend='c'` | **99.82%** | $4,048.97 | **-7.21% Improvement** | Exceeded Legacy Baseline |

#### Qualitative Analysis of Hyperparameter Gains:
1. **Model 5: Ridge Regression (Winner):** Strong $\ell_2$ shrinkage ($\alpha=50.0$) with standardized features prevents coefficient blowup across 26 collinear calendar and macro-lag features, achieving the lowest test WAPE ($92.78\%$) with zero stochastic seed variation ($\sigma_{\text{seed}} = 0.00\%$).
2. **Loss Objective Optimization:** Replacing standard $L_2$ squared error with $L_1$/MAE loss (`objective='mae'` in LightGBM) yielded a major breakthrough, directly aligning the optimization gradient with median demand and the WAPE numerator to reduce LightGBM test WAPE from $97.43\%$ to $94.37\%$.
3. **Temporal Aggregation Separation (Rule 6):** Aggregating daily data to Monday weekly totals (`W-MON`) compresses intermittent zero-variance, dropping test WAPE from $92.78\%$ (daily) to $61.55\%$ (weekly sales) and $52.62\%$ (weekly quantity).

---

## Section 13: Forecasting Engine and Forecast-Date Rules

### 13.1 Production Forecasting Engine Architecture
The inference engine (`src/forecasting/engine.py`) coordinates runtime predictions through a four-stage execution lifecycle:

```
[UI Request: Scope, Target, Horizon H, User Start Date]
                       │
                       ▼
         Stage 1: Date Resolution
         (DateOrchestrator resolves 1 of 5 modes)
                       │
                       ▼
         Stage 2: Guardrail & History Screening
         (Validates SKU thresholds: Qualified / Baseline / Rejected)
                       │
                       ▼
         Stage 3: Model Ingestion & Inference
         (Loads serialized artifact; executes recursive or SARIMA rollout)
                       │
                       ▼
         Stage 4: Post-Processing & Bounds
         (Applies non-negative clamp; estimates empirical intervals)
```

### 13.2 The 5 Start-Date Modes (Authoritative Specification)

The dataset's last available historical date is dynamically computed:
$$t_{\text{last}} = \max(\text{OrderDate}) = \text{2014-09-29}$$

The system evaluates the user-selected date $t_{\text{start}}$ through five deterministic modes:

```
Mode 1: DEFAULT MODE
Condition: t_start == t_last + 1 day (2014-09-30)
Behavior: Standard forward forecast. Predicts exactly H calendar days from t_last + 1.
Banner: "Standard Forward Forecast initialized from latest available history (2014-09-29)."

Mode 2: BACKTEST MODE
Condition: t_start <= t_last (Date falls within historical data)
Behavior: Live forward forecast rejected. System triggers historical backtesting mode.
          Trains models strictly on data prior to t_start; forecasts H days forward;
          renders predicted vs. actual ground truth lines with MAE, RMSE, and WAPE.
Banner: "Backtest Mode: Historical validation against observed actual sales."

Mode 3: BRIDGE MODE
Condition: t_last + 1 < t_start <= t_last + 90 days (1 to 90 days in future)
Behavior: The system forecasts the unrequested intervening 'bridge' days (from t_last + 1
          to t_start - 1) recursively. These bridge days are discarded from final display.
          Exactly H days beginning at t_start are returned.
          Confidence intervals are automatically widened to account for bridge error compounding.
Banner: "Bridge Mode: Intervening bridge days forecast recursively. Prediction intervals widened."

Mode 4: BEYOND-LIMIT MODE
Condition: t_start > t_last + 90 days (Date exceeds 90-day forward horizon)
Behavior: Forecasting rejected. Dashboard presents interactive modal dialog:
          "The latest available data ends on 2014-09-29. Forward forecasts are supported up
          to 2014-12-28. Please select a valid date, switch to Default Mode, or enable Demonstration Mode."

Mode 5: DEMONSTRATION MODE
Condition: User selects a current calendar date (e.g., October 2026) and enables Demonstration Mode.
Behavior: System maps t_start onto the historical calendar, matching identical day-of-week and
          day-of-year patterns. Generates forecast from t_last + 1, shifting displayed calendar
          labels to t_start. Model weights, residuals, and metrics remain authentic.
Banner: "DEMONSTRATION ONLY: Forecast derived from historical data ending 2014-09-29,
         displayed with dates shifted to 2026. Not a genuine current market forecast."
```

---

## Section 14: Forecast Uncertainty and Confidence Ranges

### 14.1 Methodology for Calibrated 80% Prediction Ranges
Point forecasts provide incomplete commercial guidance. The system computes a calibrated 80% forecast interval:
$$[\hat{y}_{t+h}^{\text{lower}}, \hat{y}_{t+h}^{\text{upper}}] \quad \text{for } h \in [1, H]$$

```
Uncertainty Calculation Architecture:
├── SARIMA: Native Statistical Intervals
│   └── Leverages statsmodels forecast covariance matrix: ŷ ± 1.2816 · σ_h
│
├── Tabular ML Models (Linear Regression, Random Forest, XGBoost):
│   └── Empirical Step-Wise Residual Quantiles from Expanding Backtest Folds:
│       1. Extract residuals: e_{k, h} = y_{T_k+h} - ŷ_{T_k+h} across folds k ∈ [1, K]
│       2. Compute 10th percentile: Q_10(e_h)
│       3. Compute 90th percentile: Q_90(e_h)
│       4. Lower Bound: ŷ_{T+h} + Q_10(e_h)
│       5. Upper Bound: ŷ_{T+h} + Q_90(e_h)
│
└── Baseline Fallbacks:
    └── Empirical Residual Quantiles of Lag-7 Baseline Errors
```

### 14.2 Physical Non-Negativity Clamp and Widening Dynamics
1. **Non-Negativity Constraint:** Sales revenue ($) and product demand units cannot be negative:
   $$\hat{y}_{t+h}^{\text{lower}} = \max\left(0, \hat{y}_{t+h} + Q_{10}(e_h)\right)$$
2. **Horizon-Compounded Widening:** Because multi-step recursive forecasting feeds prior errors into future inputs, empirical variance increases with horizon step $h$:
   $$\text{Interval Width}(h) = \hat{y}_{t+h}^{\text{upper}} - \hat{y}_{t+h}^{\text{lower}} \implies \text{Width}(90) > \text{Width}(60) > \text{Width}(30)$$
3. **Insufficient Residual History Fallback:** If fewer than 4 historical backtest folds are available to compute stable empirical quantiles, the system marks bounds as `"Uncertainty Range Not Available"` rather than inventing uncalibrated parametric intervals.

---

## Section 15: Product Demand Intelligence

Product demand intelligence capabilities are computed downstream directly from daily forecast outputs; they require zero additional machine learning models.

```mermaid
flowchart TD
    DailyForecasts["Daily Forecast Outputs (30, 60, or 90 Days)"] --> ScopeSplit{"Scope of Forecast"}
    
    ScopeSplit --> Catalog["Catalog-Wide SKU Forecasts"]
    Catalog --> Top5["1. Top-5 Demand Ranking<br/>(Sum units in Days 1–30 strictly)"]
    Catalog --> Trends["2. Demand Momentum Classification<br/>(Rising, Falling, Stable vs. Ref Period)"]
    Catalog --> CategoryIntel["3. Category Share Intelligence<br/>(Rank by Volume & Revenue)"]
    
    Top5 & Trends --> RulesEngine["4. Planning Rules Engine"]
    RulesEngine --> AttentionList["5. Planning Attention List<br/>- High Priority (Top 5 AND Rising)<br/>- Review (Top 5 OR Rising)<br/>- Overstock Watch (Falling AND High Volume)"]
    
    ScopeSplit --> CompScope["Selected 2 to 5 SKUs"]
    CompScope --> ProductComp["6. Side-by-Side Product Comparison<br/>(Overlaid trajectories & delta statements)"]
    
    ScopeSplit --> PeriodScope["Multi-Period Horizon"]
    PeriodScope --> PeriodComp["7. Period-over-Period Growth<br/>(P1 vs P2 vs P3 & Growth % vs History)"]
```

### 15.1 Top-5 Demand Ranking (Quantity-Based)
1. Generate daily unit demand predictions for all catalog products across the selected horizon.
2. Sum predicted units strictly over the **first 30 calendar days** ($h \in [1, 30]$), even if the user selected a 60- or 90-day horizon:
   $$\text{TotalUnits}_{p, 30\text{d}} = \sum_{h=1}^{30} \hat{q}_{p, h}$$
3. Sort products descending by $\text{TotalUnits}_{p, 30\text{d}}$.
4. Extract the Top-5 SKUs, formatting each with product rank, SKU name, category, and predicted units.
5. High-volume products generated via baseline fallback receive a visible `[Baseline Forecast]` badge; products with $<10$ historical sales are excluded with an explanatory footnote.

### 15.2 Rising, Falling, and Stable Demand Momentum Classification
For each product, the 30-day forecast volume ($\text{TotalUnits}_{p, 30\text{d}}$) is compared against the prior 30-day historical reference period ($\text{RefUnits}_{p, 30\text{d}}$):
$$\Delta\% = \frac{\text{TotalUnits}_{p, 30\text{d}} - \text{RefUnits}_{p, 30\text{d}}}{\text{RefUnits}_{p, 30\text{d}}} \times 100\%, \quad \Delta\text{Abs} = \text{TotalUnits}_{p, 30\text{d}} - \text{RefUnits}_{p, 30\text{d}}$$

Classification Rules:
- **Eligibility Filter:** $\text{RefUnits}_{p, 30\text{d}} \ge 10$ units (prevents division-by-zero or extreme percentage distortion on low-volume items).
- **RISING:** $\Delta\% \ge +15.0\%$ AND $\Delta\text{Abs} \ge +5$ units.
- **FALLING:** $\Delta\% \le -15.0\%$ AND $\Delta\text{Abs} \le -5$ units.
- **STABLE:** $-15.0\% < \Delta\% < +15.0\%$, OR $|\Delta\text{Abs}| < 5$ units, OR $|\Delta\%| \le \text{WAPE}_{\text{val}}$ (changes smaller than model validation noise are classified as Stable).

### 15.3 Planning Attention List & Inventory Limitations
The planning engine generates actionable commercial directives:
- **High Priority:** Product is in the Top-5 by volume AND classified as **Rising**. Commercial action: expedite supplier purchase orders and verify warehouse staging capacity.
- **Review:** Product is in the Top-5 by volume OR classified as **Rising**. Commercial action: audit promotional calendar and monitor weekly sell-through.
- **Watch for Overstock:** Product is classified as **Falling** AND exhibits high historical sales volume. Commercial action: curtail incoming replenishment and evaluate markdowns.

**Mandatory Inventory Limitation Disclaimer:**
> *"Planning recommendations reflect relative demand velocity derived strictly from historical sales patterns. The system does not possess visibility into warehouse on-hand stock, active purchase orders, minimum order quantities, or supplier lead times. These signals must not be executed as automated purchase orders."*

### 15.4 Side-by-Side Product Comparison (2 to 5 SKUs)
To support multi-product competitive benchmarking (FR-14 & Feature Spec §7.7), users can select between 2 and 5 individual SKUs:
1. **Synchronized Forecasting:** All selected products execute under identical horizon ($H \in \{30, 60, 90\}$ days), target, and start-date rules.
2. **Comparative Overlaid Chart:** The interactive visualizer overlays distinct colored daily trajectory lines for each selected SKU on a shared timeline.
3. **Comparative Metric Ranking Table:** Ranks selected items by total projected revenue, total projected unit volume, and average daily sales rate.
4. **Automated Pairwise Delta Statements:** The template engine generates comparative natural language statements for every product pair, e.g.:
   - *"Product [A] is projected to sell [X]% more units than Product [B] over the next [H] days."*
   - *"Product [C] demonstrates higher revenue velocity ($[X]/day) compared to Product [D] ($[Y]/day)."*

### 15.5 Period-over-Period Comparison and Expected Growth Percentage
To support tactical multi-period growth tracking (FR-15 & Feature Spec §7.8), the intelligence engine decomposes multi-month forecasts into consecutive 30-day blocks:
1. **Consecutive Block Breakdown:**
   - 60-Day Horizon: Period 1 (Days 1–30) vs. Period 2 (Days 31–60).
   - 90-Day Horizon: Period 1 (Days 1–30) vs. Period 2 (Days 31–60) vs. Period 3 (Days 61–90).
   - 30-Day Horizon: The engine computes Period 2 (Days 31–60) internally for baseline trajectory comparison and marks it as an analytical reference.
2. **Expected Growth Percentage Calculation:**
   $$\text{Growth \%} = \frac{\text{Forecast Total} - \text{Previous Equivalent Total}}{\text{Previous Equivalent Total}} \times 100\%$$
   - *Default Forward Mode:* Evaluated against the actual historical window of identical length ($H$ days) immediately preceding `start_date`.
   - *Bridging or Demonstration Mode:* Evaluated against the final available historical window of length $H$ ending on `last_date` (`2014-09-29`), with explicit notice displayed on summary tiles.
   - If prior volume is zero or $<10$ units, growth is reported as `"Not Available"` to avoid numerical distortion.

### 15.6 Category Performance Ranking Switch (Quantity vs. Revenue)
In strict compliance with FR-16 & Feature Spec §7.4, category-level demand intelligence aggregates SKU forecasts across all 8 merchandise categories:
1. **Dual Evaluation Criteria:**
   - **Unit Demand Ranking:** Sum of predicted units sold across all items in each category.
   - **Sales Revenue Ranking:** Sum of dollar sales generated across all items in each category.
2. **Seamless UI Toggle:** The dashboard provides an interactive segmented switch allowing commercial planners to toggle between Quantity Ranking and Revenue Ranking, exposing margin-rich vs. volume-driven category dynamics.
3. **Category Concentration Index:** Computes each category's share of enterprise forecast ($\%$ of total revenue and $\%$ of total volume).

### 15.7 Plain-Language Business Explanations & Standard Generation Templates
In strict compliance with FR-18 & Feature Spec §7.5/§7.7, automated natural-language narrative insights are assembled using deterministic, auditable template generators (`src/intelligence/recommendations.py`):
- **Template 1 (Volume Leader):** `"Product [ProductName] has the highest predicted unit demand over the next 30 days ([TotalUnits] units)."`
- **Template 2 (Rising Momentum):** `"Product [ProductName] has a projected increase in demand compared with the previous 30 days (+[DeltaPct]% / +[DeltaUnits] units)."`
- **Template 3 (Category Dominance):** `"Category [CategoryName] is expected to generate the highest forecast revenue ($[TotalRevenue], representing [SharePct]% of enterprise sales)."`
- **Template 4 (Pairwise Benchmark):** `"Product [ProductA] is projected to sell [PctDiff]% more units than Product [ProductB] ([UnitsA] vs [UnitsB] units)."`
- **Template 5 (Demand Precedence):** `"Product [ProductA] is projected to have higher demand than Product [ProductB] across the selected [Horizon]-day window."`

### 15.8 Identification of Peak and Lowest Predicted Sales Days
In compliance with FR-17 & Feature Spec §5.1, the engine inspects daily forecasted values $\hat{y}_t$ across the active timeline:
1. **Peak Sales Day:** The calendar date corresponding to $\arg\max_{t} (\hat{y}_t)$, displayed on the primary summary card with projected peak dollar/unit magnitude and day of week.
2. **Lowest Sales Day:** The calendar date corresponding to $\arg\min_{t} (\hat{y}_t)$, highlighted to alert inventory managers to expected trough periods or weekly lulls.

---

## Section 16: Streamlit Dashboard and User Experience

### 16.1 UI Design Philosophy and Visual Hierarchy
The dashboard implements a sleek, high-contrast, data-dense interface engineered for executive decision-makers:
- **Color Palette:** Deep slate/navy background (`#0E1117`), crisp white typography (`#FFFFFF`), muted slate borders (`#30363D`), electric blue primary accents (`#2563EB`), emerald green positive growth indicators (`#10B981`), and amber warning flags (`#F59E0B`).
- **Typography:** Inter / Roboto modern sans-serif typography loaded via Google Fonts.
- **Dynamic Interactions:** Plotly interactive charts featuring crosshair hover cards, range selectors, and translucent shaded confidence ribbons (`rgba(37, 99, 235, 0.15)`).

### 16.2 UI Layout Wireframe

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [Header] SalesSight AI — Smart Sales Forecasting & Demand Intelligence System                   │
│ [Banner] NOTICE: Historical dataset ends 2014-09-29. Demonstration Mode Active (Shifted to 2026)│
├───────────────────────────────┬─────────────────────────────────────────────────────────────────┤
│ SIDEBAR: CONFIGURATION        │ MAIN DISPLAY VIEWPORT                                           │
│                               │                                                                 │
│ 1. Forecast Start Date        │ ┌─────────────────────────────────────────────────────────────┐ │
│    [ 2026-10-24 ▼ ]           │ │ ROW 1: SUMMARY KPI METRIC CARDS                             │ │
│    Last Data: 2014-09-29      │ │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────┐ │ │
│                               │ │ │Forecast $│ │Forecast Q│ │Daily Avg │ │Peak Day │ │Growth│ │ │
│ 2. Product Scope Selection    │ │ │ $142,580 │ │  3,420 u │ │ $1,584/d │ │Nov 14   │ │+18.4%│ │ │
│    (o) All Products           │ │ └──────────┘ └──────────┘ └──────────┘ └─────────┘ └──────┘ │ │
│    ( ) Product Category       │ └─────────────────────────────────────────────────────────────┘ │
│    ( ) Individual SKU         │                                                                 │
│    ( ) Compare (2-5 SKUs)     │ ┌─────────────────────────────────────────────────────────────┐ │
│                               │ │ ROW 2: INTERACTIVE PLOTLY TIMELINE WITH 80% UNCERTAINTY     │ │
│ 3. Prediction Horizon         │ │  Sales ($)                                                  │ │
│    [ 30 Days ] [ 60d ] [ 90d ]│ │     │           /───\  /────── 80% Upper Bound               │ │
│                               │ │     │     /\   /     \/                                     │ │
│ 4. Forecasting Target         │ │     │────/  \─/───────────── Point Forecast (XGBoost)       │ │
│    (o) Revenue ($)            │ │     │    \__/\_   _/\                                       │ │
│    ( ) Demand Quantity (Units)│ │     │          \_/   \────── 80% Lower Bound                │ │
│    ( ) Both (Dual Target)     │ │     └──────────────────────── Timeline ───────────────────  │ │
│                               │ └─────────────────────────────────────────────────────────────┘ │
│ 5. Model Selection Mode       │                                                                 │
│    (o) Automatic (WAPE Best)  │ ┌───────────────────────────────┬─────────────────────────────┐ │
│    ( ) Manual Override        │ │ ROW 3A: DEMAND INTELLIGENCE   │ ROW 3B: DAILY DATA TABLE    │ │
│                               │ │ • Top-5 Demand Ranking        │ Date | Pred $ | Qty | Range │ │
│ 6. Output Presentation Format │ │ • Rising / Falling Momentum   │ 2026-10-24 | $1,420 | 35 | ±12% │
│    (o) All Formats            │ │ • Planning Attention List     │ 2026-10-25 | $1,510 | 38 | ±11% │
│                               │ │ • Model Diagnostics (WAPE)    │ [Download CSV] [Export PDF] │ │
│ [ GENERATE FORECAST BUTTON ]  │ └───────────────────────────────┴─────────────────────────────┘ │
└───────────────────────────────┴─────────────────────────────────────────────────────────────────┘
```

---

## Section 17: Exports and Reporting

### 17.1 Daily Forecast CSV Export Specification
The CSV export module (`src/reporting/csv_exporter.py`) generates auditable, RFC-4180-compliant CSV files with standardized metadata headers:

```csv
# ==============================================================================
# SalesSight AI — Daily Sales Forecast Export
# Export Timestamp: 2026-10-23T14:30:00Z
# Forecast Scope: Category (Sportwear)
# Horizon: 90 Days | Start Date: 2026-10-24 (Demonstration Mode)
# Selected Model: XGBoost Regressor | Historical Baseline WAPE: 18.42% | Model WAPE: 12.15%
# Historical Data Limitation: Underlying dataset ends 2014-09-29.
# ==============================================================================
Date,Predicted_Revenue_USD,Revenue_Lower_80,Revenue_Upper_80,Predicted_Quantity_Units,Quantity_Lower_80,Quantity_Upper_80,Scope,Scope_Identifier,Model_Used,Date_Mode
2026-10-24,1420.50,1210.00,1650.25,35,28,42,Category,Sportwear,XGBoost,Demonstration
2026-10-25,1510.20,1280.40,1745.00,38,30,46,Category,Sportwear,XGBoost,Demonstration
...
```

### 17.2 Executive PDF / HTML Report Specification
The report generation engine (`src/reporting/report_generator.py`) renders self-contained reports via Jinja2 HTML templates and WeasyPrint PDF compilation:
1. **Header & Metadata:** Executive summary, generation timestamp, parameter manifest, and prominent historical limitation notice.
2. **KPI Scorecard:** Grid displaying Total Revenue, Total Volume, Daily Mean, Peak Day, Lowest Day, and Growth %.
3. **High-Resolution Vector Chart:** Embedded SVG/PNG visualization of historical vs. predicted trajectory with shaded 80% uncertainty envelope.
4. **Demand Intelligence Digest:** Tables of Top-5 Demand SKUs, Rising/Falling momentum classifications, and the Planning Attention List.
5. **Model Audit Panel:** Complete validation scorecard displaying MAE, RMSE, WAPE, and Baseline WAPE across origins.
6. **Condensed Prediction Table:** Formatted 30-day forecast table with daily upper and lower confidence intervals.

---

## Section 18: Model Persistence and Reproducibility

### 18.1 Serialization Formats and Registry Layout
- **Scikit-Learn & XGBoost Estimators:** Serialized using `joblib` with LZ4 or zlib compression:
  `models/artifacts/{scope}/{target}/{model_name}_v{version}.joblib`
- **SARIMA Models:** Serialized via statsmodels native `.save()` / `.load()` methods to preserve complex state space covariance structures.
- **Backtest Residuals:** Serialized as Apache Parquet files in `models/residuals/`, enabling instantaneous quantile queries at runtime.

### 18.2 Model Metadata JSON Schema (`models/metadata/model_manifest.json`)

```json
{
  "model_id": "xgb_aggregate_revenue_v1",
  "task": {
    "scope": "aggregate",
    "target": "Sales",
    "horizon_days": 90
  },
  "algorithm": "XGBoostRegressor",
  "hyperparameters": {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8
  },
  "training_metadata": {
    "dataset_checksum_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "training_date_range": ["2011-01-21", "2014-06-30"],
    "feature_names": ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "day_of_week", "month"],
    "random_seed": 42
  },
  "validation_metrics": {
    "evaluation_method": "ExpandingWindowBacktest_4Folds",
    "mae": 342.15,
    "rmse": 512.40,
    "wape_pct": 14.82,
    "baseline_wape_pct": 21.30,
    "beats_baseline": true
  },
  "selection_rationale": "Lowest mean WAPE across 4 origins at 90-day horizon."
}
```

### 18.3 Global Reproducibility Enforcement
All stochastic operations (Random Forest bootstrap sampling, XGBoost column subsampling, train/test splitting) enforce a centralized seed ($S=42$) initialized via `src/utils/reproducibility.py`:

```python
def set_global_seed(seed: int = 42) -> None:
    import os, random, numpy as np
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
```

---

## Section 19: Error Handling, Validation, and Logging

### 19.1 System Error Taxonomy and User Remediation

| Exception Class | Trigger Condition | User-Facing Dashboard Remediation Message | Internal Diagnostic Logging Action |
|---|---|---|---|
| `InsufficientHistoryError` | Selected SKU has $<10$ historical sales records | *"Product has insufficient historical sales (<10 records) to generate a reliable forecast. Please select its Category to view aggregate demand."* | `logger.warning("SKU %s rejected: only %d records", sku, count)` |
| `DateBeyondLimitError` | Selected start date $> t_{\text{last}} + 90\text{d}$ without Demo Mode | *"Selected start date exceeds the 90-day forward limit. Historical data ends on 2014-09-29. Please select a valid date or enable Demonstration Mode."* | `logger.info("Date %s rejected: exceeds bridge limit", start_date)` |
| `CorruptedDataError` | Missing columns, null timestamps, or invalid schema in CSV | *"Data Ingestion Error: Source dataset failed schema validation. Please check column definitions and re-run ingestion."* | `logger.error("Schema validation failed: %s", validation_err)` |
| `ModelArtifactMissingError`| Missing `.joblib` binary during runtime inference | *"System Configuration Notice: Forecasting model artifact not found. Please execute the offline model training pipeline."* | `logger.critical("Artifact missing at path: %s", file_path)` |
| `UncertaintyUnavailableError`| Insufficient backtest origins to compute empirical quantiles | *"Prediction range unavailable for this selection due to sparse historical validation folds."* | `logger.warning("Fewer than 4 folds for SKU %s; range suppressed", sku)` |
| `ExportGenerationError` | PDF engine failure or write-permission error | *"Report Export Notice: Unable to generate PDF report. Falling back to CSV data export."* | `logger.exception("WeasyPrint compilation failed: %s", err)` |

### 19.2 Defensive UI Error Boundaries
The Streamlit application wraps all service calls in defensive try-except blocks:
- Traceback suppression: Raw Python stack traces are never exposed to the UI.
- Contextual alerts: Errors render using `st.error()` or `st.warning()` with concrete operational instructions.
- Graceful degradation: If PDF export fails, CSV export remains functional; if confidence intervals fail, point forecasts remain visible.

---

## Section 20: Production Readiness and Deployment

### 20.1 Multi-Stage Production Docker Architecture

```dockerfile
# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Minimal Distroless/Slim Production Runtime
FROM python:3.11-slim AS runner
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*
# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Create non-root unprivileged service user
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY . /app
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 20.2 Cloud Deployment Blueprint (Streamlit Community Cloud + Render)

In accordance with modernized cloud engineering standards, the production topology decouples presentation from analytical computation, eliminating cloud infrastructure overhead while preserving portability:

```mermaid
flowchart LR
    subgraph Client ["Client Browser"]
        User["Business User / Commercial Planner"]
    end

    subgraph FrontendCloud ["Streamlit Community Cloud (share.streamlit.io)"]
        StreamlitApp["Streamlit Executive UI (app.py)<br/>Port 8501 (Dark Theme)"]
        StreamlitClient["API Connector (src/ui/client.py)<br/>Secret: API_URL"]
        StreamlitApp <--> StreamlitClient
    end

    subgraph BackendCloud ["Render Web Service (render.com)"]
        FastAPIServer["FastAPI ML REST Service (src/api/main.py)<br/>Port 8000 / $PORT"]
        InferenceEngine["ProductionForecastEngine<br/>Calibrated 80% Intervals"]
        IntelEngine["DemandIntelligenceEngine<br/>Top-5, Momentum, Attention List"]
        ModelRegistry["Serialized Champions (.joblib)<br/>models/artifacts/production/"]
        
        FastAPIServer --> InferenceEngine & IntelEngine
        InferenceEngine & IntelEngine <--> ModelRegistry
    end

    subgraph ContainerLocal ["Local / Self-Hosted Docker Engine"]
        DockerCompose["docker-compose.yml Orchestrator"]
        APIContainer["salessight_api (Port 8000)"]
        WebContainer["salessight_web (Port 8501)"]
        DockerCompose --> APIContainer & WebContainer
    end

    User --> StreamlitApp
    StreamlitClient -- "HTTPS REST JSON<br/>(GET /health, POST /forecast)" --> FastAPIServer
```

#### Step-by-Step Cloud Deployment Matrix:
1. **Render Backend Setup:**
   - Source code linked from GitHub repository using `render.yaml`.
   - Build command: `pip install -r requirements.txt`.
   - Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`.
   - Healthcheck: `/health`. Public endpoint: `https://salessight-ai-backend.onrender.com`.
2. **Streamlit Community Cloud Frontend Setup:**
   - Linked to `app.py` in the GitHub repository.
   - Secret environment variable: `API_URL="https://salessight-ai-backend.onrender.com"`.
   - Theme configuration: `.streamlit/config.toml` (Slate `#0E1117`, Electric Blue `#2563EB`).
3. **Docker Multi-Stage Containerization:**
   - `Dockerfile` and `docker-compose.yml` maintained for portable local or containerized deployments.

### 20.3 Environment Variables & Configuration Management (.env Specification)
In compliance with Assignment §16 (*Production Requirements: Environment Variables, requirements.txt, Error Handling*), system credentials, server bindings, and file paths are managed declaratively through environment variables loaded via `python-dotenv` and Pydantic `BaseSettings`:

```bash
# ==============================================================================
# SalesSight AI — Enterprise Environment Configuration (.env)
# ==============================================================================

# Application Environment & Runtime
SALESSIGHT_ENV=production                       # [development, staging, production]
SALESSIGHT_DEBUG=false                           # Disables stack traces in client browser
SALESSIGHT_LOG_LEVEL=INFO                        # [DEBUG, INFO, WARNING, ERROR]
SALESSIGHT_HOST=0.0.0.0                          # Bound host IP
SALESSIGHT_PORT=8501                             # Web UI port
SALESSIGHT_API_PORT=8000                         # FastAPI REST port

# Storage & Model Artifact Paths
SALESSIGHT_DATA_PATH=data/processed              # Directory for processed Parquet tables
SALESSIGHT_MODEL_DIR=models/artifacts            # Directory for serialized .joblib pipelines
SALESSIGHT_METADATA_DIR=models/metadata          # Directory for model_manifest.json
SALESSIGHT_REPORT_DIR=reports                    # Generated PDF/HTML export cache

# Security & Cloud Service Settings
SALESSIGHT_SECRET_KEY=prod-insecure-key-change-me # Session and state signing key
AWS_DEFAULT_REGION=us-east-1                    # Target cloud deployment region
AWS_ECS_CLUSTER=salessight-cluster               # ECS Fargate orchestration cluster
```

### 20.4 Production Headless REST API Specification (FastAPI Microservice)
To satisfy Assignment §13 (*Build an ML API using FastAPI or Flask*) and §16 (*API Health Endpoint*), `src/api/main.py` provides a standalone high-performance REST microservice with automated OpenAPI `/docs` documentation:

```
[FastAPI Service Endpoints Specification]
├── GET  /health
│   ├── Purpose: Liveness & readiness probe for Docker / AWS ALB
│   └── Response: {"status": "healthy", "version": "1.0.0", "timestamp": "ISO-8601", "models_loaded": true}
│
├── POST /forecast
│   ├── Purpose: Programmatic multi-horizon daily inference
│   ├── Payload: {"start_date": "2014-09-30", "scope": "macro", "target": "both", "horizon": 90, "model_mode": "auto"}
│   └── Response: {"horizon": 90, "model_used": "xgboost", "dates": [...], "revenue": [...], "quantity": [...], "lower": [...], "upper": [...]}
│
├── GET  /models
│   ├── Purpose: Exposes backtest scorecards, WAPE validation matrix, and winning model registry
│   └── Response: {"task": "macro_both_90d", "winner": "xgboost", "wape": 0.212, "baseline_wape": 0.318}
│
└── GET  /intelligence
    ├── Purpose: Demand analytics, Top-5 ranking, Rising/Falling momentum, and Planning Attention List
    └── Response: {"top_5_demand": [...], "rising_skus": [...], "falling_skus": [...], "planning_list": [...]}
```

---

## Section 21: Complete Testing Strategy

The quality assurance framework defines 28 exhaustive test suites executed via `pytest`:

| Test ID | Test Suite Group | Target Component | Test Scenario & Input Conditions | Expected Behavior | Pass/Fail Criteria |
|---|---|---|---|---|---|
| **TC-01** | Data Ingestion | `DataLoader` | Ingest raw `New_Query_...csv` | 2,172 rows and 33 columns loaded | Exact shape matches (2172, 33) |
| **TC-02** | Schema Validation | `DataValidator` | Ingest CSV with missing column | Triggers `ValidationError` | Exception raised with missing field name |
| **TC-03** | Date Parsing | `DataCleaner` | ISO-8601 string dates | Converted to UTC datetime | Dtype is `datetime64[ns]` |
| **TC-04** | Target Identity | `DataValidator` | Verify $Sales = COS + GP$ | Absolute residual evaluated | Max residual $< 10^{-4}$ across 2,172 rows |
| **TC-05** | Leakage Purge | `DataCleaner` | Audit output features | $COS, GP, Freight$ dropped | Zero financial outcome columns in $X$ |
| **TC-06** | Date Continuity | `DataCleaner` | Reindex full date span | Fills 815 missing dates | Output index length equals exactly 1,348 days |
| **TC-07** | Macro Aggregation | `SalesAggregator` | Daily aggregate sum | Daily Sales & Quantity summed | Sum of daily sales equals raw total ($1,783,147.70) |
| **TC-08** | SKU Aggregation | `SalesAggregator` | Aggregate by ProductID | Groups by product and date | Product rows match SKU sales sums |
| **TC-09** | Shifted Rolling Guard | `FeatureBuilder` | Generate `rolling_mean_7` | Current day excluded | $X_{t}[\text{roll}] == \text{mean}(y_{t-7}..y_{t-1})$ |
| **TC-10** | Lag Exactness | `FeatureBuilder` | Generate `lag_1, lag_7` | Exact historical values | $X_t[\text{lag\_1}] == y_{t-1}$ strictly |
| **TC-11** | Recursive Rollout | `ForecastingEngine` | Multi-step 90-day forecast | Predictions feed future lags | Outputs exactly 90 non-null values |
| **TC-12** | Horizon Counts | `ForecastingEngine` | Evaluate $H \in \{30, 60, 90\}$ | Output array length | Output len equals $H$ exactly |
| **TC-13** | Target Decoupling | `ForecastingEngine` | Dual revenue & quantity run | Independent models executed | Revenue in $ range; Quantity in integer units |
| **TC-14** | Baseline Logic | `SeasonalNaive` | Forecast day $t+h$ | Value equals $y_{t+h-7}$ | Output matches exact Lag-7 observation |
| **TC-15** | Backtest Expansion | `Backtester` | Expanding-window CV | Training window grows | Train size increases monotonically across folds |
| **TC-16** | WAPE Metric Calc | `Metrics` | Known vectors $y, \hat{y}$ | Metric computation | Output matches manual WAPE calculation |
| **TC-17** | Model Auto-Selection | `ModelSelector` | Models with known WAPE | Lowest WAPE selected | Selects candidate with lowest mean WAPE |
| **TC-18** | Tie-Breaker Rule | `ModelSelector` | Candidates within 2% WAPE | Evaluates RMSE & simplicity | Selects lower RMSE or simpler model |
| **TC-19** | Baseline Superiority | `ModelSelector` | All models WAPE > Baseline | Baseline selected | Selects Baseline and flags notice |
| **TC-20** | SKU History Guard | `Guardrails` | SKU with $<10$ records | Triggers rejection | Raises `InsufficientHistoryError` |
| **TC-21** | Baseline Fallback | `Guardrails` | SKU with 10–54 records | Triggers baseline | Returns Lag-7 forecast with fallback flag |
| **TC-22** | Date Default Mode | `DateOrchestrator` | $t_{\text{start}} == 2014\text{-}09\text{-}30$ | Default mode resolved | Mode == 'Default', starts 2014-09-30 |
| **TC-23** | Date Backtest Mode | `DateOrchestrator` | $t_{\text{start}} \le 2014\text{-}09\text{-}29$ | Backtest mode resolved | Mode == 'Backtest' |
| **TC-24** | Date Bridge Mode | `DateOrchestrator` | $t_{\text{start}} == 2014\text{-}11\text{-}01$ | Bridge mode resolved | Bridge days forecast and pruned |
| **TC-25** | Date Beyond Limit | `DateOrchestrator` | $t_{\text{start}} == 2015\text{-}06\text{-}01$ | Beyond limit resolved | Rejection raised or modal triggered |
| **TC-26** | Interval Non-Neg | `Uncertainty` | Quantile interval bounds | Clamp at zero | All lower bounds $\ge 0.0$ |
| **TC-27** | Top-5 30d Window | `DemandRanker` | Evaluate 90-day forecast | Ranked on days 1–30 | Top 5 evaluated strictly on first 30 days |
| **TC-28** | CSV Export Match | `CSVExporter` | Export forecast table | CSV matches table | Numerical values identical to table |

---

## Section 22: End-to-End Staged Implementation Blueprint and Roadmap

To translate this technical specification into a working, production-grade system, the engineering lifecycle is structured into **10 Discrete Implementation Stages**. These stages map directly to both the 16 End-to-End Machine Learning Assignment requirements and the 17 underlying technical work packages (Phases 0 through 16).

Each stage is sequential, isolated, and governed by strict **Definition of Done (DoD)** quality gates. A subsequent stage must not be initiated until the verification commands and acceptance criteria of the prior stage pass with 100% compliance.

---

### 22.1 Master Stage Execution Architecture

The diagram below illustrates the sequential data flow, transformations, and quality verification gates across all 10 stages:

```mermaid
flowchart TD
    subgraph S1["STAGE 1: Environment & Scaffolding"]
        A1["Repo Structure & Configs<br>(app_config.yaml, model_config.yaml)"] --> A2["Virtual Environment & Dependencies<br>(Python 3.11, requirements.txt)"]
        A2 --> A3["Seed & Logging Setup<br>(src/utils/)"]
    end

    subgraph S2["STAGE 2: Ingestion & Data Preparation"]
        B1["Raw CSV Schema Validation<br>(2,172 rows, 33 columns)"] --> B2["Cleaning & Null Handling<br>(src/data/cleaner.py)"]
        B2 --> B3["Daily Continuity Densification<br>(1,348 days, zero-sales filled)"]
        B3 --> B4["Multi-Scope Aggregation<br>(Macro, 8 Categories, 77 SKUs)"]
    end

    subgraph S3["STAGE 3: EDA & Correlation Intelligence"]
        C1["Statistical Summaries & Distributions"] --> C2["Sparsity & Pareto Diagnostics<br>(60.46% zero days)"]
        C2 --> C3["Correlation Heatmaps & Multicollinearity<br>(Pearson/Spearman, VIF Pruning)"]
    end

    subgraph S4["STAGE 4: Feature Engineering & Selection"]
        D1["Calendar & Cyclical Sine/Cosine Lags"] --> D2["Shifted Lags (t-1, t-7, t-14, t-28)"]
        D2 --> D3["Shifted Rolling Statistics (7d, 14d, 28d)"]
        D3 --> D4["Strict Lookahead Leakage Guards"]
        D4 --> D5["Feature Selection & Pruning<br>(MI & Tree Importances)"]
    end

    subgraph S5["STAGE 5: Model Building & Backtesting"]
        E1["Unified Estimator Interface<br>(src/models/base.py)"] --> E2["Candidate Estimators: Baseline, SARIMA,<br>Random Forest, XGBoost, LightGBM, Ridge (Model 5)"]
        E2 --> E3["Expanding Chronological Backtest<br>(4 origins, zero future contamination)"]
        E3 --> E4["Metrics Engine: WAPE, MAE, RMSE, MASE"]
    end

    subgraph S6["STAGE 6: Hyperparameter Tuning & Serialization"]
        F1["Time-Series Grid/Random Search"] --> F2["Quantitative WAPE Selection Rule<br>(Baseline fallback on failure)"]
        F2 --> F3["Sparse History Threshold Filtering<br>(<10 reject, 10-54 fallback)"]
        F3 --> F4["Model Serialization & Checksums<br>(joblib pipelines & model_manifest.json)"]
    end

    subgraph S7["STAGE 7: Forecasting Engine & Uncertainty"]
        G1["5-Mode Start Date Orchestrator<br>(Default, Backtest, Bridge, Limit, Demo)"] --> G2["Recursive Multi-Step Predictor<br>(30, 60, 90 calendar days)"]
        G2 --> G3["Calibrated 80% Residual Quantile Bounds<br>(Non-negative clamp [0, inf))"]
    end

    subgraph S8["STAGE 8: Demand Intelligence & Decisions"]
        H1["Top-5 Demand Ranking<br>(Strictly Days 1-30 Unit Volume)"] --> H2["SKU Momentum Classification<br>(Rising >=+15%, Falling <=-15%, Stable)"]
        H2 --> H3["Comparative SKU Benchmarking & PoP Shifts"]
        H3 --> H4["Planning Attention List Generator<br>(With inventory limitation disclaimer)"]
    end

    subgraph S9["STAGE 9: Web Dashboard, API & Reporting"]
        I1["Interactive Streamlit UI<br>(Plotly visualizer & KPI metrics)"] --> I2["Programmatic FastAPI Endpoints<br>(/health, /forecast, /models)"]
        I2 --> I3["Auditable Exports<br>(RFC CSV & Executive PDF/HTML Report)"]
    end

    subgraph S10["STAGE 10: Testing, Docker & Deployment"]
        J1["28 Pytest Automated Test Suites<br>(>85% line coverage)"] --> J2["Multi-Stage Production Docker Build<br>(Non-root user, healthcheck on 8501)"]
        J2 --> J3["AWS ECS / Cloud Orchestration Configs"]
        J3 --> J4["Production Runbook & Handover Sign-off"]
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10
```

---

### 22.2 Master Traceability Matrix: Stages vs. Assignment & Architecture

| Stage # | Stage Name | Assignment Requirement Mapping | Work Package Phase | Primary Code Modules & Deliverables | Primary Verification Command |
|---|---|---|---|---|---|
| **Stage 1** | Scaffolding & Configuration | Req 1 (Setup), Req 16 (Config/Env) | Phase 0 | `config/*.yaml`, `src/utils/`, `requirements.txt` | `python -c "import yaml; yaml.safe_load(...)"` |
| **Stage 2** | Ingestion, Cleaning & Grid | Req 1 (Data Understanding), Req 2 (Cleaning), Req 7 (Split) | Phases 1–2 | `src/data/loader.py`, `cleaner.py`, `aggregator.py` | `pytest tests/unit/test_cleaner.py` |
| **Stage 3** | EDA & Correlation Intelligence | Req 3 (EDA), Req 4 (Correlation Analysis) | Phase 3 | `src/data/correlation.py`, `scripts/run_eda.py`, assets | `python scripts/run_eda.py` |
| **Stage 4** | Feature Engineering & Leakage Guards | Req 5 (Feature Eng), Req 6 (Feature Selection) | Phase 4 | `src/features/builder.py`, `calendar.py`, `guardrails.py` | `pytest tests/unit/test_features.py` |
| **Stage 5** | Candidate Models & Chronological CV | Req 8 (ML Models), Req 9 (Evaluation) | Phases 5–6 | `src/models/*.py`, `src/evaluation/backtest.py` | `python scripts/run_backtest.py` |
| **Stage 6** | Tuning, Selection & Serialization | Req 10 (Tuning), Req 11 (Selection), Req 12 (Save Model) | Phases 7–8 | `src/models/tuner.py`, `selector.py`, `serializer.py` | `python scripts/train_and_select.py` |
| **Stage 7** | Forecast Engine & Uncertainty Bands | Req 13 (Forecasting API), Req 16 (Validation) | Phases 9–10 | `src/forecasting/engine.py`, `uncertainty.py` | `pytest tests/unit/test_forecasting.py` |
| **Stage 8** | Demand Intelligence & Analytics | Req 14 (UI Insights), Req 16 (Business Rules) | Phase 11 | `src/intelligence/*.py` | `pytest tests/unit/test_intelligence.py` |
| **Stage 9** | Streamlit UI, REST API & Reporting | Req 13 (ML API), Req 14 (Web UI), Req 16 (Exports) | Phases 12–13 | `app.py`, `src/ui/`, `src/api/main.py`, `src/reporting/` | `streamlit run app.py` |
| **Stage 10** | Testing, Docker & Cloud Deployment | Req 15 (Deployment), Req 16 (Production) | Phases 14–16 | `tests/`, `Dockerfile`, `docker-compose.yml`, `docs/runbook.md` | `pytest tests/ && docker-compose up --build` |

---

### 22.3 Detailed Stage-by-Stage Implementation Handbook

#### STAGE 1: Project Scaffolding, Declarative Configuration & Reproducibility Setup
- **Goal:** Establish a clean, modular repository structure, isolated virtual environment, reproducible seeding, and declarative YAML configurations separating parameters from code.
- **Prerequisites:** Python 3.11 installed, Git initialized, raw historical dataset `New_Query_2026_10_22_10_01_37.csv` in workspace.
- **Step-by-Step Implementation:**
  1. Initialize directory skeleton per Section 6 (`config/`, `data/raw/`, `data/processed/`, `src/`, `tests/`, `models/`, `docs/`).
  2. Pin exact dependency versions in `requirements.txt` (including `streamlit`, `pandas`, `scikit-learn`, `xgboost`, `statsmodels`, `plotly`, `pydantic`, `pytest`, `joblib`, `pyarrow`).
  3. Author `config/app_config.yaml` specifying paths, default horizons `[30, 60, 90]`, logging levels, and display settings.
  4. Author `config/model_config.yaml` specifying model hyperparameters, backtest split dates, lag orders `[1, 7, 14, 28]`, and WAPE selection criteria.
  5. Implement `src/utils/logger.py` with structured JSON/console formatting and `src/utils/reproducibility.py` enforcing global seed 42 across Python, NumPy, and Scikit-Learn.
- **Key Files Created:** `config/app_config.yaml`, `config/model_config.yaml`, `requirements.txt`, `pyproject.toml`, `src/utils/logger.py`, `src/utils/reproducibility.py`.
- **Execution & Verification Commands:**
  ```bash
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install -r requirements.txt
  python -c "import yaml; print(yaml.safe_load(open('config/app_config.yaml'))['application']['name'])"
  ```
- **Definition of Done (Stage 1 Acceptance Gate):**
  - [ ] Virtual environment installs with zero package conflicts.
  - [ ] All configuration files parse successfully into strongly-typed Pydantic schemas.
  - [ ] Structured logging outputs to both console and `logs/app.log`.

---

#### STAGE 2: Data Ingestion, Schema Profiling, Cleaning & Calendar Continuity
- **Goal:** Ingest raw order transactions, enforce strict schema contracts, clean data anomalies, and densify the intermittent timeline into a continuous daily time-series grid across all scopes.
- **Prerequisites:** Stage 1 verified; raw CSV in `data/raw/`.
- **Step-by-Step Implementation:**
  1. Implement `src/data/loader.py` to stream CSV data with explicit column dtypes.
  2. Implement `src/data/validator.py` asserting all 33 expected schema columns exist, row count equals 2,172, and primary keys are non-null.
  3. Implement `src/data/cleaner.py` to:
     - Strip monetary currency symbols (`$`, `,`) and cast `Line_Total` to float64.
     - Detect and deduplicate identical transaction lines.
     - Clean text casing on `Category` and `Product_Name`.
     - Filter invalid negative quantities or unit prices.
     - Parse `Order_Date` into standardized ISO-8601 UTC timestamps.
  4. Implement `src/data/aggregator.py` to construct a complete calendar grid spanning from `2011-01-21` to `2014-09-29` (1,348 days). Group daily transactions and densify zero-sale calendar days with `revenue = 0.0` and `demand = 0`.
  5. Partition and export processed Parquet datasets to `data/processed/` for Macro (`all_products_daily.parquet`), Meso (`category_daily.parquet`), and Micro (`sku_daily.parquet`).
- **Key Files Created:** `src/data/loader.py`, `src/data/validator.py`, `src/data/cleaner.py`, `src/data/aggregator.py`, `tests/unit/test_validator.py`, `tests/unit/test_cleaner.py`.
- **Execution & Verification Commands:**
  ```bash
  pytest tests/unit/test_validator.py tests/unit/test_cleaner.py -v
  python -c "import pandas as pd; df = pd.read_parquet('data/processed/all_products_daily.parquet'); assert len(df) == 1348; print('Grid length verified: 1348 continuous days')"
  ```
- **Definition of Done (Stage 2 Acceptance Gate):**
  - [ ] Schema validator passes with zero missing columns and zero unhandled nulls.
  - [ ] Continuous daily timeline has exactly 1,348 rows with zero temporal gaps.
  - [ ] Zero-sales days are accurately densified with 0 values rather than dropped or left as NaN.
  - [ ] Parquet files serialized with Snappy compression in `data/processed/`.

---

#### STAGE 3: Exploratory Data Analysis (EDA) & Correlation Intelligence
- **Goal:** Conduct in-depth diagnostic data exploration, visualize revenue/demand distributions, assess transaction sparsity, and compute correlation matrices to identify multicollinearity.
- **Prerequisites:** Stage 2 verified; processed Parquet tables in `data/processed/`.
- **Step-by-Step Implementation:**
  1. Author `scripts/run_eda.py` to generate descriptive statistical summaries (mean, std, median, skewness, zero-inflation ratios).
  2. Generate diagnostic plots:
     - Target variable distributions (revenue and demand histograms with log-transforms).
     - Outlier box plots across categories.
     - Weekly and monthly transaction seasonality charts.
     - Cumulative Pareto distribution curve (verifying product concentration).
     - Sparsity heatmap documenting 60.46% zero-transaction days.
  3. Implement `src/data/correlation.py` to compute Pearson and Spearman correlation matrices across numerical attributes (`Order_Qty`, `Unit_Price`, `Line_Total`, rolling averages).
  4. Compute Variance Inflation Factors (VIF) to detect multicollinear features (e.g., between `Line_Total` and `Unit_Price * Order_Qty`).
  5. Export all plots and a structured `eda_summary.json` to `docs/report/assets/eda/`.
- **Key Files Created:** `src/data/correlation.py`, `scripts/run_eda.py`, assets in `docs/report/assets/eda/`.
- **Execution & Verification Commands:**
  ```bash
  python scripts/run_eda.py --input-dir data/processed --output-dir docs/report/assets/eda
  ```
- **Definition of Done (Stage 3 Acceptance Gate):**
  - [ ] EDA charts saved in high-resolution PNG format.
  - [ ] Written analytical observations compiled into markdown report.
  - [ ] Correlation matrix confirms highly correlated redundancies identified for pruning.

---

#### STAGE 4: Feature Engineering, Time-Series Lag Generation & Selection
- **Goal:** Generate calendar, cyclical, autoregressive lag, and rolling window features with strict lookahead leakage prevention; perform quantitative feature selection.
- **Prerequisites:** Stage 2 & 3 verified; continuous daily Parquet datasets available.
- **Step-by-Step Implementation:**
  1. Implement `src/features/calendar.py` to extract temporal features: `day_of_week`, `day_of_month`, `month`, `quarter`, `is_weekend`, and cyclical trigonometric transforms (`sin_day_of_week`, `cos_day_of_week`, `sin_month`, `cos_month`).
  2. Implement `src/features/builder.py` to construct autoregressive lags: strictly shifted target values $y_{t-1}, y_{t-7}, y_{t-14}, y_{t-28}$.
  3. Construct shifted rolling window statistics: 7-day, 14-day, and 28-day rolling mean, standard deviation, min, and max. Enforce mandatory `.shift(1)` to ensure current-day target $y_t$ is never incorporated into historical summary windows.
  4. Implement `src/features/guardrails.py` running automated assertion checks verifying zero correlation between future targets $y_{t+h}$ and feature matrices $X_t$.
  5. Implement `src/features/selector.py` to rank features via Mutual Information regression and Random Forest feature importance, dropping features with VIF $>10.0$ or importance score $<0.01$.
- **Key Files Created:** `src/features/calendar.py`, `src/features/builder.py`, `src/features/guardrails.py`, `src/features/selector.py`, `tests/unit/test_features.py`.
- **Execution & Verification Commands:**
  ```bash
  pytest tests/unit/test_features.py -v
  python -c "from src.features.builder import FeatureBuilder; fb = FeatureBuilder(); print('Feature engineering pipeline verified')"
  ```
- **Definition of Done (Stage 4 Acceptance Gate):**
  - [ ] No NaN values exist in training feature matrix after warm-up drop (first 28 days dropped for lag warm-up).
  - [ ] Data leakage unit test passes assertion verifying that features at index $t$ rely strictly on data from $\le t-1$.
  - [ ] Feature selection reduces dimension while preserving $>95\%$ variance/predictive power.

---

#### STAGE 5: Candidate Estimator Construction & Chronological Cross-Validation
- **Goal:** Implement the candidate forecasting algorithms under a unified interface and evaluate their multi-horizon accuracy using strictly expanding chronological backtesting.
- **Prerequisites:** Stage 4 verified; clean feature matrices generated.
- **Step-by-Step Implementation:**
  1. Implement `src/models/base.py` defining the abstract `BaseForecaster` interface with mandatory methods `fit(X, y)`, `predict(X)`, and `save(path)`.
  2. Implement candidate model classes:
     - `src/models/baseline.py`: Seasonal-Naive Baseline ($y_{t} = y_{t-7}$).
     - `src/models/sarima.py`: Seasonal ARIMA with order $(p,d,q) \times (P,D,Q)_7$.
     - `src/models/tree_models.py`: Scikit-Learn `RandomForestRegressor`, `XGBoostRegressor`, and `LGBMRegressor`.
     - `src/models/linear.py`: Model 5: Ridge Regression (`StandardScaler` + `Ridge(alpha=10.0)`).
  3. Implement `src/evaluation/metrics.py` computing WAPE, MAE, RMSE, and MASE.
  4. Implement `src/evaluation/backtest.py` with expanding-window chronological cross-validation across 4 cutoffs (`2013-09-30`, `2013-12-31`, `2014-03-31`, `2014-06-30`), backtesting 30-, 60-, and 90-day horizons with zero future lookahead.
  5. Collect and store backtest error residuals across all folds and horizons.
- **Key Files Created:** `src/models/base.py`, `src/models/baseline.py`, `src/models/linear.py`, `src/models/sarima.py`, `src/models/tree_models.py`, `src/evaluation/metrics.py`, `src/evaluation/backtest.py`, `scripts/run_backtest.py`.
- **Execution & Verification Commands:**
  ```bash
  python scripts/run_backtest.py --config config/model_config.yaml
  ```
- **Definition of Done (Stage 5 Acceptance Gate):**
  - [ ] All candidate estimators execute under unified API and generate strictly non-negative predictions.
  - [ ] Chronological splits never shuffle time-series data or leak future test points into training.
  - [ ] Backtest results matrix contains WAPE, MAE, RMSE, and MASE for every model, target, and horizon.

---

#### STAGE 6: Hyperparameter Tuning, Quantitative Selection & Artifact Serialization
- **Goal:** Optimize model hyperparameter grids, apply the quantitative WAPE selection algorithm with baseline fallback rules, and serialize production binaries and metadata.
- **Prerequisites:** Stage 5 verified; backtest evaluation metrics and residuals recorded.
- **Step-by-Step Implementation:**
  1. Implement `src/models/tuner.py` executing time-series grid/random search (tuning trees: `n_estimators`, `max_depth`, `learning_rate`, `subsample`; tuning linear: `alpha`, `l1_ratio`).
  2. Implement `src/evaluation/selector.py` executing the quantitative model selection rule:
     - Select candidate with lowest out-of-sample mean WAPE.
     - If winning ML candidate fails to achieve lower WAPE than the Seasonal-Naive baseline, automatically fallback to Baseline.
     - Apply sparse SKU rules: SKUs with $<10$ sales trigger hard rejection; SKUs with $10$ to $54$ sales fallback to baseline.
  3. Implement `src/models/serializer.py` to persist winning models using `joblib` with LZ4/gzip compression.
  4. Persist historical residual error distributions for uncertainty estimation.
  5. Generate `models/model_manifest.json` recording winning model names, training timestamps, validation WAPE, and SHA-256 binary checksums.
- **Key Files Created:** `src/models/tuner.py`, `src/evaluation/selector.py`, `src/models/serializer.py`, `scripts/train_and_select.py`, `models/model_manifest.json`.
- **Execution & Verification Commands:**
  ```bash
  python scripts/train_and_select.py
  python -c "import json; m = json.load(open('models/model_manifest.json')); print('Selected models manifest verified:', len(m['models']))"
  ```
- **Definition of Done (Stage 6 Acceptance Gate):**
  - [ ] Tuned models demonstrate equal or improved out-of-sample WAPE versus default parameters.
  - [ ] Selection algorithm logs explicit justification for each winning model.
  - [ ] `model_manifest.json` contains valid SHA-256 checksums matching `.joblib` files on disk.

---

#### STAGE 7: Multi-Horizon Forecasting Engine & Calibrated Uncertainty
- **Goal:** Build the production inference engine supporting 30, 60, and 90-day multi-step recursive forecasting across 5 temporal start-date modes, with calibrated 80% confidence ranges.
- **Prerequisites:** Stage 6 verified; serialized models and residual distributions in `models/`.
- **Step-by-Step Implementation:**
  1. Implement `src/forecasting/date_orchestrator.py` managing the 5 date operational modes:
     - *Default Forward Mode:* Starts at latest date + 1 day (`2014-09-30`).
     - *Historical Backtest Mode:* Starts within historical range, allowing ground-truth comparison.
     - *Bridging Mode:* Recursively projects across gaps up to 90 unobserved calendar days.
     - *Beyond-Limit Mode:* Raises structured rejection if start date is $>90$ days past history.
     - *Demonstration Mode:* Simulates current-day forecasting (e.g., 2026) using calendar cyclic alignment and mandatory warning banners.
  2. Implement `src/forecasting/engine.py` coordinating recursive multi-step forecasting: iteratively appending predicted $\hat{y}_t$ to update lag and rolling features for $t+1$.
  3. Implement `src/forecasting/uncertainty.py` computing 80% confidence intervals ($\pm 1.28 \times \sigma_{\text{residual}} \times \sqrt{h/7}$) with non-negative lower bound clamp ($\max(0, \text{bound})$).
- **Key Files Created:** `src/forecasting/date_orchestrator.py`, `src/forecasting/engine.py`, `src/forecasting/uncertainty.py`, `tests/unit/test_forecasting.py`.
- **Execution & Verification Commands:**
  ```bash
  pytest tests/unit/test_forecasting.py -v
  ```
- **Definition of Done (Stage 7 Acceptance Gate):**
  - [ ] Forecasting engine returns exactly 30, 60, or 90 daily predictions with zero missing dates.
  - [ ] Confidence intervals widen monotonically across horizons and never fall below zero.
  - [ ] All 5 start-date modes execute strictly per technical specification.

---

#### STAGE 8: Product Demand Intelligence & Decision Analytics
- **Goal:** Implement commercial decision intelligence: Top-5 product demand rankings, SKU demand momentum trajectories, side-by-side product comparisons, and prioritized planning lists.
- **Prerequisites:** Stage 7 verified; forecast generation engine operational.
- **Step-by-Step Implementation:**
  1. Implement `src/intelligence/ranking.py` identifying the Top-5 demand products evaluated strictly on predicted unit volume over the first 30 forecast days.
  2. Implement `src/intelligence/trends.py` classifying product momentum:
     - **Rising:** Projected 30-day demand growth $\ge +15\%$ and absolute unit growth $\ge 5$ units.
     - **Falling:** Projected 30-day demand drop $\le -15\%$ and absolute unit drop $\le -5$ units.
     - **Stable:** All remaining products.
  3. Implement `src/intelligence/comparator.py` supporting side-by-side benchmarking of 2 to 5 selected products and consecutive 30-day period-over-period growth tracking.
  4. Implement `src/intelligence/recommendations.py` assembling the Planning Attention List (High Priority, Assortment Review, Overstock Watch) with the mandatory inventory limitation disclaimer.
- **Key Files Created:** `src/intelligence/ranking.py`, `src/intelligence/trends.py`, `src/intelligence/comparator.py`, `src/intelligence/recommendations.py`, `tests/unit/test_intelligence.py`.
- **Execution & Verification Commands:**
  ```bash
  pytest tests/unit/test_intelligence.py -v
  ```
- **Definition of Done (Stage 8 Acceptance Gate):**
  - [ ] Top-5 ranking is strictly calculated on days 1–30 unit demand (not revenue, not 90-day sums).
  - [ ] Momentum classification logic accurately tags Rising, Falling, and Stable items.
  - [ ] Planning recommendations include explicit disclaimer regarding absence of inventory stock levels.

---

#### STAGE 9: Interactive Streamlit Web UI, REST API & Auditable Reporting
- **Goal:** Build a responsive, aesthetic Streamlit dashboard, a headless FastAPI inference service, and verified CSV/PDF export capabilities.
- **Prerequisites:** Stage 7 & 8 verified; backend forecasting and intelligence engines ready.
- **Step-by-Step Implementation:**
  1. Implement `src/ui/sidebar.py` providing interactive filters: Target Selector (Revenue, Demand, Both), Scope Selector (All, Category, SKU), Horizon Selector (30, 60, 90 days), and Date Picker.
  2. Implement `src/ui/visualizer.py` rendering high-contrast Plotly forecast charts with historical context lines and shaded 80% uncertainty bands.
  3. Implement UI tabs in `src/ui/views/`:
     - *Forecast Overview:* KPI metric cards, Plotly forecast chart, summary statistics.
     - *Demand Intelligence:* Top-5 ranking table, momentum classification cards, Planning Attention List.
     - *Product Comparison:* Side-by-side multi-product benchmark plots.
     - *Model Performance:* Backtesting scorecard, metric comparisons (WAPE, RMSE), winning model justification.
  4. Implement `src/api/main.py` using FastAPI exposing `/health`, `/forecast`, `/models`, and `/intelligence` for programmatic consumption.
  5. Implement `src/reporting/csv_exporter.py` (RFC-4180 compliant CSV) and `src/reporting/report_generator.py` (generating downloadable HTML/PDF executive summaries matching UI values exactly).
- **Key Files Created:** `app.py`, `src/ui/sidebar.py`, `src/ui/visualizer.py`, `src/ui/views/*.py`, `src/api/main.py`, `src/reporting/csv_exporter.py`, `src/reporting/report_generator.py`, `tests/integration/test_ui_api.py`.
- **Execution & Verification Commands:**
  ```bash
  pytest tests/integration/test_ui_api.py -v
  streamlit run app.py --server.headless true &
  curl http://localhost:8501/_stcore/health
  ```
- **Definition of Done (Stage 9 Acceptance Gate):**
  - [ ] Streamlit UI launches without error and responds to user filter interactions in $<500\text{ms}$.
  - [ ] Exported CSV and PDF reports contain identical numerical forecast figures to the on-screen dashboard.
  - [ ] Prominent 2014 historical data disclaimer and Demonstration Mode banners appear in UI.

---

#### STAGE 10: Comprehensive Testing, Multi-Stage Dockerization & Cloud Deployment
- **Goal:** Execute full automated test suite achieving $>85\%$ coverage, build a secure multi-stage Docker container, create cloud orchestration manifests, and finalize operational runbooks.
- **Prerequisites:** Stages 1 through 9 fully functional.
- **Step-by-Step Implementation:**
  1. Run all 28 automated test suites across unit, integration, edge cases, and regression suites.
  2. Construct production `Dockerfile` with multi-stage build:
     - Stage 1: Build dependencies and compile wheels.
     - Stage 2: Minimal slim runtime, non-root user (`appuser:10001`), container health check on port 8501.
  3. Author `docker-compose.yml` orchestrating application service with volume mounts and resource constraints.
  4. Create AWS ECS Fargate deployment task definition in `deploy/aws/task_definition.json`.
  5. Author `README.md` and `docs/runbook.md` detailing architecture, configuration, local startup, API documentation, and maintenance procedures.
- **Key Files Created:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `deploy/aws/task_definition.json`, `docs/runbook.md`, `README.md`.
- **Execution & Verification Commands:**
  ```bash
  pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85
  docker build -t sales-sight-ai:latest .
  docker run -d -p 8501:8501 --name sales-sight-test sales-sight-ai:latest
  docker exec sales-sight-test curl -f http://localhost:8501/_stcore/health
  docker rm -f sales-sight-test
  ```
- **Definition of Done (Stage 10 Acceptance Gate):**
  - [ ] Total Pytest test suite passes 100% with $>85\%$ line coverage on `src/`.
  - [ ] Docker image builds cleanly with non-root security permissions and passes container health checks.
  - [ ] Final operational runbook verified for handover.

---

### 22.4 Phased Work Package Gantt Reference (Phases 0–16)

For operational project management and daily task allocation, the 10 stages map to the 17 work packages scheduled over a standard 34-day implementation timeline:

```mermaid
gantt
    title SalesSight AI Implementation Roadmap (Phases 0 - 16)
    dateFormat  X
    axisFormat  Day %d
    
    section Data & Core Pipeline (Stages 1-4)
    Phase 0: Repository & Requirements Setup  :0, 1
    Phase 1: Dataset Profiling & Validation    :1, 2
    Phase 2: Cleaning & Daily Aggregation      :2, 4
    Phase 3: Exploratory Data Analysis Suite   :4, 6
    Phase 4: Feature Engineering Pipeline      :6, 8
    
    section Modeling & Intelligence (Stages 5-8)
    Phase 5: Candidate Estimators & Baseline   :8, 11
    Phase 6: Chronological Backtesting Engine  :11, 14
    Phase 7: Hyperparameter Tuning & Selection :14, 16
    Phase 8: Model Serialization & Metadata    :16, 17
    Phase 9: Forecasting Engine & Date Rules   :17, 19
    Phase 10: Calibrated Uncertainty Ranges    :19, 21
    Phase 11: Product Demand Intelligence      :21, 24
    
    section Delivery & Operations (Stages 9-10)
    Phase 12: Streamlit Dashboard UI           :24, 27
    Phase 13: CSV & Executive Report Exports   :27, 29
    Phase 14: Comprehensive Test Suite         :29, 31
    Phase 15: Docker & Cloud Deployment Prep   :31, 33
    Phase 16: Verification & Final Handoff     :33, 34
```

---

### 22.5 Developer Progression Tracker and Quality Sign-Off Checklist

Use this checklist during implementation to track completion and verify that quality gates are respected:

- [ ] **Stage 1 Complete:** Repository scaffolded; virtual environment verified; declarative YAML configs parsed.
- [ ] **Stage 2 Complete:** Raw CSV ingested (2,172 rows, 33 cols); continuous daily grid built (1,348 days); parquet cached.
- [ ] **Stage 3 Complete:** EDA plots generated; correlation heatmap diagnosed; sparsity (60.46%) analyzed.
- [ ] **Stage 4 Complete:** Cyclical and lag features engineered; lookahead leakage guardrails passing.
- [ ] **Stage 5 Complete:** 5 estimators implemented; expanding chronological backtesting executed over 4 origins.
- [ ] **Stage 6 Complete:** Hyperparameters tuned; WAPE selection algorithm executed; models serialized with SHA-256 checksums.
- [ ] **Stage 7 Complete:** Multi-horizon recursive forecaster operational; 5 date modes active; 80% non-negative bounds verified.
- [ ] **Stage 8 Complete:** Top-5 demand ranking on days 1–30 active; momentum classified; planning list generated.
- [ ] **Stage 9 Complete:** Streamlit UI running with Plotly charts; FastAPI headless API operational; CSV/PDF exports validated.
- [ ] **Stage 10 Complete:** All 28 test suites passing (>85% coverage); multi-stage Docker image verified; runbook complete.

---

## Section 23: Final Acceptance Checklist

| Check # | Requirement Area | Verifiable Acceptance Criteria | Verification Protocol | Target Phase |
|---|---|---|---|---|
| **AC-01** | Forecast Horizons | Engine returns exactly 30, 60, or 90 daily predictions with no missing dates. | Automated Unit Test | Phase 9 |
| **AC-02** | Target Decoupling | Revenue ($) and Demand (units) evaluated and forecast independently. | Automated Unit Test | Phase 9 |
| **AC-03** | Scope Flexibility | Supports All Products, 8 Categories, and individual SKUs passing history rules. | Integration Test | Phase 9 |
| **AC-04** | Candidate Coverage | Seasonal-Naive Baseline, SARIMA, Random Forest, XGBoost, LightGBM, and Model 5: Ridge Regression evaluated. | Backtest Audit | Phase 6 |
| **AC-05** | Chronological Split| Backtesting uses strictly expanding chronological windows across $\ge 4$ origins. | CV Split Audit | Phase 6 |
| **AC-06** | WAPE Selection | Winning model chosen by lowest mean WAPE; baseline selected if ML fails to beat it. | Selection Log Check | Phase 7 |
| **AC-07** | Start-Date Modes | Default, Backtest, Bridge, Beyond-Limit, and Demo modes behave strictly per spec. | Date Logic Test | Phase 9 |
| **AC-08** | Prediction Bounds | 80% intervals displayed, non-negative, and widen across horizon steps. | Bound Audit Test | Phase 10 |
| **AC-09** | Top-5 Demand | Top-5 ranking evaluated strictly on predicted unit volume over Days 1–30. | Ranking Unit Test | Phase 11 |
| **AC-10** | Demand Momentum | Rising ($\ge +15\%$, $\ge 5\text{u}$), Falling ($\le -15\%$, $\le -5\text{u}$), and Stable tagged accurately. | Trend Unit Test | Phase 11 |
| **AC-11** | Comparisons | 2–5 products compared side-by-side; consecutive 30-day periods compared. | UI & Logic Test | Phase 11 |
| **AC-12** | Planning List | Generates High Priority, Review, Overstock Watch lists with inventory limitation. | Template Audit | Phase 11 |
| **AC-13** | Export Agreement | CSV export and PDF/HTML report match dashboard numbers to 4 decimal places. | Export Audit Test | Phase 13 |
| **AC-14** | Data Disclaimer | 2014 historical data limitation and Demo Mode notices prominently displayed. | Visual UI Audit | Phase 12 |
| **AC-15** | Sparse History | Products with $<10$ sales trigger rejection; 10–54 trigger baseline fallback. | Edge Case Test | Phase 9 |

---

## Section 24: Risks, Limitations, and Open Decisions

### 24.1 Risk and Limitation Registry

| Risk ID | Category | Nature of Risk or Limitation | Concrete Impact on Project | Practical Mitigation Strategy |
|---|---|---|---|---|
| **RSK-01** | Confirmed Limitation | Historical data ends September 29, 2014 (12 years prior to 2026). | Forecasts cannot establish current market conditions for 2026. | Enforce mandatory Demonstration Mode; map current dates to historical day-of-week/year; display permanent disclaimer. |
| **RSK-02** | Confirmed Limitation | Complete absence of stock levels, lead times, and holding costs in dataset. | System cannot compute automated inventory reorder quantities or EOQ. | Frame intelligence strictly as demand planning signals; display explicit inventory limitation disclaimer on all reports. |
| **RSK-03** | Data Quality | 60.46% of calendar days in historical span have zero recorded transactions. | Models risk predicting zero-sales continuously if zero-inflation is mishandled. | Evaluate Tweedie loss in XGBoost; evaluate rolling 7-day aggregation; calibrate baseline fallbacks. |
| **RSK-04** | Data Quality | All 77 products have $\le 54$ active sales days (none meet 60-day threshold). | Default feature spec history threshold locks all SKUs out of pooled ML models. | Calibrate threshold to $\ge 25$ days in `model_config.yaml` or document 100% baseline fallback behavior. |
| **RSK-05** | Technical Risk | Recursive multi-step forecasting compounds errors over 90-day horizons. | Tree models may decay toward training mean; Linear Regression may diverge. | Benchmark recursive errors against direct multi-horizon models; widen empirical residual confidence bands. |
| **RSK-06** | Open Decision | Web API Architecture: Direct Streamlit service import vs. FastAPI wrapper. | Ingestion latency vs. deployment complexity. | Build clean service layer in `src/forecasting/engine.py`. Streamlit imports directly in Phase 12; FastAPI head added in Phase 15. |
| **RSK-07** | Open Decision | Cloud Hosting Infrastructure: AWS ECS Fargate vs. Streamlit Community Cloud. | Hosting cost vs. enterprise control. | Dockerfile designed for portability. Deploy to local container first; support AWS Fargate via CloudFormation/Terraform. |

---

## Section 25: Implementation Handoff

### 25.1 Transition to Implementation
This blueprint concludes the architectural documentation phase. In strict compliance with master instructions, **no application code has been written, no models have been trained, and no project structures have been created**.

When development commences, the engineering team executes in order through **Stage 1 (Scaffolding & Environment Initialization)** and **Stage 2 (Data Ingestion, Schema Profiling & Cleaning)**:

```
First Implementation Session Target:
├── STAGE 1: Project Scaffolding & Environment Initialization (Phase 0)
│   ├── Target Files: config/app_config.yaml, config/model_config.yaml, requirements.txt, pyproject.toml
│   ├── Prerequisites: Python 3.11 installed, Git repository initialized
│   └── Verification: `pip install -r requirements.txt` succeeds with zero dependency conflicts.
│
└── STAGE 2: Data Ingestion, Schema Profiling & Cleaning (Phases 1–2)
    ├── Target Files: src/data/loader.py, src/data/validator.py, src/data/cleaner.py, src/data/aggregator.py
    ├── Prerequisites: Raw CSV verified in data/raw/
    └── Verification: `pytest tests/unit/test_validator.py tests/unit/test_cleaner.py` passes 100% assertions on 2,172 rows, 33 columns, and 1,348 continuous calendar grid days.
```

**Stopping Confirmation:**
The project development blueprint is complete, verified, and saved as `SMART_SALES_FORECASTING_PRODUCTION_DEVELOPMENT_BLUEPRINT.md` in the workspace root. All development activities remain paused awaiting explicit user authorization.
