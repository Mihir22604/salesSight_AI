# SalesSight AI — Step 5: Comprehensive Feature Engineering Report

**Document Title:** Step 5: Comprehensive Time-Series Feature Engineering, Cyclical Encodings, Shifted Rolling Statistics, Commercial Ratios, and Leakage Prevention Report  
**Project Name:** Smart Sales Forecasting and Product Demand Intelligence System (SalesSight AI)  
**Target Generated Feature Matrices:**  
- `data/processed/macro_features.parquet` ($N = 1,320$ daily observations × 42 columns, 38 engineered predictors)  
- `data/processed/macro_features.csv` ($N = 1,320$ daily observations × 42 columns)  
- `data/processed/category_features.parquet` ($N = 10,560$ category-day observations × 50 columns)  
- `data/processed/category_features.csv` ($N = 10,560$ category-day observations × 50 columns)  
**Execution Timestamp:** 2026-10-23  
**Status:** 100% Complete & Formally Verified via Unit Tests (21/21 Passed)  
**Authoritative Reference:** `Assignment_mini_project (1).docx` (§5: *Feature Engineering*) & Master Blueprint Section 6 / Section 9  

---

## 1. Executive Summary & Objective

In accordance with **Assignment Step 5: Feature Engineering** and Section 9 of the Master Production Blueprint, this report documents the transformation of continuous daily sales time series into supervised tabular feature matrices.

Following the empirical findings of **Step 3 (EDA)** and **Step 4 (Correlation Analysis)**, the primary objectives of Step 5 are to:
1. **Synthesize Domain-Engineered Feature Spaces ($d = 38$ Predictors):** Construct a multi-faceted feature representation spanning calendar cycles, autoregressive history, shifted moving windows, commercial price/momentum ratios, categorical discretizations, and nonlinear interactions.
2. **Implement Decoupled Dual-Target Regressors:** Build independent lag and rolling feature representations for continuous dollar turnover (`Sales`) and discrete unit count (`Quantity`), satisfying the empirical decoupling proven in Steps 3 and 4.
3. **Enforce Strict Temporal Lookahead Guards:** Implement `.shift(1)` on all rolling statistics and autoregressive lags, mathematically eliminating data leakage where observation $t$ could contaminate predictor features.
4. **Construct Harmonic Cyclical Encodings:** Formulate trigonometric sine and cosine transformations for day-of-week ($m=7$) and month-of-year ($m=12$), resolving boundary discontinuities between calendar periods.
5. **Formulate High-Yield Commercial Ratios:** Synthesize the Average Selling Price (ASP) proxy, the Trajectory Momentum Ratio ($\text{MA}_7 / \text{MA}_{28}$), and the Volatility Expansion Ratio ($\text{Std}_7 / \text{Std}_{28}$) to isolate trend and regime shifts without scale inflation.
6. **Provide Model-Specific Scaling & Transformation Pipelines:** Formulate logarithmic transformations $\log(1 + y)$ and strict out-of-fold `StandardScaler` transformations for regularized linear regressors, while preserving unscaled dollar/unit values for decision tree ensembles.

---

## 2. Tabular Feature Space Architecture

The feature pipeline transforms the continuous daily series ($N = 1,348$ days) into a supervised learning matrix. Because the longest autoregressive lag and rolling window require a 28-day historical baseline, the first 28 calendar days (2011-01-21 to 2011-02-17) serve as the initialization warmup buffer.

```
[Continuous Daily Time Series: 1,348 Days]
         │
         ├── Warmup Initialization Buffer: Days 1 to 28 (Dropped to ensure zero NaNs)
         │
         ▼
[Supervised Macro Feature Matrix: 1,320 Rows × 42 Columns]
 ├── Target 1: Sales ($ Revenue, Continuous)
 ├── Target 2: Quantity (Physical Units, Discrete)
 ├── Reference Metadata: Date, OrdersCount
 └── 38 Domain-Engineered Predictor Features across 6 Specialized Families
```

### 2.1 Distribution of the 38 Engineered Features by Family

| Feature Family | Feature Count | Component Column Names | Primary Predictive Role |
|---|---|---|---|
| **1. Calendar & Cyclical** | 9 features | `day_of_month`, `day_of_week`, `month`, `quarter`, `is_weekend`, `sin_dow`, `cos_dow`, `sin_month`, `cos_month` | Captures intra-week shopping patterns, quarterly budget cadences, and annual seasonality. |
| **2. Autoregressive Lags** | 8 features | `lag_1_rev`, `lag_7_rev`, `lag_14_rev`, `lag_28_rev`, `lag_1_qty`, `lag_7_qty`, `lag_14_qty`, `lag_28_qty` | Encodes short-term persistence, weekly cycles ($m=7$), bi-weekly cycles, and monthly baselines. |
| **3. Shifted Rolling Windows** | 13 features | `rolling_mean_7_rev`, `rolling_mean_14_rev`, `rolling_mean_28_rev`, `rolling_std_7_rev`, `rolling_std_28_rev`, `rolling_min_7_rev`, `rolling_max_7_rev`, `rolling_max_28_rev`, `rolling_mean_7_qty`, `rolling_mean_14_qty`, `rolling_mean_28_qty`, `rolling_std_7_qty`, `rolling_std_28_qty` | Smooths 60.46% intermittent zero days into continuous demand intensity and volatility signals. |
| **4. Commercial Ratios** | 3 features | `asp_proxy_t1`, `mom_ratio_7_28`, `vol_ratio_7_28` | Normalizes scale to measure pricing power, growth trajectory, and demand turbulence. |
| **5. Discretized Tiers** | 2 features | `volume_tier`, `price_tier` | Ordinal categorical bins separating wholesale bulk orders from retail transactions. |
| **6. Nonlinear Interactions** | 3 features | `dow_month_cross`, `momentum_asp_interaction`, `weekend_momentum_interaction` | Cross-feature terms capturing compound seasonal and price-expansion dynamics. |

---

## 3. Harmonic Cyclical Trigonometric Encodings

Standard integer calendar representations introduce artificial discontinuities. For instance, in day-of-week integer encoding ($0 = \text{Monday}, 6 = \text{Sunday}$), Sunday and Monday have an absolute distance of $|6 - 0| = 6$, despite being consecutive calendar days.

To preserve circular temporal topology, trigonometric transformations map integer dates onto unit circle coordinates:

### 3.1 Mathematical Formulations:

1. **Intra-Week Harmonic Coordinates ($m = 7$):**
   $$\sin_{\text{dow}} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \cos_{\text{dow}} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$
2. **Annual Intra-Year Harmonic Coordinates ($m = 12$):**
   $$\sin_{\text{month}} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \cos_{\text{month}} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

### 3.2 Formal Verification of Circular Topology:
- **Trigonometric Unit Circle Identity:** For every row in the dataset, $\sin^2(\theta) + \cos^2(\theta) \equiv 1.0000 \pm 10^{-5}$.
- **Euclidean Continuity:**
  $$\text{Dist}(\text{Sunday}, \text{Monday}) = \sqrt{(\sin(6) - \sin(0))^2 + (\cos(6) - \cos(0))^2} = 2\sin\left(\frac{\pi}{7}\right) \approx 0.8678$$
  The circular distance between Sunday and Monday is identical to the distance between any other adjacent pair of days (e.g., Monday and Tuesday), resolving regression edge discontinuities.

---

## 4. Shifted Autoregressive Lags & Rolling Statistics

### 4.1 Strict Lookahead Leakage Prevention via `.shift(1)`
In operational time-series forecasting, predictions for horizon $t$ must be made using strictly historical data observed prior to $t$. If an engineer computes `df['Sales'].rolling(7).mean()` directly, pandas aligns the window with right-edge inclusion:

$$\text{rolling\_mean}_7^{\text{leaked}}(t) = \frac{1}{7} \sum_{i=0}^{6} y_{t-i} = \frac{1}{7} \left( \mathbf{y_t} + y_{t-1} + \dots + y_{t-6} \right)$$

This constitutes **catastrophic instantaneous target leakage**, allowing models to "cheat" during training by accessing current ground truth $y_t$.

**Production Enforcement:** All rolling calculations are strictly preceded by `.shift(1)`:

$$\text{rolling\_mean}_7(t) = \frac{1}{7} \sum_{i=1}^{7} y_{t-i} \quad \left(\text{implemented as } \texttt{df['target'].shift(1).rolling(7).mean()}\right)$$

### 4.2 Automated Lookahead Verification Protocol:
Every lag and rolling column was audited by [`LeakageGuard.verify_lookahead_integrity`](file:///c:/Users/mihir/Desktop/SalesSight%20AI/src/features.py):
1. **Lag Consistency Check:** Verified that $\text{lag}_k(t) \equiv y(t-k)$ with absolute numerical tolerance $< 10^{-5}$.
2. **Contemporaneous Correlation Assertion:** Asserted that no feature has instantaneous correlation $|r| = 1.0$ with target $y_t$.
3. **Corrupted Leakage Unit Test:** Confirmed that intentional lookahead contamination triggers an immediate fatal exception in the automated test suite.

---

## 5. Commercial Economic Ratios & Trajectory Momentum

To provide scale-free features that remain robust across different product price tiers and macroeconomic shifts, three economic ratios were synthesized:

### 5.1 Historical Average Selling Price (ASP) Proxy
$$\text{asp\_proxy}_{t-1} = \frac{\text{lag\_1\_rev}}{\max(1, \text{lag\_1\_qty})}$$
- **Role:** Measures the realized dollar revenue per physical unit sold on the preceding active day.
- **Summary Metrics:** Mean = $\$14.78$, Median = $\$0.00$ (reflecting zero-sales days), Standard Deviation = $\$34.42$.
- **Interpretation:** Distinguishes whether demand surges are driven by expensive luxury apparel (high ASP) or high-volume economy commodities (low ASP).

### 5.2 Trajectory Momentum Ratio ($\text{MA}_7 / \text{MA}_{28}$)
$$\text{mom\_ratio}_{7,28}(t) = \frac{\text{rolling\_mean\_7\_rev}(t)}{\max(0.01, \text{rolling\_mean\_28\_rev}(t))}$$
- **Role:** Compares the short-term 7-day velocity against the long-term 28-day baseline trend.
- **Regime Classification:**
  - $\text{mom\_ratio} > 1.15$: **Expansionary Surge Regime** (Short-term velocity $>15\%$ above baseline) $\to$ **36.52% of calendar days**.
  - $0.85 \le \text{mom\_ratio} \le 1.15$: **Stable Demand Equilibrium** $\to$ **11.43% of calendar days**.
  - $\text{mom\_ratio} < 0.85$: **Contractionary / Quiescent Regime** (Short-term velocity $<15\%$ below baseline) $\to$ **52.05% of calendar days**.
- **Diagnostic Power:** Solves the multicollinearity problem identified in Step 4. While `rolling_mean_7` and `rolling_mean_28` had severe collinearity ($r = 0.8925$), their ratio has an auxiliary $R^2 < 0.30$ ($VIF < 1.5$), providing clean, non-collinear growth acceleration signal.

### 5.3 Volatility Expansion Ratio ($\text{Std}_7 / \text{Std}_{28}$)
$$\text{vol\_ratio}_{7,28}(t) = \frac{\text{rolling\_std\_7\_rev}(t)}{\max(0.01, \text{rolling\_std\_28\_rev}(t))}$$
- **Role:** Identifies whether current demand variance is clustering or stabilizing. Mean = $0.81$, Median = $0.68$.

---

## 6. Discretization & Binning Protocols

To support piecewise linear approximations and non-parametric tree partitioning, continuous variables were discretized into business-aligned ordinal tiers:

### 6.1 Order Volume Discretization Tiers (`volume_tier`)
Derived from the 7-day rolling unit volume (`rolling_mean_7_qty`):
- **Tier 0: Small Order Velocity (0–10 units/day):** **27.58%** of days (quiescent retail baseline).
- **Tier 1: Medium Order Velocity (11–40 units/day):** **33.33%** of days (standard commercial flow).
- **Tier 2: Bulk Wholesale Velocity (>40 units/day):** **39.09%** of days (institutional purchasing surge).

### 6.2 Price Level Discretization Tiers (`price_tier`)
Derived from the historical Average Selling Price proxy (`asp_proxy_t1`):
- **Tier 0: Economy Price Tier (<$20/unit):** **74.55%** of days (mass-market products).
- **Tier 1: Standard Price Tier ($20–$50/unit):** **19.02%** of days (mid-tier sportswear & footwear).
- **Tier 2: Premium Price Tier (>$50/unit):** **6.44%** of days (luxury outerwear & high-end apparel).

---

## 7. Nonlinear Interaction Features

To allow linear models to capture compound physical relationships without polynomial basis expansion:
1. **`dow_month_cross` ($\text{day\_of\_week} \times \text{month}$):** Captures seasonal holiday shopping behavior that varies across specific days of the week (e.g., December weekend spikes).
2. **`momentum_asp_interaction` ($\text{mom\_ratio}_{7,28} \times \text{asp\_proxy}_{t-1}$):** Multiplies growth momentum by dollar price intensity. An acceleration in momentum for high-priced goods signals massive gross revenue expansion.
3. **`weekend_momentum_interaction` ($\text{is\_weekend} \times \text{mom\_ratio}_{7,28}$):** Isolates weekend surge dynamics under expanding trend regimes.

---

## 8. Transformation & Scaling Policies

### 8.1 Logarithmic Target Transformation
To mitigate the severe positive skewness ($+10.27$) documented in Step 3, [`FeatureTransformer`](file:///c:/Users/mihir/Desktop/SalesSight%20AI/src/features.py) provides exact forward and inverse log transformations:

$$y_{\log} = \log(1 + y), \quad \hat{y} = \max(0, \exp(\hat{y}_{\log}) - 1)$$

- **Mathematical Proof:** Log-transformed sales transforms exponential revenue spikes into quasi-normal distributions, preventing squared-error loss functions in Ridge and XGBoost from being dominated by extreme outlier transactions.
- **Zero Invariance:** For zero-sales days ($y = 0$), $\log(1 + 0) = 0$, preserving exact structural zero sparsity.

### 8.2 Standard Feature Scaling (`StandardScaler`)
- **Linear Models (Ridge / Lasso):** Lags, rolling averages, and interaction terms are standardized to zero mean and unit variance ($Z = \frac{X - \mu}{\sigma}$). Scaler parameters $(\mu, \sigma)$ are fitted strictly on training backtest folds.
- **Tree Models (Random Forest / XGBoost):** Feature scaling is omitted. Decision trees split on single-feature rank order; scaling has zero mathematical impact on tree topology.

---

## 9. Automated Verification & Quality Sign-Off

The Step 5 feature engineering pipeline was tested and validated via [`tests/unit/test_step5_features.py`](file:///c:/Users/mihir/Desktop/SalesSight%20AI/tests/unit/test_step5_features.py). The entire project test suite was executed:

```
tests/unit/test_step1_and_step2.py::test_step1_loader_and_schema PASSED  [  4%]
tests/unit/test_step1_and_step2.py::test_step1_understanding_profiler PASSED [  9%]
tests/unit/test_step1_and_step2.py::test_step2_cleaner_leakage_and_types PASSED [ 14%]
tests/unit/test_step1_and_step2.py::test_step2_aggregator_macro_continuity PASSED [ 19%]
tests/unit/test_step1_and_step2.py::test_step2_aggregator_category_grid PASSED [ 23%]
tests/unit/test_step3_eda.py::test_eda_target_distributions PASSED       [ 28%]
tests/unit/test_step3_eda.py::test_eda_category_performance PASSED       [ 33%]
tests/unit/test_step3_eda.py::test_eda_pareto_concentration PASSED       [ 38%]
tests/unit/test_step3_eda.py::test_eda_temporal_dynamics PASSED          [ 42%]
tests/unit/test_step3_eda.py::test_eda_visualizer_renders_figures PASSED [ 47%]
tests/unit/test_step4_correlation.py::test_transaction_correlations PASSED [ 52%]
tests/unit/test_step4_correlation.py::test_price_elasticity_economic_relationship PASSED [ 57%]
tests/unit/test_step4_correlation.py::test_vif_calculation_and_thresholding PASSED [ 61%]
tests/unit/test_step4_correlation.py::test_collinear_pair_screening PASSED [ 66%]
tests/unit/test_step4_correlation.py::test_correlation_visualizer_renders_figures PASSED [ 71%]
tests/unit/test_step5_features.py::test_macro_feature_matrix_generation PASSED [ 76%]
tests/unit/test_step5_features.py::test_cyclical_harmonic_encodings PASSED [ 80%]
tests/unit/test_step5_features.py::test_leakage_guardrail_integrity PASSED [ 85%]
tests/unit/test_step5_features.py::test_commercial_ratios_and_binning PASSED [ 90%]
tests/unit/test_step5_features.py::test_feature_transformer_scaling_and_log PASSED [ 95%]
tests/unit/test_step5_features.py::test_feature_visualizer_renders_figures PASSED [100%]

======================= 21 passed in 12.82s ========================
```

### Production Assets Ready in `report_stage/` and `data/processed/`:
- `data/processed/macro_features.parquet` (1,320 rows × 42 columns, 38 engineered features)
- `data/processed/macro_features.csv`
- `data/processed/category_features.parquet` (10,560 rows × 50 columns)
- `data/processed/category_features.csv`
- `report_stage/STEP_5_FEATURE_ENGINEERING_REPORT.md` (This document)
- `report_stage/step5_feature_engineering_metrics.json` (Complete machine-readable metrics)
- `report_stage/assets/step5_features_distributions.png` (Ratio distributions)
- `report_stage/assets/step5_features_cyclical_encoding.png` (Trigonometric circular harmonics)
- `report_stage/assets/step5_features_binning_discretization.png` (Order volume & price tier distributions)
- `report_stage/assets/step5_features_rolling_momentum.png` (Trajectory momentum tracking over 1,320 days)

**Step 5 Feature Engineering is formally complete and signed off. The system is ready to advance to Step 6: Feature Selection.**
