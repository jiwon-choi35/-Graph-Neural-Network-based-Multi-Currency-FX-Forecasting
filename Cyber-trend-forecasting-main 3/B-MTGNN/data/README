# Dataset Metadata

## Overview

This directory contains all data files used by the Bayesian MTGNN model for forecasting exchange rates and economic indicators. The dataset spans from January 2011 to December 2025 on a monthly basis (180 months total).

## Files Description

### Raw Data Files

- **data.csv** - Raw time series data for all 33 economic variables
- **data.txt** - Text format of raw data
- **graph.csv** - Graph adjacency/correlation matrix defining relationships between variables

### Preprocessed/Smoothed Data

- **sm_data.csv** - Smoothed data using exponential smoothing (CSV format)
- **sm_data.txt** - Smoothed data (text format)
- **sm_data_g.csv** - Graph-based smoothed data (CSV format)
- **sm_data_g.txt** - Graph-based smoothed data (text format)

### Historical Data & Reference

- **Collecting_date/** - Subdirectory containing historical data processing information
  - **fred_data.csv** - Raw data from FRED (Federal Reserve Economic Data)
  - **data_3-feature.csv** - Data with 3-feature selection
  - **data_3-feature_only.csv** - Only 3 selected features

## Feature Format

Each variable follows the naming convention below:

### Exchange Rates

1. **jp_fx** - Japanese Yen exchange rate trend (USD/JPY)
2. **kr_fx** - Korean Won exchange rate trend (USD/KRW)
3. **us_Trade Weighted Dollar Index** - US Dollar Trade Weighted Index

### Return/Spread Indicators

4. **jp_ret** / **kr_ret** / **us_ret** - Monthly returns for each country
5. **jp_ret_lag1** / **kr_ret_lag1** / **us_ret_lag1** - Lagged returns (1-month lag)
6. **jp_us_spread** / **kr_us_spread** / **kr_jp_spread** - Interest rate spreads between country pairs
7. **corr_us_jp** / **corr_us_kr** / **corr_kr_jp** - Rolling correlation between exchange rates

### Economic Indicators

#### CPI (Consumer Price Index)

8. **JP_CPI** / **KR_CPI** / **US_CPI** - Monthly inflation rate proxy

#### GDP (Gross Domestic Product)

9. **JP_GDP** / **KR_GDP** / **US_GDP** - Quarterly economic growth (interpolated to monthly)

#### Interest Rates

10. **JP_Interest_Rate** / **KR_Interest_Rate** / **US_Interest_Rate** - Policy rates for each country

#### Employment

11. **JP_Employment** / **KR_Employment** / **US_Employment** - Employment levels
12. **JP_Unemployment** / **KR_Unemployment** / **US_Unemployment** - Unemployment rates

#### Trade Indicators

13. **JP_Trade_Balance** / **KR_Trade_Balance** / **US_Trade_Balance** - Monthly trade balances

## Data Characteristics

### Time Coverage

- **Period**: January 2011 - December 2025
- **Frequency**: Monthly (180 observations)
- **Timeline**:
  - Historical: 2011-01 to 2025-12 (180 months)
  - Forecast: 2026-01 to 2026-12 (12 months)

### Data Quality

- **Variables**: 33 economic indicators across 3 countries
- **Missing Values**: Handled through forward-fill and interpolation during preprocessing
- **Normalization**: All variables normalized to [0, 1] range for model training

## Data Preprocessing

### Smoothing Methods

The model uses **exponential smoothing** to reduce noise:

- **sm_data.csv**: Double exponential smoothing applied to reduce short-term fluctuations
- Parameters: α (alpha) adjusted based on variable volatility

### Graph Construction

- **graph.csv** contains the correlation matrix between all variables
- Edges defined between variables with correlation > 0.3 (threshold adjustable)
- Enables graph neural network to model inter-variable dependencies

## Usage in Model

### Input Format

The model reads from **sm_data.csv** as the primary input:

```python
df = pd.read_csv('sm_data.csv')
```

### Normalization

Variables are normalized per feature:

```
normalized_value = (raw_value - min) / (max - min)
```

### Sequence Generation

- **Input Sequence Length**: 24 months (2 years historical context)
- **Forecast Horizon**: 12 months (2026)
- Multiple overlapping sequences created for training

## Data Collection Sources

All data aggregated from multiple sources:

- **Exchange Rates**: FRED, Yahoo Finance, Bank of Korea, Bank of Japan
- **Economic Indicators**: OECD, National Central Banks, IMF
- **Macroeconomic Data**: FRED, World Bank, Trading Economics

## Important Notes

1. **Date Format**: All timestamps are in YYYY-MM format (e.g., 2011-01)
2. **Missing Data**: Handled using forward-fill for continuity
3. **Stationarity**: Returns and spreads are stationary; levels tested for cointegration
4. **Scaling**: Normalized per feature for neural network stability
5. **Train/Validation/Test Split**: 60% / 20% / 20% of the 180 months

## File Size Reference

- data.csv: ~180 rows × 33 columns
- sm_data.csv: Same dimensions, post-smoothing
- Data type: Float32 for efficiency

## Related Scripts

For data preparation and smoothing:

- Refer to `forecast.py` for how data is loaded and normalized
- Refer to `train_test.py` for train/validation/test split logic
