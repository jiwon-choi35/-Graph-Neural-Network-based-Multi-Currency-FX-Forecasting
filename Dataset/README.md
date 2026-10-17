# Dataset

Raw monthly sources collected for the model, January 2011 to December 2025. The final model input built from these is in [B-MTGNN/data](../B-MTGNN/data) (`data.csv`, 33 variables).

| File | Contents |
|---|---|
| `data_3-feature_only.csv` | Three target series: US Trade Weighted Dollar Index, USD/KRW, USD/JPY |
| `data_3-feature.csv` | Targets plus derived FX features: monthly returns, lagged returns, spreads between currencies, rolling correlations |
| `fred_data.csv` | Macroeconomic indicators for US, KR and JP: interest rate, GDP, CPI, employment, unemployment, trade balance |
