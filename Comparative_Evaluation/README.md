# Comparative Evaluation

Saved results for comparing B-MTGNN with the baseline models on the 2025 test year. This folder contains predictions and metrics only. The baseline training code is not included, so the numbers here are the reference values, not something rerun from this repository.

## Evaluation setting

- Train: 2011-01 ~ 2023-12, validation: 2024-01 ~ 2024-12, test: 2025-01 ~ 2025-12
- Monthly one-step prediction with actual updates: the prediction for month t uses actual values up to t-1
- Targets: US Trade Weighted Dollar Index (US), USD/KRW (KR), USD/JPY (JP), evaluated on the smoothed series in `B-MTGNN/data/sm_data.csv`
- Model and setting selection uses the 2024 validation set only
- Metrics: RRSE and RAE on the level series

## Files

| File | Contents |
|---|---|
| `comparison_table_2025.csv` / `.md` | 2024 validation RRSE and 2025 test RRSE/RAE for every model, with the source of each value |
| `comparison_table_footnotes.csv` | Supplementary rows (linear extrapolation, Transformer without blending, B-MTGNN retrained with seeds 1-5) |
| `direction_metrics/summary_2025.csv` | Direction accuracy (%), F1 and change correlation per currency |
| `direction_metrics/direction_metrics_2025.csv`, `f1_2025.csv` | Full values: hit counts, up/down F1, macro-F1, number of predicted down months |
| `predictions/<model>/` | Monthly 2025 predictions (`time, y_true, y_pred`) |

Direction metrics compare the sign of the predicted change (ŷ_t − y_{t−1}) with the sign of the actual change (y_t − y_{t−1}). USD/KRW rose in all 12 months of 2025, so a model that always predicts "up" scores 100% and F1 = 1 there; the down-class F1 and macro-F1 are not defined for KRW. Naive-last predicts no change and is excluded from the direction metrics.

## Prediction files

| Folder | Model | Files |
|---|---|---|
| `naive_last/` | Naive-last (y_{t−1}) | `predictions_test_{US,KR,JP}.csv`, metrics, config |
| `drift/` | Drift | same |
| `ar/` | AR (order selected on 2024 validation, no post-processing) | `ar_predictions_test.csv` (all targets, `target` column), `ar_selected_metrics.csv`, `validation_grid.csv` |
| `lstm_33var/` | LSTM, 33 variables, residual delta | `predictions_test_*.csv`, metrics, config |
| `gru/` | GRU, same pipeline as LSTM | `predictions_{test,valid}_*.csv`, metrics, seed results |
| `tcn_33var_causal/` | Causal TCN, 33 variables | `predictions_test_*.csv`, metrics, config |
| `transformer_33var/` | Transformer, 33 variables (validation-selected persistence blend 0.5) | `predictions_test_*.csv`, metrics, config |
| `mtgnn_deterministic/` | MTGNN without post-processing | `predictions_test_*.csv`, metrics, config |
| `mtgnn_eq7_val_alpha/` | MTGNN with Eq. (7), alpha selected on validation | `predictions_{test,valid}_val_alpha_*.csv`, `metrics.csv` |
| `b_mtgnn/` | B-MTGNN (epoch 57, CPU, MC 10 runs) | `paper_ep57_mc10_val_alpha_*` with Eq. (7), `paper_ep57_mc10_raw_*` without |

B-MTGNN predicted-vs-actual plots for 2024 and 2025 are in `B-MTGNN/model/Bayesian/Validation/` and `Testing/`.
