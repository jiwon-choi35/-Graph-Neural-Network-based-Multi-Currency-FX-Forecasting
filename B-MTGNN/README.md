# B-MTGNN

PyTorch implementation of the Bayesian multi-task graph neural network used for forecasting the US Trade Weighted Dollar Index, USD/KRW and USD/JPY together with 30 related variables (33 nodes). The model extends [MTGNN](https://dl.acm.org/doi/abs/10.1145/3394486.3403118) with Monte Carlo dropout: at inference the trained model runs multiple times with dropout on, which gives a distribution of predictions instead of a single point.

All commands below are run from this directory (`B-MTGNN/`).

## Files

| File | Role |
|---|---|
| `net.py`, `layer.py` | Model architecture (gtnet) |
| `util.py` | Data loader, graph construction, metrics |
| `trainer.py` | Optimizer wrapper |
| `smoothing.py` | Double exponential smoothing (alpha 0.1, beta 0.3) of `data/data.txt` |
| `train_test.py` | Training with validation-based epoch selection, 2024 validation and 2025 test evaluation |
| `forecast.py` | 2026 forecast with the trained model |
| `extract_adjacency.py` | Learned adjacency matrix analysis (graph_analysis/) |
| `figure_plot.py` | Gap bar plots from the forecast gap files |

## Data

`data/` holds the model inputs.

- `data.csv`: raw monthly data with column names (2011-01 ~ 2025-12)
- `data.txt`: the same values, tab-separated without header (input of `smoothing.py`)
- `sm_data.csv`: smoothed data with header. **This is the file the model reads.**
- `graph.csv`: node list used for the predefined graph

Note: `smoothing.py` writes `data/sm_data.csv` without the header and the Date column. Running it overwrites the committed file in a different format, so it is not needed to reproduce the results.

## 1. Training and evaluation

```bash
python train_test.py --device cuda:0
```

The default preset (`--target_profile triple_050`) fixes the configuration used in the paper: seed 1, L1 loss, 180 epochs, epoch selected on the 2024 validation set, post-hoc anchor coefficient (Eq. 7) selected on validation. The selected hyper-parameters are written to `model/Bayesian/hp.txt` and the selected model to `model/Bayesian/model.pt` (epoch 57).

Outputs:

- `model/Bayesian/Validation/`, `model/Bayesian/Testing/`: predicted vs. actual plots (`*.png`) and per-variable RRSE/RAE (`*.txt`), prediction intervals (`interval_bounds_*.csv`, `interval_summary_*.csv`)
- `final_forecast_results.png`, `final_summary_table.png`: summary figures of the three target currencies

Use `--device cpu` if no GPU is available. MC dropout sampling differs between CPU and GPU, so RRSE can differ slightly by device (up to about 0.02 in our runs).

## 2. 2026 forecast

```bash
python forecast.py
```

Uses `model/Bayesian/model.pt` and writes:

- `model/Bayesian/forecast/data/`: numerical forecasts for each variable (`forecast_2026.csv`, `forecast_2026_wide.csv`, per-variable `*.txt`)
- `model/Bayesian/forecast/plots/`: multi-country normalized forecast plot
- `model/Bayesian/forecast/pt_plots/`: forecast plot for each target currency
- `model/Bayesian/forecast/gap/`: gaps between the forecast currencies

```bash
python figure_plot.py
```

Draws `normalized_gap_bar` and `normalized_gap_grouped_bar` from the gap files.

## 3. Graph analysis

```bash
python extract_adjacency.py
```

Loads `model/Bayesian/model.pt` and writes the learned adjacency matrix and the top neighbors of each node to `model/Bayesian/graph_analysis/`.
