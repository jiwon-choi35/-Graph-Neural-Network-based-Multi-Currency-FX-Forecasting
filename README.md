# Exchangeprediction

This is a Python implementation of a Bayesian multi-task graph neural network (B-MTGNN) framework for forecasting foreign exchange (FX) rates and related economic indicators.

This repository contains an end-to-end framework for forecasting exchange rate trends and pertinent economic indicators using graph neural networks. This includes data preparation, model development, hyperparameter optimization, and future forecasts up to 1 years in advance.

## Dataset

The dataset includes foreign exchange rates (USD/JPY, USD/KRW, US Trade Weighted Dollar Index) and related economic indicators across multiple countries (US, Japan, South Korea). Economic indicators include CPI, GDP, interest rates, employment, trade balance, and unemployment. Data spans from January 2011 to December 2025 on a monthly basis.

## Model Architecture

The directory **B-MTGNN** contains the core PyTorch implementation of the Bayesian multi-task graph neural network model. The model extends the MTGNN architecture with Bayesian inference using Monte Carlo dropout to capture epistemic uncertainty in predictions. This results in prediction distributions rather than point estimates.

### Key Features:

- **Multi-Task Learning**: Forecasts multiple exchange rates and economic indicators simultaneously
- **Graph Neural Networks**: Models relationships between different economic variables as a graph
- **Bayesian Inference**: Uses Monte Carlo dropout for uncertainty quantification
- **1-Year Forecasting Horizon**: Predicts trends 12 months into the future

## Model Training & Optimization

The scripts in the **B-MTGNN** directory perform:

1. **Hyperparameter Optimization** (`train_test.py`): Random search to find optimal model hyperparameters, producing `hp.txt`
2. **Model Training** (`train.py`): Trains the final model using optimal hyperparameters
3. **Performance Evaluation**: Validation and testing with metrics RSE (Root Relative Squared Error) and RAE (Relative Absolute Error)
   - Results stored in `AXIS/model/Bayesian/Validation/` and `AXIS/model/Bayesian/Testing/`

## Operational Model & Results

The directory **AXIS** contains the trained operational model and all outputs:

### Model Files:

- `model/Bayesian/model.pt` - The final trained Bayesian MTGNN model

### Forecast Outputs:

- **Data**: `model/Bayesian/forecast/data/` - Numerical forecasts for each economic variable
- **Plots**: `model/Bayesian/forecast/plots/` - Visualization of historical and forecasted trends
  - Individual plots for each exchange rate
  - Multi-country normalized comparison plot
- **Gap Analysis**: `model/Bayesian/forecast/gap/` - Gap analysis between target variables

## Forecast Results

The script `forecast.py` uses the trained model to generate:

1. **Individual Forecasts**: 12-month forecasts (2026-01 to 2026-12) for each economic variable
2. **Multi-Country Comparison**: Normalized comparison of major exchange rates showing forecast uncertainty bands (95% prediction intervals)
3. **Gap Analysis**: Differences between forecasted exchange rates and economic indicators stored in CSV format

Below is an example of exchange rate forecasts with uncertainty quantification:

<p align="center">
  <img src="https://raw.githubusercontent.com/jiwon-choi35/-Graph-Neural-Network-based-Multi-Currency-FX-Forecasting/feature/FinalMultisteps-dollarIndex/Cyber-trend-forecasting-main%203/AXIS/model/Bayesian/forecast/plots/Multi_Country_Forecast_Normalized.png" width="600" />
</p>

## Model Specifications

- **Input Sequence Length**: 24 months (2 years of historical data)
- **Forecast Horizon**: 12 months (2026)
- **MC Runs**: 20 dropout iterations for Bayesian approximation
- **Evaluation Metrics**: RSE and RAE across 33 economic variables
- **Architecture**: Graph neural network with attention mechanisms

## Usage

### Training & Optimization

```bash
python train_test.py
```

### Generating Future Forecasts

```bash
python forecast.py
```

### Visualization

Generated plots are automatically saved in:

- `AXIS/model/Bayesian/forecast/plots/` - High-resolution PNG and PDF formats

## Requirements

Python 3.8+ with PyTorch and dependencies specified in the project requirements.

## Project Structure

```
├── AXIS/                    # Model outputs and results
│   └── model/Bayesian/     # Trained model and forecasts
│       ├── model.pt
│       ├── forecast/
│       │   ├── data/       # Numerical forecasts
│       │   ├── plots/      # Visualizations
│       │   └── gap/        # Gap analysis
│       ├── Testing/        # Test set results
│       └── Validation/     # Validation set results
│
└── B-MTGNN/                 # Model implementation
    ├── forecast.py          # Generate future forecasts
    ├── train_test.py        # Hyperparameter optimization
    ├── net.py              # Model architecture
    ├── trainer.py          # Training routines
    ├── layer.py            # Custom layers
    ├── util.py             # Utilities
    └── data/               # Input data files
```

## Citation

If you use this framework, please cite this work.

## License

This project is provided for research and development purposes.





