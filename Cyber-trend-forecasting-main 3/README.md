# Exchangeprediction

This is a PyTorch implementation of the Bayesian multi-task graph neural network model for forecasting foreign exchange (FX) rates and pertinent economic indicators. The model extends the [MTGNN](https://dl.acm.org/doi/abs/10.1145/3394486.3403118) model proposed by Wu et al. The graph represents relationships between multiple exchange rates (USD/JPY, USD/KRW, US Trade Weighted Dollar Index) and relevant economic indicators such as CPI, GDP, interest rates, employment, trade balance, and unemployment across different countries. Each node represents an economic variable, and the value of the node represents the trend.

In our extension for the model, we employ the Bayesian approach to capture epistemic uncertainty. Specifically, we employ the Monte Carlo dropout method where the use of dropout neurons during inference provides a Bayesian approximation of the deep Gaussian processes. Therefore, during the prediction phase, the trained model runs multiple times, which results in a distribution of prediction (representing the uncertainty) rather than a single point.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Smoothing
All data files used by the model including the graph adjacency file can be found in the directory called **data**.

Data preprocessing and smoothing are performed to prepare the raw economic indicators for model training. The smoothed data is used as input to the neural network.

## Hyper-parameter Optimisation
The hyper-parameter optimisation is performed in the file **train_test.py**. This script performs random search to produce the optimal set of hyper-parameters. These hyper-parameters are finally saved as an output in the file called **hp.txt**, which is in the directory **model/Bayesian**. The output also includes validation and testing results when using the optimal set of hyper-parameters. These results include plots for the predicted curves against the actual curves. These are saved in the directories called **Validation** and **Testing** within the directory **model/Bayesian**. For the evaluation, 2 metrics are used namely the Root Relative Squared Error (RSE) and the Relative Absolute Error (RAE). These metrics are saved in the same directories (Validation and Testing), and the average values of these metrics across all nodes are also displayed on the console as a final output. 

Below is an example for the model validation results across key variables - exchange rates and economic indicators.

<p align="center">
  <img src="./AXIS/model/Bayesian/Validation/Jp_fx_Validation.png" width="600" />
</p>

<p align="center">
  <img src="./AXIS/model/Bayesian/Validation/Kr_fx_Validation.png" width="600" />
</p>

<p align="center">
  <img src="./AXIS/model/Bayesian/Validation/US_Unemployment_Validation.png" width="600" />
</p>

## Testing Results (Key FX Variables)

<p align="center">
  <img src="https://raw.githubusercontent.com/jiwon-choi35/-Graph-Neural-Network-based-Multi-Currency-FX-Forecasting/feature/FinalMultisteps-dollarIndex/Cyber-trend-forecasting-main%203/AXIS/model/Bayesian/Testing/Jp_fx_Testing.png" width="600" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/jiwon-choi35/-Graph-Neural-Network-based-Multi-Currency-FX-Forecasting/feature/FinalMultisteps-dollarIndex/Cyber-trend-forecasting-main%203/AXIS/model/Bayesian/Testing/Kr_fx_Testing.png" width="600" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/jiwon-choi35/-Graph-Neural-Network-based-Multi-Currency-FX-Forecasting/feature/FinalMultisteps-dollarIndex/Cyber-trend-forecasting-main%203/AXIS/model/Bayesian/Testing/Us_Trade%20Weighted%20Dollar%20Index_Testing.png" width="600" />
</p>

## Operational Model
The script in the file **train.py** trains the final model on the full data using the optimal hyper-parameters stored in the file **hp.txt**. The output is the operational model called **model.pt**, which can be used to forecast the exchange rates and economic indicators. The operational model is saved in the directory **AXIS/model/Bayesian**. 

## Future Forecast
The script in the file **forecast.py** uses the operational model **model.pt** in the directory **AXIS/model/Bayesian** to produce forecasts for the exchange rates and economic indicators. The results include numerical forecasts of each node, stored in the directory **AXIS/model/Bayesian/forecast/data**. In addition, plots for the trends of exchange rates along with their related economic indicators are provided in the directory **AXIS/model/Bayesian/forecast/plots**. The plots visualize historical data alongside forecasted values, enabling analysis of projected trends.

The numerical forecasts are saved in a structured format for further analysis and integration into downstream applications.

Below is an example for the past and predicted future data for exchange rates and their pertinent economic indicators.

<p align="center">
  <img src="./AXIS/model/Bayesian/forecast/plots/Multi_Country_Forecast_Normalized.png" width="600" />
</p>
