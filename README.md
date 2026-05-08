# AI-Driven Supply Chain Demand Forecasting
## Using Temporal Fusion Transformer (TFT)

###  Business Problem
A large FMCG retailer with 500+ SKUs experiences chronic stockouts ($15M loss annually) and excess inventory ($25M tied up). This project replaces traditional moving averages with a Deep Learning model to capture complex temporal patterns.

###  Technical Solution
Built a multi-horizon forecasting system using the **Temporal Fusion Transformer (TFT)**. Unlike LSTM or ARIMA, TFT uses attention mechanisms to handle:
- **Static Metadata:** Store ID, Category ID.
- **Known Future Inputs:** Price, Holidays, Day of Week.
- **Observed Past Inputs:** Historical Sales.

###  Tech Stack
- **Language:** Python 3.13
- **Framework:** PyTorch & PyTorch Forecasting
- **Engine:** Lightning AI
- **Visualization:** Matplotlib

###  Results
The model generates probabilistic forecasts (Quantile Loss), providing not just a single number, but a confidence interval (P10, P50, P90) to help supply chain managers mitigate risk.
- **Forecast Horizon:** 21 Days
- **History Lookback:** 60 Days

###  How to Run
1. Clone the repo: `git clone [YOUR_URL]`
2. Install dependencies: `pip install pytorch-forecasting lightning`
3. Preprocess data: `python src/preprocess.py`
4. Train model: `python src/train.py`
5. Generate forecast: `python src/predict.py`