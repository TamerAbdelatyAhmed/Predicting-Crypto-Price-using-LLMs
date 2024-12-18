# Predicting Cryptocurrencies Using Large Language Models (LLMs)

## Introduction
This project explores the use of Large Language Models (LLMs) in conjunction with traditional time series forecasting techniques to predict cryptocurrency trends. Cryptocurrencies are highly volatile assets, making accurate predictions critical for investors and analysts. By combining financial data analysis, feature engineering, ARIMA modeling, and LLM-based insights, this project aims to uncover patterns and improve forecasting accuracy.

---

## Objectives
1. **Data Analysis**: Collect and analyze historical cryptocurrency data.
2. **Forecasting with ARIMA**: Use traditional statistical models to predict daily percentage changes.
3. **LLM Integration**: Leverage LLMs to enhance prediction accuracy.
4. **Performance Comparison**: Evaluate the effectiveness of ARIMA and LLMs against actual data.

---

## Methodology

### 1. **Data Collection**
- **Source**: Historical cryptocurrency data was retrieved using the `yfinance` library.
- **Cryptocurrencies Analyzed**: Bitcoin (BTC), Ethereum (ETH), and Ripple (XRP).
- **Time Frame**: The past 30 days of data were fetched for analysis.

#### Code Snippet:
```python
import yfinance as yf
from datetime import datetime, timedelta

def pull_stocks(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    stock_data = yf.Ticker(ticker)
    stock_df = stock_data.history(start=start_date, end=end_date)
    stock_df['pct_change'] = stock_df['Close'].pct_change()
    return stock_df[['Date', 'pct_change']].dropna()

btc = pull_stocks('BTC-USD')
eth = pull_stocks('ETH-USD')
xrp = pull_stocks('XRP-USD')
```

---

### 2. **Preprocessing**
- **Feature Engineering**: Daily percentage changes were calculated as features for time series modeling.
- **Cleaning**: Missing values were removed to ensure model integrity.

---

### 3. **ARIMA Modeling**
- The ARIMA (AutoRegressive Integrated Moving Average) model was employed for time series forecasting.
- **Objective**: Predict the percentage change for the next day based on historical data.

#### Code Snippet:
```python
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(data):
    model = ARIMA(data['pct_change'], order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=1)
    return forecast[0]

btc_forecast = arima_forecast(btc)
eth_forecast = arima_forecast(eth)
xrp_forecast = arima_forecast(xrp)
```

**Results**:
- Predicted percentage changes:
  - BTC: 0.0063
  - ETH: 0.0153
  - XRP: 0.0310

---

### 4. **LLM Integration**
- **Model Used**: The project employed a local LLM (e.g., `Ollama`) to interpret historical data and predict trends.
- **Input Preparation**: The time series data was converted into a CSV string for LLM input.

#### Code Snippet:
```python
from langchain_community.llms import Ollama

def predict_with_llm(timeseries):
    llm = Ollama(model="llama3", temperature=0)
    prompt = f"""
        Predict the percentage change for the next day based on the following data:
        {timeseries}
    """
    return llm.invoke(prompt).strip()

btc_llm_forecast = predict_with_llm(btc.to_csv())
eth_llm_forecast = predict_with_llm(eth.to_csv())
xrp_llm_forecast = predict_with_llm(xrp.to_csv())
```

**LLM Predictions**:
- BTC: 0.0328
- ETH: -0.0123
- XRP: 0.0123

---

### 5. **Evaluation**
- **Comparison with Actual Data**: Predictions were compared against actual percentage changes.

| Cryptocurrency | ARIMA Prediction | LLM Prediction | Actual |
|----------------|-------------------|----------------|--------|
| BTC            | 0.0063           | 0.0328         | -0.0200|
| ETH            | 0.0153           | -0.0123        | -0.0078|
| XRP            | 0.0310           | 0.0123         | -0.0171|

- **Insights**:
  - ARIMA provided closer estimates for ETH and XRP.
  - LLM predictions showed higher variance, possibly due to sensitivity to input formatting.

---

## Challenges
1. **Volatility**: Cryptocurrencies exhibit unpredictable price swings, challenging both models.
2. **LLM Context Handling**: Ensuring LLMs accurately interpret numerical data and trends.
3. **Data Quantity**: Limited historical data may reduce model effectiveness.

---

## Conclusion
This project demonstrates the potential of combining traditional statistical methods with state-of-the-art LLMs for cryptocurrency forecasting. While ARIMA provided more consistent predictions, LLMs offer a promising avenue for capturing complex patterns in financial data.

---

## Future Work
1. **Extended Time Frames**: Incorporate longer historical data for improved accuracy.
2. **Model Fine-Tuning**: Enhance LLM prompts and parameters for better numerical reasoning.
3. **Hybrid Approaches**: Combine ARIMA and LLM predictions using ensemble methods.

---

## Skills Demonstrated
- **Data Analysis**: Proficient in Python libraries (`pandas`, `numpy`, `yfinance`).
- **Time Series Forecasting**: Applied ARIMA models for financial trend prediction.
- **LLM Integration**: Leveraged LLMs for innovative forecasting solutions.
- **Problem-Solving**: Addressed challenges in volatile and complex datasets.

---

## Attachments
- Jupyter Notebook: [Predicting Crypto Currencies using LLMs.ipynb](Predicting%20Crypto%20Currencies%20using%20LLMs.ipynb)

