# 🤖 AI Stock Price Predictor

**Author**: Shivam Bhardwaj  
**Tech Stack**: Streamlit, scikit-learn, TensorFlow, XGBoost, yfinance, Plotly

## 📝 Overview

This Streamlit application predicts the **next-day stock closing price** using a variety of machine learning and deep learning models including:

- Random Forest
- Linear Regression
- XGBoost
- Gradient Boosting
- LSTM Neural Network
- Advanced Ensemble (Random Forest + XGBoost + SVR + Gradient Boosting)

You can view performance metrics (MAE, RMSE, R²), feature importances, confidence level, and risk assessment to understand the reliability of each prediction.

---

## 🧪 Features

- 📈 Fetch historical stock data via `yfinance`
- 🧠 Train ML/DL models on historical and technical indicators
- 📉 Predict next-day closing prices
- 📊 Dynamic visualizations with Plotly
- ⚙️ Adjustable model and parameters via sidebar
- 🇮🇳 **Supports Indian stock symbols** like `RELIANCE.NS`, `TCS.NS`, `INFY.BO`, etc.
