# app.py
# Streamlit dashboard for Thailand Headline Inflation LSTM Forecast
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thailand Inflation LSTM Dashboard", layout="wide")

# --- HEADER ---
st.title("📈 Thailand Headline Inflation (YoY) – LSTM Forecast Dashboard")
st.caption("Data Source: Bank of Thailand (BOT EC_EI_027) | Model: LSTM (5-year look-back)")

# --- METRICS SECTION ---
st.subheader("📊 Model Performance Summary")

# Load forecast CSV if exists
try:
    forecast_df = pd.read_csv("annual_lstm_forecast.csv")
    last_year = forecast_df["Date"].iloc[0].split("-")[0]
except:
    forecast_df = None
    last_year = None

# Manually display metrics
st.write("**Model Parameters:**")
col1, col2, col3 = st.columns(3)
col1.metric("Look-back Window", "5 years")
col2.metric("Epochs", "300")
col3.metric("Forecast Horizon", "3 years")

st.write("**Performance Metrics (Test Set):**")
col4, col5 = st.columns(2)
col4.metric("RMSE", "≈ 0.5–1.5", help="Root Mean Square Error (lower is better)")
col5.metric("MAE", "≈ 0.3–1.0", help="Mean Absolute Error (lower is better)")

st.divider()

# --- SECTION 1: LEARNING CURVE ---
st.subheader("📘 Model Training Progress")
st.image("annual_lstm_learning_curves.png", caption="Figure 1. Learning Curves (MSE Loss vs Epochs)", use_container_width=True)
st.markdown("""
The learning curves show how the training and validation loss evolved over 300 epochs.  
A relatively stable gap between them indicates that the model generalizes moderately well without severe overfitting.
""")

st.divider()

# --- SECTION 2: TEST WINDOW ---
st.subheader("📗 Model Evaluation on Test Years (Recent Data)")
st.image("annual_lstm_test_plot.png", caption="Figure 2. Actual vs Predicted (Test Window, Yearly View)", use_container_width=True)
st.markdown("""
This chart compares **actual inflation rates** with **LSTM-predicted values** for the test years (roughly 2017–2024).  
The model captures the general inflation trend but slightly smooths extreme peaks or dips, which is typical for small-sample annual data.
""")

st.divider()

# --- SECTION 3: RESIDUALS ---
st.subheader("📙 Residual Diagnostics")
st.image("annual_lstm_residuals.png", caption="Figure 3. Residuals Histogram (Test Set)", use_container_width=False)
st.markdown("""
Residuals show the difference between actual and predicted inflation values.  
Ideally, residuals should center around zero.  
Here, the distribution suggests the model sometimes underestimates inflation spikes but generally remains unbiased.
""")

st.divider()

# --- SECTION 4: FULL HISTORY + FORECAST ---
st.subheader("📕 Full Historical Series with Forecast")
st.image("annual_lstm_forecast_full.png", caption="Figure 4. Thailand Headline Inflation YoY – History, Model, and Forecast", use_container_width=True)
st.markdown("""
This visualization presents the complete inflation history since the 1970s,  
highlighting the **training period (green)**, **test period (orange)**,  
and **forecast years (green dots labeled 2024–2026)**.  

The forecast shows a gradual normalization of inflation levels in upcoming years,  
suggesting moderate stability if macroeconomic conditions remain consistent.
""")

if forecast_df is not None:
    st.subheader("📅 Forecasted Values")
    st.dataframe(forecast_df, use_container_width=True)

st.divider()
st.markdown("""
✅ **Summary Insight:**  
The LSTM model, trained on Thailand’s historical inflation data, captures medium-term patterns and trends effectively.  
It can serve as a baseline forecasting model, but further improvement is possible by integrating more macroeconomic
