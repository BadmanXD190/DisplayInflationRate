# app.py
# Streamlit dashboard for Thailand Headline Inflation (YoY) LSTM
# Shows precomputed artifacts if present, or trains inside Streamlit.
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from matplotlib.dates import YearLocator, DateFormatter

ARTIFACTS = {
    "learning": "annual_lstm_learning_curves.png",
    "test": "annual_lstm_test_plot.png",
    "resid": "annual_lstm_residuals.png",
    "full": "annual_lstm_forecast_full.png",
    "csv": "annual_lstm_forecast.csv",
}
DATA_CSV = "thai_headline_inflation_yoy_annual.csv"

st.set_page_config(page_title="Thailand Inflation LSTM Dashboard", layout="wide")
st.title("Thailand Headline Inflation (YoY) — LSTM Forecast Dashboard")
st.caption("Source: BOT EC_EI_027. This page can load existing results or train a model and regenerate them.")

# ---------- Helpers ----------
def year_axis(ax):
    ax.xaxis.set_major_locator(YearLocator(1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def create_sequences(data, window):
    X, y, idx_map = [], [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
        idx_map.append(i)
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)
    idx_map = np.array(idx_map)
    return X, y, idx_map

@st.cache_data
def load_series(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna()
    series = df.set_index("Date")["Inflation_YoY_pct"].astype(float)
    return series

def build_model(window, dropout=0.2):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(dropout),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# ---------- Sidebar ----------
st.sidebar.header("Model and Forecast Controls")
window = st.sidebar.number_input("Look-back window (years)", 3, 10, 5, 1)
epochs = st.sidebar.number_input("Epochs", 50, 1000, 300, 50)
batch_size = st.sidebar.number_input("Batch size", 1, 64, 4, 1)
forecast_years = st.sidebar.number_input("Forecast horizon (years)", 1, 10, 3, 1)
use_mc = st.sidebar.checkbox("Show uncertainty bands (MC dropout)", value=False)
mc_samples = st.sidebar.number_input("MC samples", 20, 500, 200, 20, disabled=not use_mc)
regen = st.sidebar.button("Train model and regenerate charts")

# ---------- Load data ----------
if not os.path.exists(DATA_CSV):
    st.error(f"Missing data file: {DATA_CSV}")
    st.stop()

series = load_series(DATA_CSV)

# ---------- Branch A: regenerate ----------
if regen:
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y, idx_map = create_sequences(scaled, window)
    seq_dates = series.index[idx_map]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_train, dates_test = seq_dates[:split], seq_dates[split:]

    model = build_model(window)
    history = model.fit(
        X_train, y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Learning curves
    fig = plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Learning Curves (MSE loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    st.image(fig_to_png_bytes(fig), caption="Figure 1. Learning Curves", use_container_width=True)
    fig.savefig(ARTIFACTS["learning"]); plt.close(fig)

    # Evaluate
    y_pred = model.predict(X_test, verbose=0)
    y_pred_inv = scaler.inverse_transform(y_pred).flatten()
    y_test_inv = scaler.inverse_transform(y_test).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))

    col1, col2, col3 = st.columns(3)
    col1.metric("Look-back", f"{window} years")
    col2.metric("Epochs", str(epochs))
    col3.metric("Forecast horizon", f"{forecast_years} years")
    c4, c5 = st.columns(2)
    c4.metric("RMSE", f"{rmse:.3f}")
    c5.metric("MAE", f"{mae:.3f}")

    # Test plot (year axis)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates_test, y_test_inv, label="Actual", linewidth=2)
    ax.plot(dates_test, y_pred_inv, label="Predicted", linestyle="--")
    ax.set_title("Thailand Headline Inflation YoY — Test Window")
    ax.set_xlabel("Year"); ax.set_ylabel("Inflation YoY (%)")
    year_axis(ax); ax.legend()
    st.image(fig_to_png_bytes(fig), caption="Figure 2. Actual vs Predicted (Test)", use_container_width=True)
    fig.savefig(ARTIFACTS["test"]); plt.close(fig)

    # Forecast (with optional MC dropout)
    last_seq = scaler.transform(series.values.reshape(-1, 1))[-window:].copy()
    future_years_idx = []
    future_mean = []
    lower, upper = [], []

    last_year = series.index.max().year
    future_dates = [pd.Timestamp(f"{y}-12-01") for y in range(last_year + 1, last_year + 1 + forecast_years)]

    if use_mc:
        rng = np.random.default_rng(42)
        cur_seq = last_seq.copy()
        preds_all = []
        for step in range(forecast_years):
            draws = []
            for _ in range(int(mc_samples)):
                p = model(cur_seq.reshape(1, window, 1), training=True).numpy()[0, 0]
                draws.append(p)
            preds_all.append(draws)
            next_val = np.mean(draws)
            cur_seq = np.append(cur_seq[1:], next_val).reshape(-1, 1)
        preds_all = np.array(preds_all)  # shape: steps x samples
        mu = preds_all.mean(axis=1)
        lo = np.quantile(preds_all, 0.05, axis=1)
        hi = np.quantile(preds_all, 0.95, axis=1)
        future_mean = scaler.inverse_transform(mu.reshape(-1, 1)).flatten()
        lower = scaler.inverse_transform(lo.reshape(-1, 1)).flatten()
        upper = scaler.inverse_transform(hi.reshape(-1, 1)).flatten()
    else:
        cur_seq = last_seq.copy()
        outs = []
        for _ in range(forecast_years):
            p = model.predict(cur_seq.reshape(1, window, 1), verbose=0)[0, 0]
            outs.append(p)
            cur_seq = np.append(cur_seq[1:], p).reshape(-1, 1)
        future_mean = scaler.inverse_transform(np.array(outs).reshape(-1, 1)).flatten()

    # In-sample reconstruction for context
    y_all_pred = model.predict(X, verbose=0).flatten()
    y_all_pred_inv = scaler.inverse_transform(y_all_pred.reshape(-1, 1)).flatten()
    dates_all_pred = series.index[idx_map]

    # Full history + forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series.values, label="Actual (History)", linewidth=2)
    ax.plot(dates_all_pred, y_all_pred_inv, label="Model (in-sample)", linestyle="--", alpha=0.8)
    ax.plot(future_dates, future_mean, marker="o", linestyle="-", label="Forecast")
    if use_mc:
        ax.fill_between(future_dates, lower, upper, alpha=0.2, label="MC 90% interval")

    if len(dates_train) > 0:
        ax.axvspan(dates_train.min(), dates_train.max(), color="green", alpha=0.06, label="Train period")
    if len(dates_test) > 0:
        ax.axvspan(dates_test.min(), dates_test.max(), color="orange", alpha=0.08, label="Test period")

    for d, v in zip(future_dates, future_mean):
        ax.annotate(d.strftime("%Y"), (d, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    ax.set_title("Thailand Headline Inflation YoY — History, Model, and Forecast")
    ax.set_xlabel("Year"); ax.set_ylabel("Inflation YoY (%)")
    year_axis(ax); ax.legend()
    st.image(fig_to_png_bytes(fig), caption="Figure 3. History + Forecast", use_container_width=True)
    fig.savefig(ARTIFACTS["full"]); plt.close(fig)

    # Residual diagnostics
    residuals = y_test_inv - y_pred_inv
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(dates_test, residuals, marker="o", linestyle="-")
    ax.axhline(0, linewidth=1)
    ax.set_title("Residuals over Time (Test)")
    ax.set_xlabel("Year"); ax.set_ylabel("Residual")
    year_axis(ax)
    png1 = fig_to_png_bytes(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(5.5, 3.5))
    plt.hist(residuals, bins=10, edgecolor="black")
    plt.title("Residuals Histogram (Test)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    png2 = fig_to_png_bytes(fig)
    plt.close(fig)

    # Save combined residuals image
    with open(ARTIFACTS["resid"], "wb") as f:
        f.write(png2)

    # Save forecast CSV
    f_df = pd.DataFrame({"Date": future_dates, "Forecast_YoY_pct": future_mean})
    if use_mc:
        f_df["Lower_90"] = lower
        f_df["Upper_90"] = upper
    f_df.to_csv(ARTIFACTS["csv"], index=False)

    st.subheader("Forecast Table")
    st.dataframe(f_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download forecast CSV", data=f_df.to_csv(index=False), file_name="annual_lstm_forecast.csv")
    with open(ARTIFACTS["full"], "rb") as f:
        c2.download_button("Download history+forecast plot", data=f.read(), file_name="annual_lstm_forecast_full.png")
    with open(ARTIFACTS["test"], "rb") as f:
        c3.download_button("Download test plot", data=f.read(), file_name="annual_lstm_test_plot.png")

    st.info("Tip: enable uncertainty in the sidebar to show a 90 percent interval using Monte-Carlo dropout.")

# ---------- Branch B: just display existing artifacts ----------
else:
    st.subheader("Model Performance and Visuals (precomputed)")
    cols = st.columns(2)
    # Metrics are unknown without the training history. Show general notes.
    cols[0].metric("Look-back", "5 years")
    cols[1].metric("Forecast horizon", "3 years")
    missing = [k for k, p in ARTIFACTS.items() if not os.path.exists(p)]
    if missing:
        st.warning("Some artifacts are missing. Click **Train model and regenerate charts** in the sidebar.")
    if os.path.exists(ARTIFACTS["learning"]):
        st.image(ARTIFACTS["learning"], caption="Figure 1. Learning Curves", use_container_width=True)
    if os.path.exists(ARTIFACTS["test"]):
        st.image(ARTIFACTS["test"], caption="Figure 2. Actual vs Predicted (Test)", use_container_width=True)
    if os.path.exists(ARTIFACTS["resid"]):
        st.image(ARTIFACTS["resid"], caption="Figure 3. Residuals Histogram (Test)")
    if os.path.exists(ARTIFACTS["full"]):
        st.image(ARTIFACTS["full"], caption="Figure 4. History + Forecast", use_container_width=True)
    if os.path.exists(ARTIFACTS["csv"]):
        f_df = pd.read_csv(ARTIFACTS["csv"])
        st.subheader("Forecast Table")
        st.dataframe(f_df, use_container_width=True)
        st.download_button("Download forecast CSV", data=f_df.to_csv(index=False), file_name="annual_lstm_forecast.csv")

st.divider()
st.markdown("""
**Explanation**  
This dashboard presents Thailand’s annual headline inflation series, trains an LSTM with a sliding window, evaluates on the most recent years, and produces a multi-year forecast labeled by calendar year.  
Residual charts help check bias. The uncertainty option draws many dropout passes at inference to create a 90 percent interval.  
To improve accuracy, add macro features such as oil prices, exchange rate, policy rate, and GDP growth.
""")
