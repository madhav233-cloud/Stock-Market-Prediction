import os
import json
import hashlib
from datetime import datetime, timedelta, date

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Layer, Input, LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

# -------------------------
# UI STYLING
# -------------------------
st.markdown("""
<style>
.main { background-color:#0b0f1a; color:white; }

h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

.header {
    text-align:center;
    font-size:34px;
    font-weight:bold;
    background: linear-gradient(90deg,#00ffe0,#007cf0);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.stButton>button {
    background-color: #007cf0 !important;
    color: white !important;
    border-radius: 8px;
    font-weight: bold;
    opacity: 1 !important;
}
.stButton>button:hover {
    background-color: #00c6ff !important;
    color: black !important;
}

.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: #111728;
    padding: 8px 0;
    border-radius: 8px;
}
.ticker-move {
    display: inline-block;
    white-space: nowrap;
    animation: ticker 20s linear infinite;
}
@keyframes ticker {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

.card {
    background:#141a2e;
    padding:15px;
    border-radius:10px;
    text-align:center;
}

.table-card {
    background:#111728;
    padding:15px;
    border-radius:12px;
}
.row {
    display:flex;
    justify-content:space-between;
    padding:10px;
    border-bottom:1px solid #2a2f45;
}
.green { color:#00ff88; }
.red { color:#ff4d4d; }
.header-row {
    font-weight:bold;
    color:#cccccc;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>🚀 AI Stock Prediction Dashboard</div>", unsafe_allow_html=True)

# -------------------------
# AUTHENTICATION
# -------------------------
def load_users():
    if os.path.exists("users.json"):
        try:
            with open("users.json", "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def authenticate_user(username, password):
    users = load_users()
    if username in users:
        hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
        return users[username] == hashed_pwd
    return False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    outer = st.columns([2, 1, 2])
    with outer[1]:
        st.markdown("### 🔐 Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if user and pwd:
                if authenticate_user(user, pwd):
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            else:
                st.warning("Please enter both username and password.")
    st.stop()

# -------------------------
# LIVE TICKER
# -------------------------
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
ticker_text = ""

for t in tickers:
    try:
        data = yf.download(t, period="1d", progress=False, multi_level_index=False)
        if not data.empty:
            price = round(float(data["Close"].iloc[-1]), 2)
            ticker_text += f"{t}: ₹{price} 🔹 "
        else:
            ticker_text += f"{t}: N/A 🔹 "
    except Exception:
        ticker_text += f"{t}: Error 🔹 "

st.markdown(f"""
<div class="ticker-wrap">
<div class="ticker-move">{ticker_text}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# MODEL & SCALERS
# -------------------------
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1))
        self.b = self.add_weight(shape=(input_shape[1], 1))

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

def load_model_with_legacy_support(model_path, custom_objects=None):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file `{model_path}` missing from repository root.")
        st.stop()

    try:
        return load_model(model_path, custom_objects=custom_objects, compile=False)
    except (TypeError, ValueError) as err:
        error_str = str(err)
        if "shape" not in error_str and "batch_shape" not in error_str:
            raise

        st.warning("⚠️ Using legacy model compatibility mode...")
        with h5py.File(model_path, "r") as f:
            if "model_weights" not in f:
                raise RuntimeError("Cannot find model_weights in .h5 file") from err

        try:
            inputs = Input(shape=(45, 15))
            x = LSTM(64, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(32, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            x = Attention()(x)
            x = Dense(16, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1)(x)

            rebuilt_model = Model(inputs=inputs, outputs=outputs)
            rebuilt_model.load_weights(model_path, skip_mismatch=True, by_name=False)
            return rebuilt_model
        except Exception as rebuild_err:
            raise RuntimeError(f"Failed to rebuild model: {rebuild_err}") from err

model = load_model_with_legacy_support("model.h5", custom_objects={"Attention": Attention})

if not os.path.exists("X_scaler.pkl") or not os.path.exists("y_scaler.pkl"):
    st.error("❌ Scaler files (`X_scaler.pkl` or `y_scaler.pkl`) missing.")
    st.stop()

with open("X_scaler.pkl", "rb") as f:
    X_scaler = pickle.load(f)

with open("y_scaler.pkl", "rb") as f:
    y_scaler = pickle.load(f)

stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Bank Nifty": "^NSEBANK"
}

# -------------------------
# INPUT SECTION
# -------------------------
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_name = st.selectbox("Select Stock", list(stocks.keys()))
    with col2:
        run_btn = st.button("Run Analysis")

ticker = stocks[stock_name]

# -------------------------
# RUN MODEL
# -------------------------
if run_btn:
    df = yf.download(ticker, start="2018-01-01", multi_level_index=False)
    if df.empty:
        st.error(f"Failed to download market data for {ticker}.")
        st.stop()

    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df["MA20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()

    df["Momentum"] = df["Close"] - df["Close"].shift(3)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Trend"] = df["MA5"] - df["MA50"]
    df["Volatility"] = df["Close"].rolling(5).std()

    news_api_key = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", ""))
    news_data = []

    if news_api_key:
        try:
            newsapi = NewsApiClient(api_key=news_api_key)
            articles = newsapi.get_everything(q=stock_name, language="en", page_size=20)
            for a in articles.get("articles", []):
                news_data.append({"Date": a["publishedAt"][:10], "headline": a["title"]})
        except Exception as e:
            st.warning(f"Could not fetch news sentiment: {e}")

    news_df = pd.DataFrame(news_data)

    if not news_df.empty:
        analyzer = SentimentIntensityAnalyzer()
        news_df["Date"] = pd.to_datetime(news_df["Date"])
        news_df["Sentiment"] = news_df["headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
        daily_sent = news_df.groupby("Date")["Sentiment"].mean().reset_index()
    else:
        daily_sent = pd.DataFrame(columns=["Date", "Sentiment"])

    df = pd.merge(df, daily_sent, on="Date", how="left")
    df["Sentiment"] = df["Sentiment"].fillna(0)
    df["Sent_1"] = df["Sentiment"].shift(1)
    df["Sent_2"] = df["Sentiment"].shift(2)

    df["Change"] = df["Close"].diff()
    df = df.dropna().reset_index(drop=True)

    features = [
        "Open", "High", "Low", "Volume", "MA20", "RSI", "MACD",
        "Sentiment", "Sent_1", "Sent_2",
        "Momentum", "MA5", "MA50", "Trend", "Volatility"
    ]

    X = df[features]
    y = df["Change"]

    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y.values.reshape(-1, 1))

    def create_data(X, y, step=45):
        Xs, ys = [], []
        for i in range(step, len(X)):
            Xs.append(X[i-step:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_data(X_scaled, y_scaled, 45)

    split = int(0.8 * len(X_seq))
    X_test = X_seq[split:]
    y_test = y_seq[split:]

    pred = model.predict(X_test)
    pred = y_scaler.inverse_transform(pred)
    actual = y_scaler.inverse_transform(y_test)

    actual_prices = []
    pred_prices = []

    start_idx = len(df) - len(actual)

    for i in range(len(actual)):
        prev_idx = max(0, start_idx + i - 1)
        prev_price = df["Close"].iloc[prev_idx]
        actual_prices.append(float(prev_price + actual[i][0]))
        pred_prices.append(float(prev_price + pred[i][0]))

    mae = mean_absolute_error(actual_prices, pred_prices)
    error_pct = (mae / np.mean(actual_prices)) * 100
    market = "📈 Bullish" if pred_prices[-1] > pred_prices[-2] else "📉 Bearish"

    st.markdown("### 📊 Model Performance")
    k1, k2, k3 = st.columns(3)

    k1.markdown(f"<div class='card'>MAE<br><b>₹{round(mae,2)}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='card'>Error %<br><b>{round(error_pct,2)}%</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='card'>Market<br><b>{market}</b></div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    center = st.columns([1, 3, 1])[1]
    with center:
        st.subheader("📊 Prediction")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actual_prices, name="Actual"))
        fig.add_trace(go.Scatter(y=pred_prices, name="Predicted"))
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig)

        st.subheader("📉 Error")
        errors = np.array(actual_prices) - np.array(pred_prices)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=errors))
        fig2.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig2)

        st.subheader("🔁 Direction")
        actual_dir = np.sign(np.diff(actual_prices))
        pred_dir = np.sign(np.diff(pred_prices))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=actual_dir, name="Actual"))
        fig3.add_trace(go.Scatter(y=pred_dir, name="Predicted"))
        fig3.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig3)

        st.subheader("📊 Candlestick")
        fig4 = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        fig4.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig4)

    st.subheader("📅 Last 5 Days (Actual vs Predicted)")

    dates = df["Date"].iloc[-len(actual_prices):].reset_index(drop=True)

    full_df = pd.DataFrame({
        "Date": dates,
        "Actual Price": actual_prices,
        "Predicted Price": pred_prices
    })

    full_df["Date"] = pd.to_datetime(full_df["Date"]).dt.date

    yesterday = date.today() - timedelta(days=1)
    filtered_df = full_df[full_df["Date"] <= yesterday]
    last5_df = filtered_df.tail(5)

    center = st.columns([1, 2, 1])[1]
    with center:
        st.markdown("<div class='table-card'>", unsafe_allow_html=True)

        for _, row in last5_df.iterrows():
            color = "green" if row["Predicted Price"] > row["Actual Price"] else "red"

            st.markdown(f"""
            <div class='row'>
                <div>{row['Date']}</div>
                <div>₹{round(row['Actual Price'], 2)}</div>
                <div class='{color}'>₹{round(row['Predicted Price'], 2)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🧠 AI Explainability")

    latest = df.iloc[-1]

    if latest["RSI"] > 60:
        st.write("• RSI shows bullish momentum")
    else:
        st.write("• RSI shows weak trend")

    if latest["MACD"] > 0:
        st.write("• MACD indicates upward trend")
    else:
        st.write("• MACD indicates bearish trend")

    if latest["Sentiment"] > 0:
        st.write("• News sentiment is positive")
    else:
        st.write("• News sentiment is negative")
