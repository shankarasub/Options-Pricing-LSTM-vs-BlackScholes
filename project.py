import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Black-Scholes Pricing Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

# LSTM Model for Predicting Option Prices
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

def train_lstm_model(data):
    X_train, y_train, scaler = prepare_data(data)
    model = create_lstm_model()
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    return model, scaler

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(period="5y")
    return stock_hist

# Streamlit App Styling: Custom HTML and CSS
st.markdown("""
    <style>
    .main {
        background-color: black;
        color: white;
    }
    .stButton>button {
        background-color: red;
        color: white;
        border-radius: 10px;
        width: 200px;
        height: 50px;
        font-size: 16px;
    }
    .price-box {
        background-color: #444;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        color: #fff;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .price-box p {
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main App Title
st.markdown('<h1 style="color: red;">Options Pricing Black Scholes vs LSTM</h1>', unsafe_allow_html=True)
assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX", "META", "NVDA", "BABA", "INTC", "CSCO", "CMCSA", "PEP",
          "ADBE", "AVGO", "TXN", "QCOM", "TMUS", "VZ", "T", "AMAT", "MU", "ADI", "LRCX", "MRVL", "TSM", "ASML",
          "AMZN", "IBM", "ORCL", "CRM", "PYPL", "ACN", "UBER", "SQ", "SHOP", "DOCU", "WORK", "TWLO", "OKTA",
          "CRWD", "WDAY", "ZM", "ROKU", "DDOG", "ETSY", "PINS", "SNAP", "UBER", "LYFT", "SPOT", "BIDU", "JD",
          "NTES", "BILI", "IQ", "TME", "EDU", "LK", "PDD", "NIO", "XPEV", "LI", "BYND", "PTON", "PLTR", "RBLX",
          "HUM", "CI", "UNH", "ANTM", "CNC", "MOH", "GILD", "REGN", "VRTX", "BIIB", "AMGN", "ILMN", "EXAS",
          "CRSP", "EDIT", "NTLA", "NVAX", "MRNA", "BNTX", "CVAC", "DHR", "TMO", "BDX", "ABT", "ISRG", "EW"]

# Sidebar Inputs
st.sidebar.header("Options Parameters")
stock_ticker = st.sidebar.selectbox("Select Stock Ticker", assets)
strike_price = st.sidebar.number_input("Strike Price", value=150)
expiry = st.sidebar.date_input("Expiration Date")
volatility = st.sidebar.slider("Implied Volatility (%)", min_value=10, max_value=100, value=20) / 100
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=1.0) / 100
option_type = st.sidebar.selectbox("Option Type", ("call", "put"))

# Fetch Stock Data
stock_data = get_stock_data(stock_ticker)
stock_price = stock_data['Close'].iloc[-1]

# Time to Expiry calculation
time_to_expiry = (pd.Timestamp(expiry) - pd.Timestamp.today()).days / 365.0

# Button to Train and Predict
if st.sidebar.button("Train and Predict"):
    if time_to_expiry <= 0:
        st.warning("Expiration date must be in the future.")
    else:
        # First, calculate the Black-Scholes price
        bs_price = black_scholes(stock_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)

        # Then, train the LSTM model and make the prediction
        stock_hist_data = stock_data["Close"].values.reshape(-1, 1)
        lstm_model, scaler = train_lstm_model(stock_hist_data)

        # Predict using the last 60 days of stock data
        last_60_days = stock_hist_data[-60:]
        scaled_last_60_days = scaler.transform(last_60_days)
        X_test = np.reshape(scaled_last_60_days, (1, 60, 1))
        lstm_pred = lstm_model.predict(X_test)
        lstm_pred_price = scaler.inverse_transform(lstm_pred)

        # Check if the option is out-of-the-money and explain why its value is low
        explanation = ""
        if option_type == "call" and stock_price <= strike_price:
            explanation = "(Worthless because the stock price is below the strike price.)"
        elif option_type == "put" and stock_price >= strike_price:
            explanation = "(Worthless because the stock price is above the strike price.)"

        # Display both results together in visually appealing boxes
        st.markdown(f"""
        <div class="price-box">
            <p>Black-Scholes Price for {option_type.capitalize()} Option:</p>
            <p>${bs_price:.2f} {explanation}</p>
        </div>
        <div class="price-box">
            <p>LSTM Predicted Price:</p>
            <p>${lstm_pred_price[0][0]:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# Visualize Stock Price History
st.subheader(f"Historical Data for {stock_ticker}")
fig = go.Figure()

# Add stock price history line
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing Price'))

# Add marker for the most recent closing price
fig.add_trace(go.Scatter(x=[stock_data.index[-1]], y=[stock_price], mode='markers',
                         marker=dict(color='red', size=10), name='Current Price'))

# Update layout for better readability
fig.update_layout(
    title=f'{stock_ticker} Closing Price History',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    showlegend=True
)

st.plotly_chart(fig)
