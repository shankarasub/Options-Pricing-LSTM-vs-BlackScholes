This project compares two approaches for pricing financial options:
Black-Scholes Model: A widely-used mathematical model for calculating theoretical prices of options.
LSTM (Long Short-Term Memory): A deep learning model that uses historical stock price data to predict future prices and estimate option values.
The goal of this project is to demonstrate both traditional finance modeling and machine learning techniques for option pricing and compare their outcomes.

KEY FEATURES:
Black-Scholes Pricing: Computes option prices based on theoretical assumptions such as constant volatility and a normal distribution of returns.
LSTM-Based Prediction: Trains an LSTM model on historical stock price data to predict future stock prices and option prices.
Interactive Web Interface: The app allows users to choose stock tickers, adjust strike price, implied volatility, and time to expiry, and see the Black-Scholes and LSTM-predicted option prices side by side.
Stock Price Visualization: Displays the historical price data of the selected stock, including the most recent closing price.

HOW THE MODELS WORK:
1. Black-Scholes Model:
The Black-Scholes model is an analytical solution for pricing European call and put options. It takes into account the stock price, strike price, volatility, time to expiry, and the risk-free interest rate to calculate the theoretical option price.
2. LSTM Model:
An LSTM model is a type of recurrent neural network (RNN) suitable for time series prediction. This project uses an LSTM model to predict future stock prices based on historical stock data, and these predictions are used to estimate the option price.

TECHNOLOGIS USED:
Python: Main programming language.
Streamlit: For building the web app interface.
yFinance: For fetching historical stock price data.
NumPy & Pandas: For data manipulation and analysis.
SciPy: For implementing the Black-Scholes formula.
TensorFlow (Keras): For building and training the LSTM model.
Plotly: For visualizing stock price data.
