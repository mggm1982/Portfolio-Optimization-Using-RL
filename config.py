import os

# --- Stock Tickers ---
# Use the .NS suffix for National Stock Exchange (NSE)
NIFTY_50_STOCKS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "LT.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS", "HINDUNILVR.NS"
]
# Ticker for the Nifty 50 Index
NIFTY_50_INDEX = "^NSEI"

# --- Data Settings ---
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"

# --- Directory Settings ---
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# --- Environment Settings ---
# How many of the last days of data to use for testing
TEST_SPLIT_DAYS = 500
# Number of past days of data to observe at each step
LOOKBACK_WINDOW = 60
# Initial cash in the portfolio
INITIAL_BALANCE = 1000000
# Transaction fee (e.g., 0.001 for 0.1%)
TRANSACTION_FEE_PERCENT = 0.0005

# --- Model Training Settings ---
# Total steps to train the agent
TOTAL_TIMESTEPS = 100000


