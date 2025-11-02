import os
import pandas as pd
from typing import List, Tuple
from datetime import datetime

# Import user config and data collector
from config import (
    NIFTY_50_STOCKS, NIFTY_50_INDEX, START_DATE, END_DATE,
    DATA_DIR, TEST_SPLIT_DAYS, LOOKBACK_WINDOW,
    INITIAL_BALANCE, TRANSACTION_FEE_PERCENT, TOTAL_TIMESTEPS
)
import data_collector as dc


def ensure_data():
    """Ensure all CSVs exist; if missing, trigger download."""
    tickers = NIFTY_50_STOCKS + [NIFTY_50_INDEX]
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    missing = []
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            missing.append(t)
    if missing:
        print(f"[data_interface] Missing data for: {missing}. Attempting download...")
        try:
            dc.download_stock_data()
        except Exception as e:
            print("[data_interface] Warning: download failed.", e)


def _read_close_csv(ticker: str) -> pd.Series:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV for {ticker} not found.")
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if date_col is None or "Close" not in df.columns:
        raise ValueError(f"Unexpected CSV format for {ticker}.")
    s = pd.Series(df["Close"].values, index=pd.to_datetime(df[date_col]), name=ticker)
    return s.sort_index().loc[START_DATE:END_DATE]


def load_prices() -> Tuple[pd.DataFrame, pd.Series]:
    """Return (10-stock prices DF, NIFTY-50 series)."""
    ensure_data()
    stock_series = [_read_close_csv(t) for t in NIFTY_50_STOCKS]
    prices_df = pd.concat(stock_series, axis=1).ffill().dropna(how="all")
    nifty_series = _read_close_csv(NIFTY_50_INDEX).reindex(prices_df.index).ffill()
    return prices_df, nifty_series


def get_cfg():
    return dict(
        SYMBOLS=NIFTY_50_STOCKS,
        INDEX=NIFTY_50_INDEX,
        START_DATE=START_DATE,
        END_DATE=END_DATE,
        DATA_DIR=DATA_DIR,
        TEST_SPLIT_DAYS=TEST_SPLIT_DAYS,
        LOOKBACK_WINDOW=LOOKBACK_WINDOW,
        INITIAL_BALANCE=INITIAL_BALANCE,
        TRANSACTION_FEE_PERCENT=TRANSACTION_FEE_PERCENT,
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
    )
