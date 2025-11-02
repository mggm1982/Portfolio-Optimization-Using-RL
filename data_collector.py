import yfinance as yf
import pandas as pd
import os
import time
from config import NIFTY_50_STOCKS, NIFTY_50_INDEX, START_DATE, END_DATE, DATA_DIR

def download_stock_data():
    """
    Downloads historical stock data for the given tickers and saves them to CSV files.
    Includes robust retry logic to handle rate limiting.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    print(f"Downloading Stock Data from {START_DATE} to {END_DATE}...")
    
    # Combine stocks and index for downloading
    all_tickers = NIFTY_50_STOCKS + [NIFTY_50_INDEX]
    
    successful_downloads = 0
    total_tickers = len(all_tickers)

    for i, ticker in enumerate(all_tickers):
        print(f"Downloading {ticker} ({i+1}/{total_tickers})...", end="", flush=True)
        
        for attempt in range(1, 4):  # Try up to 3 times
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=START_DATE, end=END_DATE, interval="1d")
                
                if data.empty:
                    print(f" FAILED. No data found for {ticker}.")
                    break  # Don't retry if no data exists

                file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
                data.to_csv(file_path)
                print(f" SUCCESS. Saved to {file_path}")
                
                successful_downloads += 1
                time.sleep(1) # Small delay to be polite
                break  # Success, move to next ticker

            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg:
                    wait_time = 5 * attempt # Wait 5, 10, 15 seconds
                    print(f" FAILED (Rate Limit). Retrying in {wait_time}s... (Attempt {attempt}/3)")
                    time.sleep(wait_time)
                else:
                    print(f" FAILED for {ticker}. Error: {e}")
                    break # Don't retry on other errors
            
            if attempt == 3:
                print(f" FAILED permanently for {ticker} after 3 attempts.")

    print("-" * 30)
    print(f"Download Summary:")
    print(f"Successfully downloaded {successful_downloads} of {total_tickers} tickers.")
    print(f"Failed to download {total_tickers - successful_downloads} tickers.")
    print("-" * 30)

    if successful_downloads < total_tickers:
        print("Warning: Some data failed to download. Backtest may be incomplete.")
    else:
        print("Data collection complete.")

if __name__ == "__main__":
    download_stock_data()

