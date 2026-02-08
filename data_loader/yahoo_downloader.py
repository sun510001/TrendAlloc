import yfinance as yf
import pandas as pd
import os
import datetime
from typing import List, Dict
from logger import logger
from utils.decorators import retry

class YahooIncrementalLoader:
    """
    Handles incremental downloading of financial data from Yahoo Finance.
    Saves raw data to CSV.
    """

    def __init__(self, storage_path: str = "./data"):
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def _get_last_date(self, file_path: str) -> datetime.date:
        """
        Reads the CSV to find the last recorded date.
        Returns None if file doesn't exist or is empty.
        """
        if not os.path.exists(file_path):
            return None
        try:
            # Read only the index (Date) to save memory
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            if df.empty:
                return None
            return df.index[-1].date()
        except Exception as e:
            logger.error(f"Error reading existing file {file_path}: {e}")
            return None

    @retry(max_retries=3, delay=5)
    def download_symbol(self, ticker: str, name: str, start_date_fallback: str = "1985-01-01"):
        """
        Downloads data for a single symbol incrementally.
        
        Args:
            ticker (str): Yahoo Finance ticker (e.g., '^NDX').
            name (str): Local filename alias (e.g., 'QQQ_Proxy').
            start_date_fallback (str): Start date if no local data exists.
        """
        file_path = os.path.join(self.storage_path, f"{name}.csv")
        last_date = self._get_last_date(file_path)
        
        today = datetime.date.today()
        
        # Determine download start date
        if last_date:
            if last_date >= today - datetime.timedelta(days=1):
                logger.info(f"[{name}] Data is up to date ({last_date}). Skipping.")
                return
            
            # Start from the next day
            start_download = last_date + datetime.timedelta(days=1)
            is_update = True
            logger.info(f"[{name}] Found existing data. Updating from {start_download}...")
        else:
            start_download = datetime.datetime.strptime(start_date_fallback, "%Y-%m-%d").date()
            is_update = False
            logger.info(f"[{name}] No local data. Downloading full history from {start_download}...")

        # Fetch data
        # auto_adjust=True handles splits/dividends for stocks, crucial for long term
        df_new = yf.download(ticker, start=start_download, progress=False, auto_adjust=True)

        if df_new.empty:
            logger.warning(f"[{name}] No new data found for ticker {ticker}.")
            return

        # Formatting
        df_new.index.name = 'Date'
        # Ensure we only keep standard columns to avoid MultiIndex issues in yfinance
        if 'Close' in df_new.columns:
             # Handle yfinance v0.2+ structure if it returns multi-level columns
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new = df_new.xs(ticker, axis=1, level=1, drop_level=True)
        
        # Save logic
        if is_update:
            # Append without header
            df_new.to_csv(file_path, mode='a', header=False)
            logger.info(f"[{name}] Appended {len(df_new)} records.")
        else:
            # Write new file
            df_new.to_csv(file_path)
            logger.info(f"[{name}] Saved {len(df_new)} records to new file.")

    def download_batch(self, ticker_map: Dict[str, str], start_year: int = 1985):
        """
        Batch process a dictionary of {Name: Ticker}.
        """
        start_date = f"{start_year}-01-01"
        for name, ticker in ticker_map.items():
            self.download_symbol(ticker, name, start_date)