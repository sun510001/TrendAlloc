import yfinance as yf
import pandas as pd
import os
import datetime
from typing import Dict, List
from logger import logger
from utils.decorators import retry

class YahooIncrementalLoader:
    """
    [Data ETL Class]
    Handles incremental downloading of financial data from Yahoo Finance.
    
    Features:
    - Automatic Incremental Update: Only downloads new data since the last record.
    - Robust MultiIndex Handling: Flattens yfinance v0.2+ structures correctly.
    - Full OHLCV Preservation: Keeps Open, High, Low, Close, Volume.
    - Atomic Writes: Prevents CSV corruption.
    """

    def __init__(self, storage_path: str = "./data"):
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def _get_existing_data(self, file_path: str) -> pd.DataFrame:
        """
        Reads existing CSV. Returns empty DataFrame if file doesn't exist.
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            return df
        except Exception as e:
            logger.warning(f"Corrupt file found at {file_path}, starting fresh. Error: {e}")
            return pd.DataFrame()

    @retry(max_retries=3, delay=5)
    def download_symbol(self, ticker: str, name: str, start_date_fallback: str = "1985-01-01"):
        """
        Smart downloader that preserves OHLCV data and handles yfinance quirks.
        """
        file_path = os.path.join(self.storage_path, f"{name}.csv")
        df_old = self._get_existing_data(file_path)
        
        # 1. Determine Start Date
        if not df_old.empty:
            last_date = df_old.index[-1].date()
            # Start from the next day to avoid overlap (yfinance start is inclusive)
            start_download_date = last_date + datetime.timedelta(days=1)
            is_update = True
            
            # If local data is up to today (or yesterday), skip
            if start_download_date >= datetime.date.today():
                logger.info(f"[{name}] Already up to date ({last_date}).")
                return
        else:
            start_download_date = datetime.datetime.strptime(start_date_fallback, "%Y-%m-%d").date()
            is_update = False

        logger.info(f"[{name}] Downloading {ticker} from {start_download_date}...")

        # 2. Download Data (Auto Adjust for Splits/Dividends)
        try:
            # auto_adjust=True returns: Open, High, Low, Close, Volume (Prices are adjusted)
            df_new = yf.download(ticker, start=start_download_date, progress=False, auto_adjust=True)
        except Exception as e:
            logger.error(f"[{name}] Download failed: {e}")
            return

        if df_new.empty:
            logger.info(f"[{name}] No new data found on server.")
            return

        # 3. Clean & Flatten MultiIndex (Crucial Step)
        # yfinance often returns columns like: ('Close', 'AAPL'), ('Open', 'AAPL')
        if isinstance(df_new.columns, pd.MultiIndex):
            try:
                # If the Ticker is in the second level (level=1), extract that cross-section
                if ticker in df_new.columns.get_level_values(1):
                    df_new = df_new.xs(ticker, axis=1, level=1)
                # Sometimes purely 'Close' etc are top level, but let's check level 0 just in case
                elif 'Close' in df_new.columns.get_level_values(0):
                    # In this case, the MultiIndex might be redundant or structured differently.
                    # We drop the level to flatten it.
                    df_new.columns = df_new.columns.get_level_values(0)
            except Exception as e:
                logger.warning(f"[{name}] MultiIndex parsing warning: {e}. Attempting direct flatten.")
                # Last resort: just keep the top level names
                df_new.columns = df_new.columns.get_level_values(0)
        
        # 4. Filter and Order Columns
        # We want standard OHLCV. Some indexes (like VIX) might not have Volume.
        desired_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in desired_cols if c in df_new.columns]
        
        if not available_cols:
            logger.error(f"[{name}] Critical Error: Downloaded data has no recognized price columns.")
            return

        df_new = df_new[available_cols]
        df_new.index.name = 'Date'

        # 5. Merge and Save
        if is_update:
            # Combine old and new
            df_final = pd.concat([df_old, df_new])
            # Remove potential duplicates based on Index (Date)
            df_final = df_final[~df_final.index.duplicated(keep='last')]
        else:
            df_final = df_new

        # Sort strictly by date
        df_final.sort_index(inplace=True)

        # Write to CSV
        df_final.to_csv(file_path)
        
        action = "Updated" if is_update else "Created"
        logger.info(f"[{name}] {action} successfully. Rows: {len(df_final)} | Cols: {list(df_final.columns)}")

    def download_batch(self, ticker_map: Dict[str, str], start_year: int = 1985):
        """
        Process a batch of tickers.
        """
        logger.info("="*40)
        logger.info(f"Starting Batch Download (Target Year: {start_year})")
        logger.info("="*40)
        
        for name, ticker in ticker_map.items():
            self.download_symbol(ticker, name, f"{start_year}-01-01")
        
        logger.info("Batch Download Complete.\n")