import akshare as ak
import pandas as pd
import os
from typing import List, Dict, Optional
from logger import logger
from utils.decorators import retry

class USMarketLoader:
    """
    A class to download, clean, and store US stock data using AkShare.
    Designed to prepare data for VectorBT backtesting.
    """

    def __init__(self, storage_path: str = "./data"):
        """
        Initialize the loader.

        Args:
            storage_path (str): Directory where CSV files will be saved.
        """
        self.storage_path = storage_path
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Create the storage directory if it does not exist."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            logger.info(f"Created storage directory at: {self.storage_path}")

    @retry(max_retries=5, delay=3)
    def _fetch_single_symbol(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol from AkShare.
        
        Note: AkShare 'stock_us_daily' returns Chinese column names by default.
        We need to map them to English for standardization.

        Args:
            symbol (str): The ticker symbol (e.g., 'SPY').
            start_date (str): Format 'YYYYMMDD'.
            end_date (str): Format 'YYYYMMDD'.

        Returns:
            pd.DataFrame: Cleaned DataFrame with standard OHLCV columns and datetime index.
        """
        logger.info(f"Fetching data for {symbol}...")
        
        # AkShare API call for US stocks (daily data, forward adjusted)
        # adjust='qfq' means forward-adjusted (crucial for backtesting)
        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        
        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Rename columns map (AkShare specific)
        # Note: Actual column names from AkShare might vary slightly, 
        # usually they are: date, open, high, low, close, volume...
        # We enforce a standard map.
        column_map = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Normalize column names to lowercase for mapping, then rename
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns=column_map, inplace=True)

        # Ensure Date is datetime object and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Filter by date range
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df = df.loc[mask]

        # Type conversion ensuring numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def download_batch(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols and save them to CSV.

        Args:
            symbols (List[str]): List of ticker symbols.
            start_date (str): Start date 'YYYYMMDD'.
            end_date (str): End date 'YYYYMMDD'.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbol to its DataFrame.
        """
        results = {}
        for symbol in symbols:
            try:
                df = self._fetch_single_symbol(symbol, start_date, end_date)
                
                # Save to CSV
                file_path = os.path.join(self.storage_path, f"{symbol}.csv")
                df.to_csv(file_path)
                logger.info(f"Successfully saved {symbol} to {file_path} (Records: {len(df)})")
                
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
        
        return results

    def load_aligned_close_price(self, symbols: List[str]) -> pd.DataFrame:
        """
        Load locally saved CSVs and combine them into a single DataFrame 
        containing only 'Close' prices, aligned by Date index.
        
        This is the format VectorBT prefers for simple portfolio backtesting.
        
        Args:
            symbols (List[str]): List of symbols to load.

        Returns:
            pd.DataFrame: Index=Date, Columns=Symbols (Close Prices).
        """
        close_data = pd.DataFrame()

        for symbol in symbols:
            file_path = os.path.join(self.storage_path, f"{symbol}.csv")
            if not os.path.exists(file_path):
                logger.warning(f"File not found for {symbol}, skipping alignment.")
                continue
            
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            close_data[symbol] = df['Close']
        
        # Clean: Drop rows where any symbol is missing (Wait for all markets to be open)
        # or fillna(method='ffill') depending on strategy strictness.
        # For 'Permanent Portfolio', we usually prefer 'dropna' to align trading days exactly.
        original_len = len(close_data)
        close_data.dropna(inplace=True)
        new_len = len(close_data)
        
        if original_len != new_len:
            logger.info(f"Aligned data: Dropped {original_len - new_len} rows due to mismatched dates.")

        return close_data