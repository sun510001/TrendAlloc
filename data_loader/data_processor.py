import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from logger import logger
from data_loader.yahoo_downloader import YahooIncrementalLoader


class DataProcessor:
    """
    [ETL Module]
    Responsible for processing raw CSV data into a clean, synchronized portfolio matrix.
    
    This class includes financial engineering engines to convert raw market yields into 
    synthetic total return price series and handles temporal alignment across multiple assets.
    """

    def __init__(self, raw_path: str = "./data", processed_path: str = "./data_processed") -> None:
        """
        Initialize the DataProcessor with paths for raw and processed data.

        Args:
            raw_path (str): Directory containing raw asset CSV files. Defaults to "./data".
            processed_path (str): Directory where processed artifacts will be saved. 
                                 Defaults to "./data_processed".
        """
        self.raw_path: str = raw_path
        self.processed_path: str = processed_path
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

    def _load_raw(self, safe_name: str) -> pd.Series:
        """
        Load the 'Close' price column from a raw asset CSV file.

        Args:
            safe_name (str): The sanitized filename (without extension).

        Returns:
            pd.Series: A time series of closing prices indexed by Date.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
        """
        path = os.path.join(self.raw_path, f"{safe_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")
        
        # Parse dates and set index
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        
        # Remove duplicates and ensure chronological order
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        # Extract Closing price
        if 'Close' in df.columns:
            return df['Close']
        else:
            # Fallback for single-column data files
            return df.iloc[:, 0]

    def bond_pricing_engine(self, yield_series: pd.Series, duration: float = 20.0, initial_price: float = 100.0) -> pd.Series:
        """
        [Financial Engineering]
        Converts a Yield Series (e.g., 4.5 for 4.5%) into a Synthetic Total Return Bond Price Series.
        
        The approximation formula used:
        Daily Return ~= (Yield / 100 / 252) - (Duration * Change_in_Yield)
                      = Interest Income     + Capital Gain/Loss

        Args:
            yield_series (pd.Series): Time series of yields in percentage format.
            duration (float): Approximate duration of the bond index. Defaults to 20.0.
            initial_price (float): Base price for the synthetic index. Defaults to 100.0.

        Returns:
            pd.Series: A synthetic price series starting from initial_price.
        """
        # Convert yield index (e.g. 4.5) to decimal (0.045)
        y = yield_series / 100.0
        
        # Calculate daily yield changes
        dy = y.diff().fillna(0)
        
        # Components of daily total return
        interest_income = y.shift(1).fillna(y.iloc[0]) / 252.0
        capital_gain = -duration * dy
        
        total_daily_return = interest_income + capital_gain
        
        # Cumulative product to build the price index
        price_series = initial_price * (1 + total_daily_return).cumprod()
        
        return price_series

    def cash_pricing_engine(self, yield_series: pd.Series, initial_price: float = 100.0) -> pd.Series:
        """
        [Financial Engineering]
        Converts a short-term Treasury Bill Yield into a Cash Price Series.
        
        Args:
            yield_series (pd.Series): Time series of short-term yields in percentage format.
            initial_price (float): Base price for the synthetic index. Defaults to 100.0.

        Returns:
            pd.Series: A risk-free accumulation price series.
        """
        y = yield_series / 100.0
        
        # Daily return approximation using simple interest
        daily_ret = y.shift(1).fillna(y.iloc[0]) / 252.0
        
        price_series = initial_price * (1 + daily_ret).cumprod()
        return price_series

    def process_and_align(self, assets: List[Dict[str, Any]]) -> None:
        """
        Main ETL pipeline: Load raw data, apply pricing engines, and synchronize dates.
        
        This method iterates through all configured assets, transforms them into 
        comparable price series, aligns them by shared trading dates, and saves 
        the final matrix to a CSV.

        Args:
            assets (List[Dict[str, Any]]): List of asset configurations including 
                                          name, kind, and engine specifications.

        Raises:
            ValueError: If an asset configuration is invalid or missing required data.
        """
        logger.info("Starting Multi-Asset Data Processing & Alignment...")

        try:
            prices: Dict[str, pd.Series] = {}

            for asset in assets:
                name = asset["name"]
                kind = asset.get("kind", "price")
                engine = asset.get("engine")

                # Default duration for bond yield assets; ignored for others
                duration: float = 20.0
                if kind == "yield" and engine == "bond":
                    raw_duration = asset.get("duration", 20.0)
                    if raw_duration is not None:
                        duration = float(raw_duration)

                # Use the same sanitization rule as the downloader
                safe_name = YahooIncrementalLoader.sanitize_filename(name)
                logger.info(f"Processing asset '{name}' (source: {safe_name}.csv)...")
                
                raw_series = self._load_raw(safe_name)

                if kind == "price":
                    price_series = raw_series
                elif kind == "yield":
                    if engine == "bond":
                        price_series = self.bond_pricing_engine(raw_series, duration=duration)
                    elif engine == "cash":
                        price_series = self.cash_pricing_engine(raw_series)
                    else:
                        raise ValueError(f"Unknown engine '{engine}' for asset '{name}'")
                else:
                    raise ValueError(f"Unsupported asset kind '{kind}' for asset '{name}'")

                prices[name] = price_series

            if not prices:
                raise ValueError("No assets were successfully processed.")

            # Create synchronized DataFrame (Inner Join on Date index)
            portfolio_df = pd.DataFrame(prices)
            original_len = len(portfolio_df)
            portfolio_df.dropna(inplace=True)
            final_len = len(portfolio_df)

            logger.info(f"Alignment complete. Rows: {original_len} -> {final_len}")
            logger.info(f"Available range: {portfolio_df.index.min().date()} to {portfolio_df.index.max().date()}")

            # Save the synchronized price matrix
            output_file = os.path.join(self.processed_path, "aligned_assets.csv")
            portfolio_df.to_csv(output_file)
            logger.info(f"Aligned assets saved successfully to: {output_file}")

        except Exception as e:
            logger.exception(f"Data pipeline processing failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Example execution for local testing
    sample_assets = [
        {"name": "Stocks", "ticker": "^NDX", "kind": "price"},
        {"name": "Gold", "ticker": "^XAU", "kind": "price"},
        {"name": "Bonds", "ticker": "^TYX", "kind": "yield", "engine": "bond", "duration": 20.0},
        {"name": "Cash", "ticker": "^IRX", "kind": "yield", "engine": "cash"},
    ]
    processor = DataProcessor()
    processor.process_and_align(sample_assets)
