import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
from logger import logger
from data_loader.yahoo_downloader import sanitize_filename


class DataProcessor:
    """
    [ETL Module]
    Process raw CSV data into a clean, aligned portfolio matrix.
    Includes a 'Bond Pricing Engine' to convert Yields to Synthetic Price Series.
    """

    def __init__(self, raw_path: str = "./data", processed_path: str = "./data_processed"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

    def _load_raw(self, safe_name: str) -> pd.Series:
        """Load 'Close' column from raw CSV using sanitized filename."""
        path = os.path.join(self.raw_path, f"{safe_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Parse dates and set index
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        # Ensure we return a Series, handling potential column name issues
        if 'Close' in df.columns:
            return df['Close']
        else:
            # Fallback for single-column CSVs
            return df.iloc[:, 0]

    def bond_pricing_engine(self, yield_series: pd.Series, duration: float = 20.0, initial_price: float = 100.0) -> pd.Series:
        """
        [Financial Engineering]
        Converts a Yield Series (e.g., 4.5 for 4.5%) into a Synthetic Bond Price Series.
        
        Formula:
        Daily Return ~= (Yield / 100 / 252) - (Duration * Change_in_Yield)
                      = Interest Income     + Capital Gain/Loss
        """
        # 1. Convert yield index (4.5) to decimal (0.045)
        y = yield_series / 100.0
        
        # 2. Calculate Yield Change (Delta y)
        # diff() creates a NaN at the first position, fill with 0
        dy = y.diff().fillna(0)
        
        # 3. Calculate components
        # Interest Income: Based on yesterday's yield. shift(1) creates NaN, fill with y[0]
        interest_income = y.shift(1).fillna(y.iloc[0]) / 252.0
        
        # Capital Gain: Inverse relationship between yield and price
        capital_gain = -duration * dy
        
        # 4. Total Daily Return
        total_return = interest_income + capital_gain
        
        # 5. Construct Price Index
        # (1 + r).cumprod() creates the price path
        price_series = initial_price * (1 + total_return).cumprod()
        
        return price_series

    def cash_pricing_engine(self, yield_series: pd.Series, initial_price: float = 100.0) -> pd.Series:
        """
        [Financial Engineering]
        Converts T-Bill Yield into a Cash Price Series (Risk-free accumulation).
        """
        # Convert yield (e.g., 3.0) to decimal (0.03)
        y = yield_series / 100.0
        
        # Daily Return = Yield / 252 (Simple interest approximation)
        daily_ret = y.shift(1).fillna(y.iloc[0]) / 252.0
        
        price_series = initial_price * (1 + daily_ret).cumprod()
        return price_series

    def process_and_align(self, assets: List[Dict[str, Any]]):
        """
        Main pipeline: Load Raw -> Apply Engines -> Align -> Save.
        Dynamically supports any asset list configuration.
        """
        logger.info("Starting Data Processing & Alignment...")

        try:
            prices = {}

            # 1. Load and transform each asset according to its configuration
            for asset in assets:
                name = asset["name"]
                kind = asset.get("kind", "price")
                engine = asset.get("engine")
                duration = float(asset.get("duration", 20.0))

                safe_name = sanitize_filename(name)
                logger.info(f"Loading raw data for asset '{name}' (file: {safe_name}.csv)...")
                raw_series = self._load_raw(safe_name)

                if kind == "price":
                    price_series = raw_series
                elif kind == "yield":
                    if engine == "bond":
                        price_series = self.bond_pricing_engine(raw_series, duration=duration)
                    elif engine == "cash":
                        price_series = self.cash_pricing_engine(raw_series)
                    else:
                        raise ValueError(f"Unsupported engine for yield asset '{name}': {engine}")
                else:
                    raise ValueError(f"Unsupported asset kind for '{name}': {kind}")

                prices[name] = price_series

            if not prices:
                raise ValueError("No asset price series generated. Check asset configuration and raw data.")

            # 2. Align Data (Inner Join on Date index)
            portfolio_df = pd.DataFrame(prices)
            original_len = len(portfolio_df)
            portfolio_df.dropna(inplace=True)
            new_len = len(portfolio_df)

            logger.info(f"Alignment complete. Rows: {original_len} -> {new_len}")
            logger.info(f"Date Range: {portfolio_df.index.min().date()} to {portfolio_df.index.max().date()}")
            logger.info(f"Assets: {list(portfolio_df.columns)}")

            # 3. Save Final Artifact
            output_file = os.path.join(self.processed_path, "aligned_assets.csv")
            portfolio_df.to_csv(output_file)
            logger.info(f"Successfully saved processed data to: {output_file}")
            
            # Preview for verification
            print("\n" + "="*40)
            print("PROCESSED DATA PREVIEW")
            print("="*40)
            print(portfolio_df.head())
            print("...")
            print(portfolio_df.tail())
            print("="*40)

        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            raise  # Re-raise to stop execution if ETL fails


if __name__ == "__main__":
    # For local testing with a default 4-asset configuration (matching main_download.py)
    default_assets: List[Dict[str, Any]] = [
        {"name": "Stocks", "ticker": "^NDX", "kind": "price"},
        {"name": "Gold", "ticker": "^XAU", "kind": "price"},
        {"name": "Bonds", "ticker": "^TYX", "kind": "yield", "engine": "bond", "duration": 20.0},
        {"name": "Cash", "ticker": "^IRX", "kind": "yield", "engine": "cash"},
    ]
    processor = DataProcessor()
    processor.process_and_align(default_assets)
