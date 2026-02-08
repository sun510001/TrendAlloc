import pandas as pd
import numpy as np
import os
from logger import logger

class DataProcessor:
    """
    Process raw CSV data into a clean, aligned portfolio matrix.
    Includes a 'Bond Pricing Engine' to convert Yields to Price Series.
    """

    def __init__(self, raw_path: str = "./data", processed_path: str = "./data_processed"):
        self.raw_path = raw_path
        self.processed_path = processed_path
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

    def _load_raw(self, name: str) -> pd.Series:
        """Load 'Close' column from raw CSV."""
        path = os.path.join(self.raw_path, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        # Handle potential duplicates or unordered index
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        return df['Close']

    def bond_pricing_engine(self, yield_series: pd.Series, duration: float = 20.0, initial_price: float = 100.0) -> pd.Series:
        """
        Converts a Yield Series (e.g., 4.5 for 4.5%) into a Synthetic Bond Price Series.
        
        Logic:
         Daily Return ~= (Yield / 100 / 252) - (Duration * Change_in_Yield)
                       = Interest Income     + Capital Gain/Loss
        """
        # 1. Convert yield from index (4.5) to decimal (0.045)
        y = yield_series / 100.0
        
        # 2. Calculate Yield Change (Delta y)
        dy = y.diff().fillna(0)
        
        # 3. Calculate components
        interest_income = y.shift(1) / 252.0  # Earn yield based on yesterday's rate
        capital_gain = -duration * dy
        
        # 4. Total Daily Return
        total_return = interest_income + capital_gain
        
        # 5. Construct Price Index
        price_series = initial_price * (1 + total_return).cumprod()
        return price_series

    def cash_pricing_engine(self, yield_series: pd.Series, initial_price: float = 100.0) -> pd.Series:
        """
        Converts T-Bill Yield into a Cash Price Series.
        Risk-free accumulation.
        """
        # Convert yield (e.g., 3.0) to decimal (0.03)
        y = yield_series / 100.0
        
        # Daily Return = Yield / 252 (Simple interest approx)
        daily_ret = y.shift(1) / 252.0
        daily_ret = daily_ret.fillna(0)
        
        price_series = initial_price * (1 + daily_ret).cumprod()
        return price_series

    def process_and_align(self):
        """
        Main pipeline: Load Raw -> Apply Engines -> Align -> Save.
        """
        logger.info("Starting Data Processing & Alignment...")

        try:
            # 1. Load Raw Data
            # Note: Names must match what was used in Downloader
            nasdaq_raw = self._load_raw("QQQ_Proxy") # ^NDX
            gold_raw = self._load_raw("GOLD_Proxy")  # GC=F
            treasury_yield = self._load_raw("US30Y_Yield") # ^TYX
            tbill_yield = self._load_raw("CASH_Yield")     # ^IRX

            # 2. Apply Pricing Engines
            logger.info("Calculating synthetic asset prices...")
            
            # Equity & Gold (Direct usage)
            equity_price = nasdaq_raw
            gold_price = gold_raw
            
            # Long Bond (Synthetic TLT)
            # Duration ~18-20y for long bonds. We use 20.
            bond_price = self.bond_pricing_engine(treasury_yield, duration=20.0)
            
            # Cash (Synthetic SHV)
            cash_price = self.cash_pricing_engine(tbill_yield)

            # 3. Align Data (Inner Join to find common trading days)
            # We use concat with axis=1, then dropna
            portfolio_df = pd.concat([
                equity_price, 
                bond_price, 
                gold_price, 
                cash_price
            ], axis=1)
            
            portfolio_df.columns = ["QQQ", "TLT", "GLD", "SHV"] # Naming for Strategy
            
            # Remove NaN (Align dates where all markets were open)
            original_len = len(portfolio_df)
            portfolio_df.dropna(inplace=True)
            new_len = len(portfolio_df)
            
            logger.info(f"Alignment complete. Rows: {original_len} -> {new_len}")
            
            # 4. Save Final Artifact
            output_file = os.path.join(self.processed_path, "permanent_portfolio_aligned.csv")
            portfolio_df.to_csv(output_file)
            logger.info(f"Successfully saved processed data to: {output_file}")
            
            # Preview
            print("\n" + "="*40)
            print("PROCESSED DATA PREVIEW (1985+)")
            print("="*40)
            print(portfolio_df.head())
            print("...")
            print(portfolio_df.tail())
            print("="*40)

        except Exception as e:
            logger.exception(f"Processing failed: {e}")