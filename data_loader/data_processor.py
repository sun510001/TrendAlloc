import pandas as pd
import numpy as np
import os
from logger import logger

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

    def _load_raw(self, name: str) -> pd.Series:
        """
        Load 'Close' column from raw CSV.
        """
        path = os.path.join(self.raw_path, f"{name}.csv")
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

    def process_and_align(self):
        """
        Main pipeline: Load Raw -> Apply Engines -> Align -> Save.
        """
        logger.info("Starting Data Processing & Alignment...")

        try:
            # 1. Load Raw Data
            # CRITICAL: Names must match the keys in your TICKER_MAP
            nasdaq_raw = self._load_raw("Stocks")  # Corresponds to ^NDX
            gold_raw = self._load_raw("Gold")      # Corresponds to ^XAU
            bond_yield = self._load_raw("Bonds")   # Corresponds to ^TYX (Yield)
            cash_yield = self._load_raw("Cash")    # Corresponds to ^IRX (Yield)

            # 2. Apply Pricing Engines
            logger.info("Calculating synthetic asset prices from yields...")
            
            # Equity: Use raw price directly
            equity_price = nasdaq_raw
            
            # Gold: Use raw price directly (^XAU is an index price, not yield)
            gold_price = gold_raw
            
            # Bonds: Convert 30Y Yield (^TYX) to Price
            # Duration=20 is a standard approximation for 30Y Treasury Bonds
            bond_price = self.bond_pricing_engine(bond_yield, duration=20.0)
            
            # Cash: Convert 3-Month Yield (^IRX) to Price
            cash_price = self.cash_pricing_engine(cash_yield)

            # 3. Align Data (Inner Join)
            # This automatically handles the "Gold has no holidays" issue.
            # concat(axis=1) matches indices. dropna() removes any row where ANY asset is missing.
            # Since stocks have the most holidays, the result will align to the stock market calendar.
            portfolio_df = pd.concat([
                equity_price, 
                bond_price, 
                gold_price, 
                cash_price
            ], axis=1)
            
            # Rename columns to standard strategy names
            portfolio_df.columns = ["Stocks", "Bonds", "Gold", "Cash"]
            
            # Remove NaN (Align dates where all markets were open)
            original_len = len(portfolio_df)
            portfolio_df.dropna(inplace=True)
            new_len = len(portfolio_df)
            
            logger.info(f"Alignment complete. Rows: {original_len} -> {new_len}")
            logger.info(f"Date Range: {portfolio_df.index.min().date()} to {portfolio_df.index.max().date()}")

            # 4. Save Final Artifact
            output_file = os.path.join(self.processed_path, "permanent_portfolio_aligned.csv")
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
            raise # Re-raise to stop execution if ETL fails

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_and_align()