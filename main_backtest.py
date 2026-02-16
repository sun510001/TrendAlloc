import os
from strategies.backtest_engine import PermanentPortfolioStrategy
from logger import logger
from strategies.algorithms import my_new_rebalance


def main():
    # --- 1. Configuration Settings ---
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data_processed')
    DATA_FILE = os.path.join(DATA_DIR, 'aligned_assets.csv')
    
    # Backtest Parameters
    INITIAL_CAPITAL = 100_000.0
    TRANSACTION_FEES = 0.0005 # 5bps (0.05%)
    REBALANCE_FREQ = 'QE'     # 'QE' = Quarter End, 'YE' = Year End
    
    # *** Date Range Configuration ***
    # Format: 'YYYY-MM-DD'
    # Set to None to use the full available history
    START_DATE = '2017-01-01'  # Example: Start from Dot-com bubble
    END_DATE = '2026-02-05'    # Example: End after 2022 inflation shock
    # START_DATE = None 
    # END_DATE = None

    # *** Benchmark Configuration ***
    # Assets to compare against in the final plot
    SELECTED_BENCHMARKS = ['Nasdaq100', 'GoldIndex', 'US30Y', 'US3M']
    
    # ---------------------------------
    
    # 2. Check File
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file not found: {DATA_FILE}")
        return

    try:
        # 3. Initialize Strategy (Loads Data)
        strategy = PermanentPortfolioStrategy(DATA_FILE, INITIAL_CAPITAL)
        
        # 4. Run Backtest with Date Range
        strategy.run_backtest(
            start_date=START_DATE, 
            end_date=END_DATE,
            rebalance_freq=REBALANCE_FREQ, 
            fees=TRANSACTION_FEES,
            rebalance_fn=my_new_rebalance,
        )
        
        # 5. Stats Output
        stats = strategy.get_performance_stats()
        
        if not stats:
            logger.warning("No stats generated. Check date range.")
            return

        print("\n" + "="*50)
        print(f"PERMANENT PORTFOLIO ({START_DATE or 'Start'} to {END_DATE or 'End'})")
        print("="*50)
        print(f"Total Return:    {stats['Total Return']*100:.2f}%")
        print(f"CAGR:            {stats['CAGR']*100:.2f}%")
        print(f"Sharpe Ratio:    {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown:    {stats['Max Drawdown']*100:.2f}%")
        print(f"Volatility:      {stats['Volatility']*100:.2f}%")
        print("="*50 + "\n")
        
        # 6. Plotting
        saved_path = strategy.plot_results(
            output_dir=DATA_DIR, 
            benchmark_cols=SELECTED_BENCHMARKS 
        )
        
        if saved_path:
            print(f"Visualization saved to: {saved_path}")
        
    except Exception as e:
        logger.error(f"Critical Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()