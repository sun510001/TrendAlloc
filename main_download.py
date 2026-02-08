from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.data_processor import DataProcessor
from logger import logger

def main():
    # --- Configuration ---
    # Mapping: { "Local_Filename_Alias" : "Yahoo_Ticker" }
    TICKER_MAP = {
        "QQQ_Proxy": "^NDX",     # Nasdaq 100 Index (since 1985)
        "GOLD_Proxy": "GC=F",    # Gold Futures (Continuous)
        "US30Y_Yield": "^TYX",   # 30-Year Treasury Yield
        "CASH_Yield": "^IRX"     # 13-Week Treasury Bill Yield
    }
    START_YEAR = 1985

    # --- Step 1: Incremental ETL (Extract) ---
    logger.info(">>> STEP 1: DOWNLOADING RAW DATA (INCREMENTAL) <<<")
    downloader = YahooIncrementalLoader(storage_path="./data")
    downloader.download_batch(TICKER_MAP, start_year=START_YEAR)

    # --- Step 2: Transformation & Pricing (Transform) ---
    logger.info(">>> STEP 2: PROCESSING & SYNTHETIC PRICING <<<")
    processor = DataProcessor(raw_path="./data", processed_path="./data_processed")
    processor.process_and_align()

    logger.info(">>> DATA PIPELINE FINISHED <<<")

if __name__ == "__main__":
    main()