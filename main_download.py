from data_loader.tools_gold_probe import GoldDataProbe
from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.data_processor import DataProcessor
from logger import logger

def main():
    # --- Configuration ---
    TICKER_MAP = {
        "Stocks": "^NDX",   # Nasdaq 100 Index (since 1985)
        "Gold":   "^XAU",    # Philadelphia Gold/Silver Index (代替 GLD)
        "Bonds":  "^TYX",   # 30-Year Treasury Yield
        "Cash":   "^IRX"    # 13-Week Treasury Bill Yield
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