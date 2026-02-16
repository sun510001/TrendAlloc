from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.data_processor import DataProcessor
from logger import logger


# Asset configuration: can be later provided by frontend
ASSETS = [
    {
        "name": "Nasdaq100",
        "ticker": "^NDX",   # Nasdaq 100 Index
        "kind": "price",
    },
    {
        "name": "GoldIndex",
        "ticker": "^XAU",   # Philadelphia Gold/Silver Index
        "kind": "price",
    },
    {
        "name": "US30Y",
        "ticker": "^TYX",   # 30-Year Treasury Yield (yield series)
        "kind": "yield",
        "engine": "bond",
        "duration": 20.0,
    },
    {
        "name": "US3M",
        "ticker": "^IRX",   # 13-Week Treasury Bill Yield (yield series)
        "kind": "yield",
        "engine": "cash",
    },
]


def main():
    # --- Configuration ---
    START_YEAR = 1985

    # --- Step 1: Incremental ETL (Extract) ---
    logger.info(">>> STEP 1: DOWNLOADING RAW DATA (INCREMENTAL) <<<")
    downloader = YahooIncrementalLoader(storage_path="./data")
    downloader.download_batch(ASSETS, start_year=START_YEAR)

    # --- Step 2: Transformation & Pricing (Transform) ---
    logger.info(">>> STEP 2: PROCESSING & SYNTHETIC PRICING <<<")
    processor = DataProcessor(raw_path="./data", processed_path="./data_processed")
    processor.process_and_align(ASSETS)

    logger.info(">>> DATA PIPELINE FINISHED <<<")


if __name__ == "__main__":
    main()
