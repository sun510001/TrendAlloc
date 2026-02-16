from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.data_processor import DataProcessor
from backend.assets_config import AssetConfigManager
from logger import logger

class DataPipelineRunner:
    """
    Runner class to orchestrate the end-to-end data pipeline.
    
    This includes loading asset configurations, downloading raw data from 
    external sources, and processing/aligning the data for backtesting.
    """

    def __init__(self, start_year: int = 1985) -> None:
        """
        Initialize the pipeline runner.

        Args:
            start_year (int): The default start year for data download if no 
                              specific date is configured. Defaults to 1985.
        """
        self.start_year: int = start_year

    def run(self) -> None:
        """
        Execute the data pipeline: Load -> Download -> Process.

        This method coordinates the sequence of operations required to prepare
        market data for the backtest engine.
        """
        # Load asset configurations from persistent storage
        assets = AssetConfigManager.load_assets()
        if not assets:
            logger.error("No assets configured. Please add assets via the UI or config/assets.json first.")
            return

        # Step 1: Incremental ETL (Extract)
        logger.info(">>> STEP 1: DOWNLOADING RAW DATA (INCREMENTAL) <<<")
        downloader = YahooIncrementalLoader(storage_path="./data")
        downloader.download_batch(assets, start_year=self.start_year)

        # Step 2: Transformation & Alignment (Transform)
        logger.info(">>> STEP 2: PROCESSING & SYNTHETIC PRICING <<<")
        processor = DataProcessor(raw_path="./data", processed_path="./data_processed")
        processor.process_and_align(assets)

        logger.info(">>> DATA PIPELINE FINISHED <<<")

if __name__ == "__main__":
    # Create and execute the runner
    runner = DataPipelineRunner(start_year=1985)
    runner.run()
