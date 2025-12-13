"""Debug script to identify the dataset building error."""
import logging
import traceback
from pathlib import Path
from ufcscraper.dataset_builder import DatasetBuilder

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

data_folder = Path("ufc_data")

try:
    logger.info("Starting dataset builder debug...")
    builder = DatasetBuilder(data_folder)

    logger.info("Building datasets with force_rebuild=True...")
    training_dataset, prediction_dataset = builder.build_datasets(
        min_fights_per_fighter=1,
        test_mode=False,
        force_rebuild=True
    )

    logger.info(f"SUCCESS! Training: {len(training_dataset)} fights, Prediction: {len(prediction_dataset)} fights")

except Exception as e:
    logger.error(f"Error occurred: {e}")
    logger.error("Full traceback:")
    traceback.print_exc()

    # Try to get more details about the error
    import sys
    exc_type, exc_value, exc_tb = sys.exc_info()

    logger.error("\nDetailed error information:")
    logger.error(f"Exception type: {exc_type.__name__}")
    logger.error(f"Exception message: {exc_value}")

    if exc_tb:
        import traceback as tb
        logger.error("\nStack trace:")
        for frame in tb.extract_tb(exc_tb):
            logger.error(f"  File: {frame.filename}:{frame.lineno}")
            logger.error(f"  Function: {frame.name}")
            logger.error(f"  Line: {frame.line}")
