"""
Standalone script for building UFC fight prediction datasets.

This script creates training and prediction datasets with comprehensive
feature engineering from the scraped UFC data.

Usage:
------

To run the script, use the following command:

.. code-block:: bash

    ufcscraper_build_datasets --data-folder /path/to/data --min-fights 2 --test-mode

Arguments:
----------

- **data-folder**: Folder containing the UFC data CSV files (required).
- **min-fights**: Minimum fights per fighter to include in dataset (default: 1).
- **test-mode**: Run in test mode with reduced data for faster processing.
- **log-level**: Set the logging level (e.g., INFO, DEBUG).
- **force-rebuild**: Force rebuild even if datasets are up to date.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ufcscraper.dataset_builder import DatasetBuilder

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main function for building UFC fight prediction datasets.

    This function:
    1. Sets up logging and parses arguments
    2. Initializes the DatasetBuilder
    3. Builds training and prediction datasets
    4. Provides summary statistics
    5. Validates feature consistency

    Args:
        args: Command-line arguments. If None, arguments are parsed using get_args.
    """
    if args is None:
        args = get_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=args.log_level,
        format="%(levelname)s:%(message)s",
    )

    logger.info("Starting UFC fight prediction dataset building...")

    # Initialize dataset builder
    try:
        builder = DatasetBuilder(args.data_folder)
    except Exception as e:
        logger.error(f"Failed to initialize DatasetBuilder: {e}")
        return

    # Check if rebuild is needed
    if not args.force_rebuild and not args.test_mode:
        logger.info("Checking if datasets need updating...")
        if not builder.update_datasets_incrementally():
            logger.info("Datasets are already up to date. Use --force-rebuild to rebuild anyway.")
            return

    # Build datasets
    try:
        logger.info("")
        logger.info("Building datasets...")
        training_dataset, prediction_dataset = builder.build_datasets(
            min_fights_per_fighter=args.min_fights,
            test_mode=args.test_mode
        )

        if training_dataset.empty and prediction_dataset.empty:
            logger.error("No datasets were created. Check your data files.")
            return

    except Exception as e:
        logger.error(f"Failed to build datasets: {e}")
        logger.error("Check that all required data files exist and are properly formatted.")
        return

    # Provide summary
    logger.info("")
    logger.info("Dataset Summary:")
    logger.info("=" * 50)

    if not training_dataset.empty:
        logger.info(f"Training Dataset:")
        logger.info(f"  • {len(training_dataset):,} fights")
        logger.info(f"  • {len(training_dataset.columns):,} total columns")

        # Count non-null targets
        valid_targets = training_dataset['winner'].notna().sum()
        logger.info(f"  • {valid_targets:,} fights with valid outcomes")

        # Check class distribution
        if valid_targets > 0:
            winner_dist = training_dataset['winner'].value_counts()
            logger.info(f"  • Fighter 1 wins: {winner_dist.get(1, 0)} ({winner_dist.get(1, 0)/valid_targets*100:.1f}%)")
            logger.info(f"  • Fighter 2 wins: {winner_dist.get(0, 0)} ({winner_dist.get(0, 0)/valid_targets*100:.1f}%)")

    if not prediction_dataset.empty:
        logger.info(f"Prediction Dataset:")
        logger.info(f"  • {len(prediction_dataset):,} upcoming fights")
        logger.info(f"  • {len(prediction_dataset.columns):,} total columns")

    # Feature breakdown
    try:
        feature_summary = builder.get_feature_summary()
        logger.info("")
        logger.info("Feature Categories:")
        for category, count in feature_summary['feature_categories'].items():
            logger.info(f"  • {category.replace('_', ' ').title()}: {count} features")

        logger.info(f"  • Total predictive features: {feature_summary['total_features']}")

    except Exception as e:
        logger.warning(f"Could not generate feature summary: {e}")

    # Data quality checks
    logger.info("")
    logger.info("Data Quality Checks:")
    logger.info("=" * 50)

    if not training_dataset.empty:
        # Check for missing values in key features
        key_features = [col for col in training_dataset.columns if 'fighter_1_' in col or 'fighter_2_' in col]
        if key_features:
            missing_pct = (training_dataset[key_features].isnull().sum() / len(training_dataset) * 100)
            high_missing = missing_pct[missing_pct > 50]

            if len(high_missing) > 0:
                logger.warning(f"Features with >50% missing values: {len(high_missing)}")
                for feature, pct in high_missing.head().items():
                    logger.warning(f"  • {feature}: {pct:.1f}% missing")
            else:
                logger.info("✓ No features with excessive missing values")

        # Check for feature variance
        numeric_features = training_dataset.select_dtypes(include=['number']).columns
        zero_var_features = []
        for col in numeric_features:
            if col != 'winner' and training_dataset[col].nunique() <= 1:
                zero_var_features.append(col)

        if zero_var_features:
            logger.warning(f"Features with zero variance: {len(zero_var_features)}")
        else:
            logger.info("✓ All features have sufficient variance")

    # Success message
    logger.info("")
    logger.info("✓ Dataset building completed successfully!")
    logger.info(f"✓ Files saved in: {args.data_folder}")
    logger.info("  • training_dataset.csv")
    logger.info("  • prediction_dataset.csv")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Use training_dataset.csv to train your ML model")
    logger.info("  2. Use prediction_dataset.csv to make predictions on upcoming fights")
    logger.info("  3. Run this script again after scraping new data to update datasets")


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments and return them as an argparse.Namespace object.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build UFC fight prediction datasets with comprehensive feature engineering"
    )

    parser.add_argument(
        "--data-folder",
        type=Path,
        required=True,
        help="Folder containing the UFC data CSV files"
    )

    parser.add_argument(
        "--min-fights",
        type=int,
        default=1,
        help="Minimum fights per fighter to include in dataset (default: 1)"
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with reduced data for faster processing"
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if datasets are up to date"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Set the logging level"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()