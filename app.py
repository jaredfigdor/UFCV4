#!/usr/bin/env python3
"""
UFC Data Automation System
==========================

Fully automated system that:
1. Scrapes new upcoming data if needed
2. Moves completed events to historical data (only after event day +1)
3. Builds updated ML datasets with latest data
4. Maintains data consistency throughout

Usage:
    python app.py

This should be run daily or before making predictions to ensure
all data is current and ML datasets are up-to-date.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ufcscraper.data_consistency import DataConsistencyManager
from ufcscraper.dataset_builder import DatasetBuilder
from ufcscraper.ml_predictor import UFCPredictor
from ufcscraper.ufc_scraper import UFCScraper


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ufc_automation.log')
        ]
    )


def check_for_new_data_needed(data_folder: Path) -> tuple[bool, bool]:
    """
    Check if new data scraping is needed.

    Returns:
        Tuple of (needs_upcoming_scrape, needs_completed_scrape)
    """
    logger = logging.getLogger(__name__)

    # Check if data files exist
    upcoming_events_file = data_folder / "upcoming_event_data.csv"
    upcoming_fights_file = data_folder / "upcoming_fight_data.csv"
    events_file = data_folder / "event_data.csv"
    fights_file = data_folder / "fight_data.csv"

    needs_upcoming = False
    needs_completed = False

    # Check upcoming data freshness
    if not upcoming_events_file.exists() or not upcoming_fights_file.exists():
        logger.info("Missing upcoming data files - scraping needed")
        needs_upcoming = True
    else:
        # Check if upcoming data is older than 12 hours
        file_age = datetime.now() - datetime.fromtimestamp(upcoming_events_file.stat().st_mtime)
        if file_age > timedelta(hours=12):
            logger.info(f"Upcoming data is {file_age} old - refresh needed")
            needs_upcoming = True

    # Check completed data freshness
    if not events_file.exists() or not fights_file.exists():
        logger.info("Missing completed data files - scraping needed")
        needs_completed = True
    else:
        # Check if there are recent events that might need updating
        try:
            events_df = pd.read_csv(events_file)
            if len(events_df) > 0:
                latest_event = pd.to_datetime(events_df['event_date']).max()
                days_since_latest = (datetime.now() - latest_event).days
                if days_since_latest < 7:  # Recent events might need updates
                    logger.info("Recent events detected - checking for updates")
                    needs_completed = True
        except Exception as e:
            logger.warning(f"Could not check completed data freshness: {e}")

    return needs_upcoming, needs_completed


def scrape_new_data(data_folder: Path, scrape_upcoming: bool, scrape_completed: bool) -> None:
    """Scrape new data if needed."""
    logger = logging.getLogger(__name__)

    if not scrape_upcoming and not scrape_completed:
        logger.info("No scraping needed - data is current")
        return

    logger.info("Initializing UFC scraper...")

    # Use single session for Windows compatibility
    scraper = UFCScraper(
        data_folder=data_folder,
        n_sessions=1,
        delay=0.1
    )

    if scrape_upcoming:
        logger.info("Scraping upcoming events and fights...")
        try:
            # Scrape upcoming events
            logger.info("Scraping upcoming events...")
            scraper.upcoming_event_scraper.scrape_events()

            # Scrape upcoming fights
            logger.info("Scraping upcoming fights...")
            scraper.upcoming_fight_scraper.scrape_fights()

            # CRITICAL: Update fighter data after scraping upcoming fights
            # Upcoming fights may have new fighters not yet in fighter_data.csv
            logger.info("Updating fighter data for upcoming fights...")
            scraper.fighter_scraper.scrape_fighters()

            logger.info(" Upcoming data scraping completed")
        except Exception as e:
            logger.error(f"Error scraping upcoming data: {e}")
            raise

    if scrape_completed:
        logger.info("Scraping completed events and fights...")
        try:
            # Only scrape recent fighters to avoid long processing
            logger.info("Updating fighter data...")
            scraper.fighter_scraper.scrape_fighters()

            # Scrape recent completed events
            logger.info("Scraping completed events...")
            scraper.event_scraper.scrape_events()

            # Scrape recent completed fights
            logger.info("Scraping completed fights...")
            scraper.fight_scraper.scrape_fights()

            logger.info(" Completed data scraping finished")
        except Exception as e:
            logger.error(f"Error scraping completed data: {e}")
            raise


def move_completed_events(data_folder: Path) -> tuple[int, list[str]]:
    """
    Move events from upcoming to completed if they finished more than 1 day ago.

    Returns:
        Tuple of (number of events moved, list of event IDs that were moved)
    """
    logger = logging.getLogger(__name__)

    upcoming_events_file = data_folder / "upcoming_event_data.csv"
    upcoming_fights_file = data_folder / "upcoming_fight_data.csv"

    if not upcoming_events_file.exists():
        logger.info("No upcoming events file found")
        return 0, []

    try:
        upcoming_events = pd.read_csv(upcoming_events_file)
        upcoming_fights = pd.read_csv(upcoming_fights_file) if upcoming_fights_file.exists() else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading upcoming data: {e}")
        return 0, []

    if len(upcoming_events) == 0:
        logger.info("No upcoming events to check")
        return 0, []

    # Find events that are now completed (event date + 1 day < today)
    today = datetime.now().date()
    upcoming_events['event_date'] = pd.to_datetime(upcoming_events['event_date']).dt.date

    completed_events = upcoming_events[
        upcoming_events['event_date'] < (today - timedelta(days=1))
    ]

    if len(completed_events) == 0:
        logger.info("No events ready to move to completed")
        return 0, []

    logger.info(f"Found {len(completed_events)} events to move to completed:")
    moved_event_ids = []
    for _, event in completed_events.iterrows():
        logger.info(f"  - {event['event_name']} ({event['event_date']})")
        moved_event_ids.append(event['event_id'])

    # Initialize data consistency manager to handle the move
    consistency_manager = DataConsistencyManager(data_folder)

    # Run the full consistency management to properly move events
    try:
        logger.info("Running data consistency management...")
        consistency_manager.manage_event_consistency()
        logger.info(f" Successfully moved {len(moved_event_ids)} events to completed data")
        logger.info(f" Events will now be re-scraped to get complete fight results")
    except Exception as e:
        logger.error(f"Error in consistency management: {e}")
        moved_event_ids = []

    return len(moved_event_ids), moved_event_ids


def scrape_completed_event_fights(data_folder: Path, event_ids: list[str]) -> None:
    """
    Scrape fight data for specific completed events.

    This is called after events are moved from upcoming to completed to get
    the full fight results (17 columns) from UFCStats.com.

    Args:
        data_folder: Path to data folder
        event_ids: List of event IDs to scrape fights for
    """
    logger = logging.getLogger(__name__)

    if not event_ids:
        return

    logger.info(f" Scraping complete fight data for {len(event_ids)} newly completed events...")

    try:
        # Simply re-scrape all completed events to get the new fight data
        # This is efficient because the scraper only scrapes new fights (not already in fight_data.csv)
        scraper = UFCScraper(
            data_folder=data_folder,
            n_sessions=1,  # Single session for Windows compatibility
            delay=0.5
        )

        logger.info(" Scraping fights from completed events...")
        scraper.fight_scraper.scrape_fights(get_all_events=False)

        logger.info(" Updating fighter data...")
        scraper.fighter_scraper.scrape_fighters()

        logger.info(f" Successfully scraped fight data for newly completed events")

    except Exception as e:
        logger.error(f"Error scraping completed event fights: {e}")
        raise


def build_updated_datasets(data_folder: Path, force_rebuild: bool = False) -> tuple[int, int]:
    """
    Build updated ML datasets with latest data using intelligent caching.

    Args:
        data_folder: Path to data folder
        force_rebuild: Force rebuild even if cache is valid

    Returns:
        Tuple of (training_fights, prediction_fights)
    """
    logger = logging.getLogger(__name__)

    logger.info(" Building/loading ML datasets...")

    try:
        dataset_builder = DatasetBuilder(data_folder)

        # Build datasets with intelligent caching
        training_dataset, prediction_dataset = dataset_builder.build_datasets(
            min_fights_per_fighter=1,
            test_mode=False,
            force_rebuild=force_rebuild
        )

        training_count = len(training_dataset)
        prediction_count = len(prediction_dataset)
        feature_count = len(training_dataset.columns) - 1  # Exclude target

        logger.info(f" ML datasets ready:")
        logger.info(f"   Training dataset: {training_count:,} fights")
        logger.info(f"   Prediction dataset: {prediction_count} fights")
        logger.info(f"   Features: {feature_count} advanced features")

        return training_count, prediction_count

    except Exception as e:
        logger.error(f"Error with datasets: {e}")
        raise


def validate_data_integrity(data_folder: Path) -> bool:
    """Validate data integrity across all files."""
    logger = logging.getLogger(__name__)

    logger.info(" Validating data integrity...")

    try:
        consistency_manager = DataConsistencyManager(data_folder)
        is_valid = consistency_manager.validate_data_integrity()

        if is_valid:
            logger.info(" Data integrity validation passed")
        else:
            logger.warning(" Data integrity issues detected")

        return is_valid

    except Exception as e:
        logger.error(f"Error validating data integrity: {e}")
        return False


def train_and_predict(data_folder: Path, retrain_model: bool = False) -> tuple[dict, pd.DataFrame]:
    """
    Train ML model and make fight predictions.

    Args:
        data_folder: Path to data folder
        retrain_model: Whether to retrain the model or use existing one

    Returns:
        Tuple of (model_metrics, predictions_dataframe)
    """
    logger = logging.getLogger(__name__)

    logger.info(" Initializing ML prediction system...")

    try:
        predictor = UFCPredictor(data_folder)

        # Check if we should retrain or can use existing model
        should_train = retrain_model or not predictor.is_model_trained()

        if should_train:
            logger.info(" Training new ML model...")

            # Load and preprocess data
            training_df, prediction_df = predictor.load_and_preprocess_data()

            # Train model with hyperparameter optimization
            model_metrics = predictor.train_model(
                training_df,
                model_type="xgboost",  # Best for sports prediction with anti-overfitting
                optimize_hyperparameters=True,
                test_size=0.2
            )

            logger.info(" Model training completed")
        else:
            logger.info(" Loading existing trained model...")
            if not predictor.load_model():
                logger.info(" Could not load model, training new one...")
                training_df, prediction_df = predictor.load_and_preprocess_data()
                model_metrics = predictor.train_model(
                    training_df,
                    model_type="xgboost",  # Best for sports prediction with anti-overfitting
                    optimize_hyperparameters=True
                )
            else:
                logger.info(" Existing model loaded successfully")
                # Load data for predictions
                _, prediction_df = predictor.load_and_preprocess_data()
                model_metrics = predictor.get_model_info()

        # Make predictions for upcoming fights
        logger.info(" Making fight predictions...")
        predictions = predictor.predict_fights(prediction_df)

        # Add readable fight information
        predictions = _enhance_predictions_with_context(predictions, data_folder)

        # Save predictions with upsert logic (update existing, insert new)
        predictions_file = data_folder / "fight_predictions.csv"
        predictions = _upsert_predictions(predictions, predictions_file, logger)

        logger.info(f" Predictions saved to {predictions_file}")
        logger.info(f" Generated predictions for {len(predictions)} upcoming fights")

        # Also save a readable summary with upsert logic
        readable_file = data_folder / "fight_predictions_summary.csv"
        if 'fighter_1_name' in predictions.columns and 'fighter_2_name' in predictions.columns:
            summary_cols = [
                'fight_id', 'event_name', 'event_date', 'fighter_1_name', 'fighter_2_name',
                'predicted_winner_name', 'confidence', 'fighter_1_win_probability', 'fighter_2_win_probability',
                'weight_class_name'
            ]

            # Only include columns that exist
            available_cols = [col for col in summary_cols if col in predictions.columns]
            readable_predictions = predictions[available_cols].copy()

            # Upsert into summary file
            readable_predictions = _upsert_predictions(readable_predictions, readable_file, logger)

            # Sort by confidence (highest first) for display
            if 'confidence' in readable_predictions.columns:
                readable_predictions = readable_predictions.sort_values('confidence', ascending=False)
                readable_predictions.to_csv(readable_file, index=False)

            logger.info(f" Readable predictions summary saved to {readable_file}")

        # Show top feature importance
        top_features = predictor.get_top_features(10)
        if top_features:
            logger.info(" Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(top_features, 1):
                logger.info(f"  {i:2d}. {feature}: {importance:.3f}")

        return model_metrics, predictions

    except Exception as e:
        logger.error(f"Error in ML prediction pipeline: {e}")
        raise


def _upsert_predictions(new_predictions: pd.DataFrame, file_path: Path, logger) -> pd.DataFrame:
    """
    Upsert predictions into CSV file - update existing records, insert new ones.

    Args:
        new_predictions: New predictions DataFrame with fight_id column
        file_path: Path to the predictions CSV file
        logger: Logger instance for logging

    Returns:
        Combined DataFrame with all predictions (historical + new/updated)
    """
    if 'fight_id' not in new_predictions.columns:
        logger.warning("No fight_id column found in predictions - saving without upsert")
        new_predictions.to_csv(file_path, index=False)
        return new_predictions

    # Load existing predictions if file exists
    if file_path.exists():
        try:
            existing_predictions = pd.read_csv(file_path)
            logger.info(f"Loaded {len(existing_predictions)} existing predictions from {file_path.name}")

            # Check if existing file has fight_id column
            if 'fight_id' not in existing_predictions.columns:
                logger.warning(f"Existing {file_path.name} has no fight_id column - replacing file")
                new_predictions.to_csv(file_path, index=False)
                return new_predictions

            # Identify which fights are updates vs new inserts
            new_fight_ids = set(new_predictions['fight_id'])
            existing_fight_ids = set(existing_predictions['fight_id'])

            updated_fights = new_fight_ids & existing_fight_ids
            inserted_fights = new_fight_ids - existing_fight_ids

            # Remove old versions of updated fights from existing data
            if updated_fights:
                existing_predictions = existing_predictions[
                    ~existing_predictions['fight_id'].isin(updated_fights)
                ]
                logger.info(f"  Updating {len(updated_fights)} existing fight predictions")

            if inserted_fights:
                logger.info(f"  Inserting {len(inserted_fights)} new fight predictions")

            # Combine: keep all old predictions (except updated ones) + all new predictions
            combined_predictions = pd.concat(
                [existing_predictions, new_predictions],
                ignore_index=True
            )

            # Ensure consistent column order (new predictions columns first)
            # This handles cases where new predictions have additional columns
            all_columns = list(new_predictions.columns)
            for col in existing_predictions.columns:
                if col not in all_columns:
                    all_columns.append(col)

            # Reorder columns and fill missing values
            combined_predictions = combined_predictions.reindex(columns=all_columns)

            logger.info(f"  Total predictions after upsert: {len(combined_predictions)}")

        except Exception as e:
            logger.warning(f"Error loading existing predictions: {e} - creating new file")
            combined_predictions = new_predictions
    else:
        logger.info(f"Creating new predictions file: {file_path.name}")
        combined_predictions = new_predictions

    # Save combined predictions
    combined_predictions.to_csv(file_path, index=False)

    return combined_predictions


def _enhance_predictions_with_context(predictions: pd.DataFrame, data_folder: Path) -> pd.DataFrame:
    """Enhance predictions with readable fight context."""
    logger = logging.getLogger(__name__)
    try:
        enhanced = predictions.copy()

        # Load fighter data for names
        fighters_file = data_folder / "fighter_data.csv"
        if fighters_file.exists():
            fighters_df = pd.read_csv(fighters_file)

            # Create full fighter names
            fighters_df['full_name'] = (
                fighters_df['fighter_f_name'].fillna('') + ' ' +
                fighters_df['fighter_l_name'].fillna('')
            ).str.strip()

            # Add nickname if available
            fighters_df['display_name'] = fighters_df.apply(
                lambda row: f"{row['full_name']} '{row['fighter_nickname']}'"
                if pd.notna(row['fighter_nickname']) and row['fighter_nickname'].strip()
                else row['full_name'], axis=1
            )

            fighter_names = dict(zip(fighters_df['fighter_id'], fighters_df['display_name']))

            # Map fighter names
            if 'fighter_1' in enhanced.columns:
                enhanced['fighter_1_name'] = enhanced['fighter_1'].map(fighter_names)
                enhanced['fighter_2_name'] = enhanced['fighter_2'].map(fighter_names)

                # Fill missing names with fighter IDs as fallback
                enhanced['fighter_1_name'] = enhanced['fighter_1_name'].fillna('Unknown Fighter')
                enhanced['fighter_2_name'] = enhanced['fighter_2_name'].fillna('Unknown Fighter')
            else:
                enhanced['fighter_1_name'] = 'Unknown Fighter'
                enhanced['fighter_2_name'] = 'Unknown Fighter'
        else:
            enhanced['fighter_1_name'] = 'Unknown Fighter'
            enhanced['fighter_2_name'] = 'Unknown Fighter'

        # Load event data for event names
        if 'event_id' in enhanced.columns:
            upcoming_events_file = data_folder / "upcoming_event_data.csv"
            if upcoming_events_file.exists():
                events_df = pd.read_csv(upcoming_events_file)
                event_data = dict(zip(events_df['event_id'],
                                    zip(events_df['event_name'], events_df['event_date'])))

                enhanced['event_name'] = enhanced['event_id'].map(lambda x: event_data.get(x, ('Unknown Event', ''))[0])
                enhanced['event_date'] = enhanced['event_id'].map(lambda x: event_data.get(x, ('', 'Unknown Date'))[1])

        # Clean up weight class names (must match encoding in feature_engineering.py)
        # NOTE: We need to look up the original weight class from the source data
        # because the numeric encoding loses information about Women's vs Men's divisions
        if 'fight_id' in enhanced.columns:
            # Load upcoming fight data to get original weight classes
            upcoming_fights_file = data_folder / "upcoming_fight_data.csv"
            if upcoming_fights_file.exists():
                upcoming_fights_df = pd.read_csv(upcoming_fights_file)
                weight_class_lookup = dict(zip(upcoming_fights_df['fight_id'], upcoming_fights_df['weight_class']))
                enhanced['weight_class_name'] = enhanced['fight_id'].map(weight_class_lookup).fillna('Unknown')
            else:
                enhanced['weight_class_name'] = 'Unknown'
        else:
            enhanced['weight_class_name'] = 'Unknown'

        # Create readable winner names
        enhanced['predicted_winner_name'] = enhanced.apply(
            lambda row: row['fighter_1_name'] if row['fighter_1_win_probability'] > row['fighter_2_win_probability']
            else row['fighter_2_name'], axis=1
        )

        logger.info(f"Enhanced {len(enhanced)} predictions with fighter names and context")
        return enhanced

    except Exception as e:
        logger.warning(f"Could not enhance predictions with context: {e}")
        logger.warning(f"Available columns: {list(predictions.columns)}")
        return predictions


def main() -> None:
    """Main automation workflow."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UFC Data Automation & Prediction System")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild of datasets even if cache is valid")
    parser.add_argument("--retrain-model", action="store_true",
                        help="Force retrain of ML model")
    parser.add_argument("--no-web", action="store_true",
                        help="Skip launching web interface (for automation/headless mode)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for web interface (default: 5000)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(" Starting UFC Data Automation System")
    logger.info("=" * 50)

    # Configuration
    data_folder = Path("ufc_data")
    data_folder.mkdir(exist_ok=True)

    try:
        # Step 1: Check if new data scraping is needed
        logger.info(" Step 1: Checking data freshness...")
        needs_upcoming, needs_completed = check_for_new_data_needed(data_folder)

        # Step 2: Scrape new data if needed
        if needs_upcoming or needs_completed:
            logger.info("Step 2: Scraping new data...")
            scrape_new_data(data_folder, needs_upcoming, needs_completed)
        else:
            logger.info(" Step 2: Skipping scraping - data is current")

        # Step 3: Move completed events
        logger.info(" Step 3: Moving completed events...")
        moved_events_count, moved_event_ids = move_completed_events(data_folder)

        # Step 3.5: Scrape fight details for newly completed events
        if moved_events_count > 0:
            logger.info(" Step 3.5: Scraping complete fight data for newly completed events...")
            scrape_completed_event_fights(data_folder, moved_event_ids)

        # Step 4: Build updated ML datasets
        logger.info(" Step 4: Building ML datasets...")
        # Force rebuild if we moved events, scraped new data, or user requested it
        force_rebuild = (needs_upcoming or needs_completed or moved_events_count > 0 or args.force_rebuild)
        training_count, prediction_count = build_updated_datasets(data_folder, force_rebuild=force_rebuild)

        # Step 5: Validate data integrity
        logger.info(" Step 5: Validating data integrity...")
        is_valid = validate_data_integrity(data_folder)

        # Step 6: Train ML model and make predictions
        logger.info(" Step 6: Training ML model and making predictions...")
        model_metrics, predictions = train_and_predict(data_folder, retrain_model=args.retrain_model)

        # Summary
        logger.info("=" * 50)
        logger.info(" UFC AUTOMATION & PREDICTION SYSTEM COMPLETED!")
        logger.info(f" Training fights: {training_count:,}")
        logger.info(f" Upcoming fights: {prediction_count}")
        logger.info(f" Events moved: {moved_events_count}")
        logger.info(f" Data integrity: {'PASSED' if is_valid else 'ISSUES DETECTED'}")

        if isinstance(model_metrics, dict):
            logger.info(f" Model accuracy: {model_metrics.get('test_accuracy', 0):.1%}")
            logger.info(f" Model AUC: {model_metrics.get('test_auc', 0):.3f}")

        logger.info(f" Fight predictions: {len(predictions)} fights predicted")
        logger.info("=" * 50)
        logger.info(" FIGHT PREDICTIONS READY!")

        # Show some sample predictions
        if len(predictions) > 0:
            logger.info("\n UPCOMING FIGHT PREDICTIONS:")
            display_predictions = predictions.head(5)
            if 'confidence' in predictions.columns:
                display_predictions = predictions.nlargest(5, 'confidence')

            for _, pred in display_predictions.iterrows():
                fighter1 = pred.get('fighter_1_name', 'Unknown Fighter')
                fighter2 = pred.get('fighter_2_name', 'Unknown Fighter')
                predicted_winner = pred.get('predicted_winner_name', 'Unknown')
                confidence = pred.get('confidence', 0)
                prob1 = pred.get('fighter_1_win_probability', 0)
                prob2 = pred.get('fighter_2_win_probability', 0)
                weight_class = pred.get('weight_class_name', 'Unknown')
                event_name = pred.get('event_name', 'Unknown Event')
                event_date = pred.get('event_date', '')

                logger.info(f"  {fighter1} vs {fighter2}")
                logger.info(f"    Weight Class: {weight_class}")
                if event_name != 'Unknown Event':
                    event_info = f"Event: {event_name}"
                    if event_date:
                        event_info += f" ({event_date})"
                    logger.info(f"    {event_info}")
                logger.info(f"    Predicted Winner: {predicted_winner} ({confidence:.1%} confidence)")
                logger.info(f"    Win Probabilities: {fighter1} {prob1:.1%} vs {fighter2} {prob2:.1%}")
                logger.info("")

        if not is_valid:
            logger.warning(" Please check data integrity issues")

        # Step 7: Launch web interface
        if not args.no_web:
            logger.info("")
            logger.info(" Step 7: Launching web interface...")
            logger.info("=" * 50)

            try:
                from ufcscraper.web_app import launch_web_app
                launch_web_app(
                    data_folder=data_folder,
                    port=args.port,
                    auto_open=True,
                    debug=False
                )
            except KeyboardInterrupt:
                logger.info("\n Web interface closed")
            except Exception as e:
                logger.error(f"Error launching web interface: {e}")
                logger.warning("Continuing without web interface...")
        else:
            logger.info(" Step 7: Skipping web interface (--no-web flag)")

        if not is_valid:
            sys.exit(1)

    except Exception as e:
        logger.error(f" Automation failed: {e}")
        logger.error("Please check the logs and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()