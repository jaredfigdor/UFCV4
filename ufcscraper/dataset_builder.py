"""
Dataset builder module for UFC fight prediction.

This module handles the creation of training and prediction datasets by combining
fight data, fighter data, and round data with comprehensive feature engineering.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import numpy as np

from ufcscraper.feature_engineering import FeatureEngineering

if TYPE_CHECKING:
    from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Dataset builder for UFC fight prediction.

    This class creates training and prediction datasets with comprehensive
    feature engineering while ensuring temporal consistency and leak-free features.
    """

    def __init__(self, data_folder: Path | str):
        """
        Initialize the dataset builder.

        Args:
            data_folder: Path to the folder containing UFC data CSV files.
        """
        self.data_folder = Path(data_folder)
        self.feature_engineer = FeatureEngineering()

        # Define file paths
        self.events_file = self.data_folder / "event_data.csv"
        self.upcoming_events_file = self.data_folder / "upcoming_event_data.csv"
        self.fights_file = self.data_folder / "fight_data.csv"
        self.upcoming_fights_file = self.data_folder / "upcoming_fight_data.csv"
        self.fighters_file = self.data_folder / "fighter_data.csv"
        self.rounds_file = self.data_folder / "round_data.csv"

        # Output paths
        self.training_dataset_file = self.data_folder / "training_dataset.csv"
        self.prediction_dataset_file = self.data_folder / "prediction_dataset.csv"
        self.dataset_metadata_file = self.data_folder / "dataset_metadata.json"

    def build_datasets(
        self,
        min_fights_per_fighter: int = 1,
        test_mode: bool = False,
        force_rebuild: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build both training and prediction datasets with intelligent caching.

        Args:
            min_fights_per_fighter: Minimum fights required for a fighter to be included
            test_mode: If True, process smaller subset for testing
            force_rebuild: If True, force rebuild even if cache is valid

        Returns:
            Tuple of (training_dataset, prediction_dataset)
        """
        logger.info(" Building UFC fight prediction datasets...")

        # Load raw data
        logger.info("Loading raw data files...")
        raw_data = self._load_raw_data()

        if test_mode:
            logger.info("Test mode: Limiting dataset size for faster processing")
            raw_data = self._limit_data_for_testing(raw_data)

        # Filter fighters with minimum fight requirements
        raw_data = self._filter_fighters_by_experience(raw_data, min_fights_per_fighter)

        # Check for separate training and prediction cache usage
        training_dataset = None
        prediction_dataset = None

        if not force_rebuild and not test_mode:
            # Try to load training dataset from cache
            training_dataset = self._load_training_cache(raw_data, min_fights_per_fighter)

            # Try to load prediction dataset from cache
            prediction_dataset = self._load_prediction_cache(raw_data)

        # Build training dataset if not cached
        if training_dataset is None:
            logger.info("Building training dataset...")
            training_dataset = self._build_training_dataset(raw_data)
            self._save_training_cache(training_dataset, raw_data, min_fights_per_fighter)

        # Build prediction dataset if not cached
        if prediction_dataset is None:
            logger.info("Building prediction dataset...")
            prediction_dataset = self._build_prediction_dataset(raw_data)
            self._save_prediction_cache(prediction_dataset, raw_data)

        # Validate feature consistency
        logger.info("Validating feature consistency...")
        if not self.feature_engineer.validate_features(training_dataset, prediction_dataset):
            raise ValueError("Feature consistency validation failed")

        logger.info(f" Datasets built successfully:")
        logger.info(f"  Training: {len(training_dataset)} fights, {len(training_dataset.columns)} features")
        logger.info(f"  Prediction: {len(prediction_dataset)} fights, {len(prediction_dataset.columns)} features")

        return training_dataset, prediction_dataset

    def _load_raw_data(self) -> dict:
        """Load all raw data files."""
        raw_data = {}

        # Load completed events and fights
        try:
            raw_data['events'] = pd.read_csv(self.events_file, encoding='utf-8')
            logger.info(f"Loaded {len(raw_data['events'])} completed events")
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            raw_data['events'] = pd.DataFrame()

        try:
            raw_data['fights'] = pd.read_csv(self.fights_file, encoding='utf-8')
            logger.info(f"Loaded {len(raw_data['fights'])} completed fights")
        except Exception as e:
            logger.error(f"Error loading fights: {e}")
            raw_data['fights'] = pd.DataFrame()

        # Load upcoming events and fights
        try:
            raw_data['upcoming_events'] = pd.read_csv(self.upcoming_events_file, encoding='utf-8')
            logger.info(f"Loaded {len(raw_data['upcoming_events'])} upcoming events")
        except Exception as e:
            logger.warning(f"Error loading upcoming events: {e}")
            raw_data['upcoming_events'] = pd.DataFrame()

        try:
            raw_data['upcoming_fights'] = pd.read_csv(self.upcoming_fights_file, encoding='utf-8')
            logger.info(f"Loaded {len(raw_data['upcoming_fights'])} upcoming fights")
        except Exception as e:
            logger.warning(f"Error loading upcoming fights: {e}")
            raw_data['upcoming_fights'] = pd.DataFrame()

        # Load fighters and rounds (shared)
        try:
            raw_data['fighters'] = pd.read_csv(self.fighters_file, encoding='utf-8')
            logger.info(f"Loaded {len(raw_data['fighters'])} fighters")
        except Exception as e:
            logger.error(f"Error loading fighters: {e}")
            raw_data['fighters'] = pd.DataFrame()

        try:
            raw_data['rounds'] = pd.read_csv(self.rounds_file, encoding='utf-8')
            logger.info(f"Loaded {len(raw_data['rounds'])} round records")
        except Exception as e:
            logger.warning(f"Error loading rounds: {e}")
            raw_data['rounds'] = pd.DataFrame()

        return raw_data

    def _limit_data_for_testing(self, raw_data: dict) -> dict:
        """Limit data size for testing purposes."""
        # Take last 1000 fights for testing
        if len(raw_data['events']) > 100:
            recent_events = raw_data['events'].tail(100)
            recent_event_ids = set(recent_events['event_id'])

            raw_data['events'] = recent_events
            raw_data['fights'] = raw_data['fights'][
                raw_data['fights']['event_id'].isin(recent_event_ids)
            ]

            # Limit rounds to these fights
            recent_fight_ids = set(raw_data['fights']['fight_id'])
            raw_data['rounds'] = raw_data['rounds'][
                raw_data['rounds']['fight_id'].isin(recent_fight_ids)
            ]

            logger.info(f"Limited to {len(raw_data['fights'])} recent fights for testing")

        return raw_data

    def _filter_fighters_by_experience(self, raw_data: dict, min_fights: int) -> dict:
        """Filter out fighters with insufficient fight history."""
        if min_fights <= 1:
            return raw_data

        # Count fights per fighter
        all_fighters_in_fights = pd.concat([
            raw_data['fights'][['fighter_1']].rename(columns={'fighter_1': 'fighter_id'}),
            raw_data['fights'][['fighter_2']].rename(columns={'fighter_2': 'fighter_id'})
        ])

        fighter_fight_counts = all_fighters_in_fights['fighter_id'].value_counts()
        experienced_fighters = set(fighter_fight_counts[fighter_fight_counts >= min_fights].index)

        # Filter fights to only include experienced fighters
        original_fight_count = len(raw_data['fights'])
        raw_data['fights'] = raw_data['fights'][
            raw_data['fights']['fighter_1'].isin(experienced_fighters) &
            raw_data['fights']['fighter_2'].isin(experienced_fighters)
        ]

        logger.info(f"Filtered fights from {original_fight_count} to {len(raw_data['fights'])} "
                   f"(min {min_fights} fights per fighter)")

        # Filter rounds accordingly
        fight_ids = set(raw_data['fights']['fight_id'])
        raw_data['rounds'] = raw_data['rounds'][
            raw_data['rounds']['fight_id'].isin(fight_ids)
        ]

        return raw_data

    def _build_training_dataset(self, raw_data: dict) -> pd.DataFrame:
        """Build the training dataset from completed fights."""
        if raw_data['fights'].empty:
            logger.warning("No completed fights available for training dataset")
            return pd.DataFrame()

        # Filter fights with valid winners
        training_fights = raw_data['fights'][
            raw_data['fights']['winner'].notna()
        ].copy()

        if len(training_fights) == 0:
            logger.warning("No fights with valid winners found")
            return pd.DataFrame()

        logger.info(f"Creating features for {len(training_fights)} training fights")

        # Merge training fights with event dates for temporal filtering
        all_fights_with_dates = training_fights.merge(
            raw_data['events'][['event_id', 'event_date']],
            on='event_id',
            how='left'
        )

        # Create features with temporal awareness (CRITICAL: pass all_completed_fights)
        training_dataset = self.feature_engineer.create_fight_features(
            fights_df=training_fights,
            fighters_df=raw_data['fighters'],
            rounds_df=raw_data['rounds'],
            events_df=raw_data['events'],
            is_prediction=False,
            all_completed_fights=all_fights_with_dates  # FIXED: Pass all fights for temporal filtering
        )

        # Remove rows with missing target
        training_dataset = training_dataset[training_dataset['winner'].notna()]

        # Add metadata
        training_dataset['dataset_type'] = 'training'
        training_dataset['created_at'] = pd.Timestamp.now()

        return training_dataset

    def _build_prediction_dataset(self, raw_data: dict) -> pd.DataFrame:
        """Build the prediction dataset from upcoming fights."""
        if raw_data['upcoming_fights'].empty:
            logger.warning("No upcoming fights available for prediction dataset")
            return pd.DataFrame()

        # Add missing columns to upcoming fights (they don't have winner, result, etc.)
        upcoming_fights = raw_data['upcoming_fights'].copy()

        # Add columns that exist in completed fights but not upcoming
        missing_columns = {
            'winner': None,
            'num_rounds': 3,  # Default to 3 rounds
            'gender': 'M',    # Default (could be improved with better logic)
            'result': None,
            'result_details': None,
            'finish_round': None,
            'finish_time': None,
            'time_format': '3 Rnd (5-5-5)',
            'scores_1': None,
            'scores_2': None,
            'referee': None
        }

        for col, default_value in missing_columns.items():
            if col not in upcoming_fights.columns:
                upcoming_fights[col] = default_value

        logger.info(f"Creating features for {len(upcoming_fights)} upcoming fights")

        # Prepare all completed fights with event dates for historical context
        completed_fights_with_dates = raw_data['fights'].merge(
            raw_data['events'][['event_id', 'event_date']],
            on='event_id',
            how='left'
        )

        # Create features (using all completed fights as historical context)
        prediction_dataset = self.feature_engineer.create_fight_features(
            fights_df=upcoming_fights,
            fighters_df=raw_data['fighters'],
            rounds_df=raw_data['rounds'],
            events_df=raw_data['upcoming_events'],
            is_prediction=True,
            all_completed_fights=completed_fights_with_dates
        )

        # Add metadata
        prediction_dataset['dataset_type'] = 'prediction'
        prediction_dataset['created_at'] = pd.Timestamp.now()

        return prediction_dataset

    def _save_datasets(self, training_dataset: pd.DataFrame, prediction_dataset: pd.DataFrame):
        """Save datasets to CSV files."""
        try:
            training_dataset.to_csv(self.training_dataset_file, index=False, encoding='utf-8')
            logger.info(f"Training dataset saved to {self.training_dataset_file}")
        except Exception as e:
            logger.error(f"Error saving training dataset: {e}")

        try:
            prediction_dataset.to_csv(self.prediction_dataset_file, index=False, encoding='utf-8')
            logger.info(f"Prediction dataset saved to {self.prediction_dataset_file}")
        except Exception as e:
            logger.error(f"Error saving prediction dataset: {e}")

    def get_feature_summary(self) -> dict:
        """Get summary of features in the datasets."""
        try:
            training_df = pd.read_csv(self.training_dataset_file, encoding='utf-8')
        except:
            training_df = pd.DataFrame()

        try:
            prediction_df = pd.read_csv(self.prediction_dataset_file, encoding='utf-8')
        except:
            prediction_df = pd.DataFrame()

        feature_categories = {
            'fight_context': [col for col in training_df.columns if any(x in col for x in ['weight_class', 'title_fight', 'gender', 'scheduled'])],
            'physical': [col for col in training_df.columns if any(x in col for x in ['age', 'height', 'weight', 'reach', 'stance', 'advantage'])],
            'record': [col for col in training_df.columns if any(x in col for x in ['wins', 'losses', 'draws', 'total_fights', 'percentage', 'experience'])],
            'historical': [col for col in training_df.columns if any(x in col for x in ['last_', 'avg_', 'days_since'])],
            'matchup': [col for col in training_df.columns if any(x in col for x in ['orthodox_vs', 'matchup'])],
            'metadata': [col for col in training_df.columns if any(x in col for x in ['fight_id', 'event_date', 'dataset_type', 'created_at'])]
        }

        summary = {
            'training_fights': len(training_df),
            'prediction_fights': len(prediction_df),
            'total_features': len(training_df.columns) - 1 if 'winner' in training_df.columns else len(training_df.columns),
            'feature_categories': {cat: len(features) for cat, features in feature_categories.items()},
            'feature_breakdown': feature_categories
        }

        return summary

    def update_datasets_incrementally(self) -> bool:
        """
        Update datasets incrementally if new data is available.

        Returns:
            True if datasets were updated, False if no update needed
        """
        try:
            # Check if raw data files are newer than dataset files
            raw_file_times = []
            for file_path in [self.events_file, self.fights_file, self.fighters_file, self.rounds_file]:
                if file_path.exists():
                    raw_file_times.append(file_path.stat().st_mtime)

            if not raw_file_times:
                logger.warning("No raw data files found")
                return False

            latest_raw_time = max(raw_file_times)

            # Check dataset file times
            dataset_times = []
            for file_path in [self.training_dataset_file, self.prediction_dataset_file]:
                if file_path.exists():
                    dataset_times.append(file_path.stat().st_mtime)

            if not dataset_times or latest_raw_time > min(dataset_times):
                logger.info("Raw data is newer than datasets, rebuilding...")
                self.build_datasets()
                return True
            else:
                logger.info("Datasets are up to date")
                return False

        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            return False

    def _can_use_cached_datasets(self) -> bool:
        """
        Check if cached datasets can be used (i.e., source data hasn't changed).

        Returns:
            True if cached datasets are valid and up-to-date
        """
        try:
            # Check if dataset files and metadata exist
            if not all(f.exists() for f in [
                self.training_dataset_file,
                self.prediction_dataset_file,
                self.dataset_metadata_file
            ]):
                logger.debug("Cache miss: Dataset files don't exist")
                return False

            # Load metadata
            with open(self.dataset_metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if source data files are newer than cached datasets
            dataset_creation_time = metadata.get('creation_timestamp', 0)

            # Get latest modification time of source data
            source_files = [
                self.events_file, self.fights_file, self.fighters_file,
                self.rounds_file, self.upcoming_events_file, self.upcoming_fights_file
            ]

            latest_source_time = 0
            for file_path in source_files:
                if file_path.exists():
                    latest_source_time = max(latest_source_time, file_path.stat().st_mtime)

            if latest_source_time > dataset_creation_time:
                logger.debug("Cache miss: Source data is newer than cached datasets")
                return False

            logger.debug("Cache hit: Using existing datasets")
            return True

        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
            return False

    def _load_cached_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load cached datasets from files.

        Returns:
            Tuple of (training_dataset, prediction_dataset)
        """
        try:
            training_dataset = pd.read_csv(self.training_dataset_file, encoding='utf-8')
            prediction_dataset = pd.read_csv(self.prediction_dataset_file, encoding='utf-8')

            # Load and log metadata
            with open(self.dataset_metadata_file, 'r') as f:
                metadata = json.load(f)

            logger.info(f" Loaded cached datasets:")
            logger.info(f"  Training: {len(training_dataset)} fights, {len(training_dataset.columns)} features")
            logger.info(f"  Prediction: {len(prediction_dataset)} fights, {len(prediction_dataset.columns)} features")
            logger.info(f"  Cache created: {metadata.get('creation_date', 'Unknown')}")

            return training_dataset, prediction_dataset

        except Exception as e:
            logger.error(f"Error loading cached datasets: {e}")
            raise

    def _save_datasets_with_metadata(
        self,
        training_dataset: pd.DataFrame,
        prediction_dataset: pd.DataFrame,
        min_fights_per_fighter: int,
        test_mode: bool
    ):
        """Save datasets with metadata for caching."""
        try:
            # Save datasets
            training_dataset.to_csv(self.training_dataset_file, index=False, encoding='utf-8')
            prediction_dataset.to_csv(self.prediction_dataset_file, index=False, encoding='utf-8')

            # Create metadata
            metadata = {
                'creation_timestamp': pd.Timestamp.now().timestamp(),
                'creation_date': pd.Timestamp.now().isoformat(),
                'training_fights': len(training_dataset),
                'prediction_fights': len(prediction_dataset),
                'total_features': len(training_dataset.columns),
                'min_fights_per_fighter': min_fights_per_fighter,
                'test_mode': test_mode,
                'feature_columns': list(training_dataset.columns)
            }

            # Save metadata
            with open(self.dataset_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Training dataset saved to {self.training_dataset_file}")
            logger.info(f"Prediction dataset saved to {self.prediction_dataset_file}")
            logger.info(f"Dataset metadata saved to {self.dataset_metadata_file}")

        except Exception as e:
            logger.error(f"Error saving datasets with metadata: {e}")

    def _save_datasets(self, training_dataset: pd.DataFrame, prediction_dataset: pd.DataFrame):
        """Legacy save method - now calls enhanced version."""
        self._save_datasets_with_metadata(training_dataset, prediction_dataset, 1, False)

    # ============================================================================
    # Smart Caching System for Training and Prediction Datasets
    # ============================================================================

    def _load_training_cache(self, raw_data: dict, min_fights_per_fighter: int) -> Optional[pd.DataFrame]:
        """
        Load training dataset from cache if valid.

        Args:
            raw_data: Raw data dictionary
            min_fights_per_fighter: Minimum fights requirement

        Returns:
            Training dataset if cache is valid, None otherwise
        """
        try:
            training_cache_file = self.data_folder / "training_dataset_cache.csv"
            training_metadata_file = self.data_folder / "training_cache_metadata.json"

            if not training_cache_file.exists() or not training_metadata_file.exists():
                logger.debug("Training cache miss: Files don't exist")
                return None

            # Load cache metadata
            with open(training_metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if parameters match
            if metadata.get('min_fights_per_fighter') != min_fights_per_fighter:
                logger.debug("Training cache miss: Parameter mismatch")
                return None

            # Check if completed events have changed
            current_events = set(raw_data['events']['event_id'].tolist()) if len(raw_data['events']) > 0 else set()
            cached_events = set(metadata.get('processed_events', []))

            if current_events != cached_events:
                logger.info(f"Training cache miss: Event changes detected")
                logger.info(f"  Cached events: {len(cached_events)}")
                logger.info(f"  Current events: {len(current_events)}")
                logger.info(f"  New events: {current_events - cached_events}")
                logger.info(f"  Removed events: {cached_events - current_events}")
                return None

            # Check completed data file modification times
            completed_files = [self.events_file, self.fights_file, self.fighters_file, self.rounds_file]
            cache_creation_time = metadata.get('creation_timestamp', 0)

            for file_path in completed_files:
                if file_path.exists() and file_path.stat().st_mtime > cache_creation_time:
                    logger.debug(f"Training cache miss: {file_path.name} is newer than cache")
                    return None

            # Load cached training dataset
            training_dataset = pd.read_csv(training_cache_file, encoding='utf-8')
            logger.info(f" Using cached training dataset ({len(training_dataset)} fights)")
            return training_dataset

        except Exception as e:
            logger.debug(f"Training cache load failed: {e}")
            return None

    def _save_training_cache(self, training_dataset: pd.DataFrame, raw_data: dict, min_fights_per_fighter: int):
        """
        Save training dataset to cache with metadata.

        Args:
            training_dataset: Training dataset to cache
            raw_data: Raw data dictionary
            min_fights_per_fighter: Minimum fights requirement
        """
        try:
            training_cache_file = self.data_folder / "training_dataset_cache.csv"
            training_metadata_file = self.data_folder / "training_cache_metadata.json"

            # Save training dataset
            training_dataset.to_csv(training_cache_file, index=False, encoding='utf-8')

            # Create metadata
            current_events = raw_data['events']['event_id'].tolist() if len(raw_data['events']) > 0 else []
            metadata = {
                'creation_timestamp': time.time(),
                'processed_events': current_events,
                'num_events': len(current_events),
                'num_fights': len(training_dataset),
                'min_fights_per_fighter': min_fights_per_fighter,
                'cache_type': 'training'
            }

            # Save metadata
            with open(training_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f" Training dataset cached ({len(training_dataset)} fights)")

        except Exception as e:
            logger.warning(f"Failed to save training cache: {e}")

    def _load_prediction_cache(self, raw_data: dict) -> Optional[pd.DataFrame]:
        """
        Load prediction dataset from cache if valid.

        Args:
            raw_data: Raw data dictionary

        Returns:
            Prediction dataset if cache is valid, None otherwise
        """
        try:
            prediction_cache_file = self.data_folder / "prediction_dataset_cache.csv"
            prediction_metadata_file = self.data_folder / "prediction_cache_metadata.json"

            if not prediction_cache_file.exists() or not prediction_metadata_file.exists():
                logger.debug("Prediction cache miss: Files don't exist")
                return None

            # Load cache metadata
            with open(prediction_metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if upcoming data has changed
            upcoming_files = [self.upcoming_events_file, self.upcoming_fights_file]
            cache_creation_time = metadata.get('creation_timestamp', 0)

            for file_path in upcoming_files:
                if file_path.exists() and file_path.stat().st_mtime > cache_creation_time:
                    logger.debug(f"Prediction cache miss: {file_path.name} is newer than cache")
                    return None

            # Also check if completed data changed (affects historical context)
            completed_files = [self.events_file, self.fights_file]
            for file_path in completed_files:
                if file_path.exists() and file_path.stat().st_mtime > cache_creation_time:
                    logger.debug(f"Prediction cache miss: {file_path.name} changed (affects context)")
                    return None

            # Load cached prediction dataset
            prediction_dataset = pd.read_csv(prediction_cache_file, encoding='utf-8')
            logger.info(f" Using cached prediction dataset ({len(prediction_dataset)} fights)")
            return prediction_dataset

        except Exception as e:
            logger.debug(f"Prediction cache load failed: {e}")
            return None

    def _save_prediction_cache(self, prediction_dataset: pd.DataFrame, raw_data: dict):
        """
        Save prediction dataset to cache with metadata.

        Args:
            prediction_dataset: Prediction dataset to cache
            raw_data: Raw data dictionary
        """
        try:
            prediction_cache_file = self.data_folder / "prediction_dataset_cache.csv"
            prediction_metadata_file = self.data_folder / "prediction_cache_metadata.json"

            # Save prediction dataset
            prediction_dataset.to_csv(prediction_cache_file, index=False, encoding='utf-8')

            # Create metadata
            upcoming_events = raw_data['upcoming_events']['event_id'].tolist() if len(raw_data['upcoming_events']) > 0 else []
            metadata = {
                'creation_timestamp': time.time(),
                'upcoming_events': upcoming_events,
                'num_upcoming_events': len(upcoming_events),
                'num_prediction_fights': len(prediction_dataset),
                'cache_type': 'prediction'
            }

            # Save metadata
            with open(prediction_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f" Prediction dataset cached ({len(prediction_dataset)} fights)")

        except Exception as e:
            logger.warning(f"Failed to save prediction cache: {e}")