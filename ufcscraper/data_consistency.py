"""
Data consistency management module for UFC Scraper.

This module contains the DataConsistencyManager class which handles
automatic data consistency between completed and upcoming events,
ensuring that events are properly migrated from upcoming to completed
status and maintaining data integrity across all CSV files.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


class DataConsistencyManager:
    """Manages data consistency between completed and upcoming UFC events.

    This class handles the automatic migration of events from upcoming
    status to completed status, ensuring data integrity and consistency
    across all CSV files in the UFC scraper system.
    """

    def __init__(self, data_folder: Path | str):
        """Initialize the DataConsistencyManager.

        Args:
            data_folder: Path to the folder containing the CSV data files.
        """
        self.data_folder = Path(data_folder)

        # Define file paths
        self.completed_events_file = self.data_folder / "event_data.csv"
        self.upcoming_events_file = self.data_folder / "upcoming_event_data.csv"
        self.completed_fights_file = self.data_folder / "fight_data.csv"
        self.upcoming_fights_file = self.data_folder / "upcoming_fight_data.csv"
        self.round_data_file = self.data_folder / "round_data.csv"

    def manage_event_consistency(self) -> None:
        """Main method to manage data consistency between upcoming and completed events.

        This method:
        1. Identifies events that have moved from upcoming to completed based on dates
        2. Migrates event data from upcoming to completed files
        3. Migrates fight data from upcoming to completed files
        4. Removes migrated events from upcoming files
        5. Moves future events from completed to upcoming if needed
        6. Logs all operations performed
        """
        logger.info("Starting data consistency management...")

        # Load data
        completed_events = self._load_events(self.completed_events_file)
        upcoming_events = self._load_events(self.upcoming_events_file)

        if completed_events.empty and upcoming_events.empty:
            logger.info("No data consistency operations needed (no data files)")
            return

        # Fix events that are in the wrong files based on dates
        self._fix_events_by_date(completed_events, upcoming_events)

        # Find events to migrate (after date-based fixes)
        completed_events = self._load_events(self.completed_events_file)
        upcoming_events = self._load_events(self.upcoming_events_file)

        events_to_migrate = self._find_events_to_migrate(completed_events, upcoming_events)

        if events_to_migrate:
            logger.info(f"Found {len(events_to_migrate)} events to migrate from upcoming to completed")
            # Migrate events and fights
            self._migrate_events(events_to_migrate)
            self._migrate_fights(events_to_migrate)
            # Clean up upcoming files
            self._remove_completed_events_from_upcoming(events_to_migrate)

        logger.info("Data consistency management completed successfully")

    def _fix_events_by_date(self, completed_events: pd.DataFrame, upcoming_events: pd.DataFrame) -> None:
        """Fix events that are in the wrong files based on their actual dates.

        Args:
            completed_events: DataFrame of completed events.
            upcoming_events: DataFrame of upcoming events.
        """
        from datetime import date

        today = date.today()

        # Check completed events for future events that should be moved to upcoming
        if not completed_events.empty and 'event_date' in completed_events.columns:
            future_events_in_completed = completed_events[
                pd.to_datetime(completed_events['event_date']).dt.date > today
            ]

            if not future_events_in_completed.empty:
                logger.info(f"Found {len(future_events_in_completed)} future events in completed data")
                self._move_events_to_upcoming(future_events_in_completed)

        # Check upcoming events for past events that should be moved to completed
        if not upcoming_events.empty and 'event_date' in upcoming_events.columns:
            past_events_in_upcoming = upcoming_events[
                pd.to_datetime(upcoming_events['event_date']).dt.date <= today
            ]

            if not past_events_in_upcoming.empty:
                logger.info(f"Found {len(past_events_in_upcoming)} past events in upcoming data")
                self._move_events_to_completed(past_events_in_upcoming)

    def _move_events_to_upcoming(self, events_to_move: pd.DataFrame) -> None:
        """Move events from completed to upcoming files.

        Args:
            events_to_move: DataFrame of events to move.
        """
        # Add to upcoming events file
        for _, event in events_to_move.iterrows():
            logger.info(f"Moving future event '{event['event_name']}' ({event['event_date']}) to upcoming")

        # Append to upcoming events file
        if not events_to_move.empty:
            if self.upcoming_events_file.exists():
                events_to_move.to_csv(self.upcoming_events_file, mode='a', header=False, index=False, encoding='utf-8')
            else:
                events_to_move.to_csv(self.upcoming_events_file, index=False, encoding='utf-8')

            # Remove from completed events file
            event_ids_to_remove = set(events_to_move['event_id'])
            self._remove_events_from_file(self.completed_events_file, event_ids_to_remove, "events")

    def _move_events_to_completed(self, events_to_move: pd.DataFrame) -> None:
        """Move events from upcoming to completed files.

        Args:
            events_to_move: DataFrame of events to move.
        """
        # Add to completed events file
        for _, event in events_to_move.iterrows():
            logger.info(f"Moving past event '{event['event_name']}' ({event['event_date']}) to completed")

        # Append to completed events file
        if not events_to_move.empty:
            if self.completed_events_file.exists():
                events_to_move.to_csv(self.completed_events_file, mode='a', header=False, index=False, encoding='utf-8')
            else:
                events_to_move.to_csv(self.completed_events_file, index=False, encoding='utf-8')

            # Remove from upcoming events file
            event_ids_to_remove = set(events_to_move['event_id'])
            self._remove_events_from_file(self.upcoming_events_file, event_ids_to_remove, "events")

    def _load_events(self, file_path: Path) -> pd.DataFrame:
        """Load event data from CSV file.

        Args:
            file_path: Path to the CSV file to load.

        Returns:
            DataFrame containing event data, or empty DataFrame if file doesn't exist.
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()

        try:
            # Try different encodings to handle Unicode issues
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    return pd.read_csv(file_path, parse_dates=['event_date'], encoding=encoding)
                except UnicodeDecodeError:
                    continue
            # If all fail, try with error handling
            return pd.read_csv(file_path, parse_dates=['event_date'], encoding='utf-8', encoding_errors='replace')
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def _find_events_to_migrate(self, completed_events: pd.DataFrame, upcoming_events: pd.DataFrame) -> Set[str]:
        """Find events that appear in both upcoming and completed data.

        These are events that have moved from upcoming to completed status
        and need to be cleaned up from the upcoming data.

        Args:
            completed_events: DataFrame of completed events.
            upcoming_events: DataFrame of upcoming events.

        Returns:
            Set of event IDs that need to be migrated.
        """
        if 'event_id' not in completed_events.columns or 'event_id' not in upcoming_events.columns:
            logger.warning("Missing event_id column in event data")
            return set()

        completed_event_ids = set(completed_events['event_id'])
        upcoming_event_ids = set(upcoming_events['event_id'])

        # Events that are now completed but still in upcoming
        events_to_migrate = completed_event_ids.intersection(upcoming_event_ids)

        return events_to_migrate

    def _migrate_events(self, event_ids: Set[str]) -> None:
        """Migrate event data from upcoming to completed (already done by scraper).

        Since the scraper already adds completed events to the completed file,
        we just need to log this operation.

        Args:
            event_ids: Set of event IDs that have been migrated.
        """
        logger.info(f"Events already migrated to completed by scraper: {len(event_ids)} events")
        for event_id in event_ids:
            logger.debug(f"Event migrated: {event_id}")

    def _migrate_fights(self, event_ids: Set[str]) -> None:
        """Migrate fight data from upcoming to completed (already done by scraper).

        Since the scraper already adds completed fights to the completed file,
        we just need to log this operation.

        Args:
            event_ids: Set of event IDs whose fights have been migrated.
        """
        if not self.upcoming_fights_file.exists():
            return

        try:
            # Try different encodings to handle Unicode issues
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    upcoming_fights = pd.read_csv(self.upcoming_fights_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                upcoming_fights = pd.read_csv(self.upcoming_fights_file, encoding='utf-8', encoding_errors='replace')
            if 'event_id' not in upcoming_fights.columns:
                return

            # Count fights that would be migrated
            fights_to_migrate = upcoming_fights[upcoming_fights['event_id'].isin(event_ids)]
            fight_count = len(fights_to_migrate)

            if fight_count > 0:
                logger.info(f"Fights already migrated to completed by scraper: {fight_count} fights")

        except Exception as e:
            logger.error(f"Error checking upcoming fights for migration: {e}")

    def _remove_completed_events_from_upcoming(self, event_ids: Set[str]) -> None:
        """Remove completed events from upcoming event and fight files.

        Args:
            event_ids: Set of event IDs to remove from upcoming files.
        """
        # Remove from upcoming events
        self._remove_events_from_file(self.upcoming_events_file, event_ids, "events")

        # Remove from upcoming fights
        self._remove_events_from_file(self.upcoming_fights_file, event_ids, "fights")

    def _remove_events_from_file(self, file_path: Path, event_ids: Set[str], data_type: str) -> None:
        """Remove events from a specific CSV file.

        Args:
            file_path: Path to the CSV file to clean.
            event_ids: Set of event IDs to remove.
            data_type: Type of data being cleaned (for logging).
        """
        if not file_path.exists():
            return

        try:
            # Try different encodings to handle Unicode issues
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')

            if 'event_id' not in df.columns:
                logger.warning(f"No event_id column in {file_path}")
                return

            original_count = len(df)
            df_cleaned = df[~df['event_id'].isin(event_ids)]
            removed_count = original_count - len(df_cleaned)

            if removed_count > 0:
                # Save cleaned data
                df_cleaned.to_csv(file_path, index=False, encoding='utf-8')
                logger.info(f"Removed {removed_count} completed {data_type} from upcoming data in {file_path.name}")
            else:
                logger.debug(f"No {data_type} to remove from {file_path.name}")

        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {e}")

    def validate_data_integrity(self) -> bool:
        """Validate data integrity across all files.

        Returns:
            True if data integrity is valid, False otherwise.
        """
        logger.info("Validating data integrity...")

        issues_found = False

        # Check for events in both upcoming and completed
        completed_events = self._load_events(self.completed_events_file)
        upcoming_events = self._load_events(self.upcoming_events_file)

        if not completed_events.empty and not upcoming_events.empty:
            if 'event_id' in completed_events.columns and 'event_id' in upcoming_events.columns:
                overlapping_events = set(completed_events['event_id']).intersection(
                    set(upcoming_events['event_id'])
                )

                if overlapping_events:
                    logger.warning(f"Found {len(overlapping_events)} events in both upcoming and completed data")
                    issues_found = True

        # Check for orphaned fight data
        if self.completed_fights_file.exists() and not completed_events.empty:
            try:
                # Try different encodings to handle Unicode issues
                for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                    try:
                        fights = pd.read_csv(self.completed_fights_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    fights = pd.read_csv(self.completed_fights_file, encoding='utf-8', encoding_errors='replace')
                if 'event_id' in fights.columns and 'event_id' in completed_events.columns:
                    fight_event_ids = set(fights['event_id'])
                    event_ids = set(completed_events['event_id'])
                    orphaned_fights = fight_event_ids - event_ids

                    if orphaned_fights:
                        logger.warning(f"Found {len(orphaned_fights)} fights with no corresponding event")
                        issues_found = True

            except Exception as e:
                logger.error(f"Error validating fight data: {e}")
                issues_found = True

        if not issues_found:
            logger.info("Data integrity validation passed")

        return not issues_found