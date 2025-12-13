"""Test the data consistency fix to move today's events to upcoming."""
import logging
import sys
from pathlib import Path
import pandas as pd

from ufcscraper.data_consistency import DataConsistencyManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

data_folder = Path("ufc_data")

logger.info("=" * 70)
logger.info("TESTING DATA CONSISTENCY FIX")
logger.info("=" * 70)

# Check current state
events = pd.read_csv(data_folder / "event_data.csv")
upcoming_events = pd.read_csv(data_folder / "upcoming_event_data.csv")
fights = pd.read_csv(data_folder / "fight_data.csv")
upcoming_fights = pd.read_csv(data_folder / "upcoming_fight_data.csv")

events['event_date'] = pd.to_datetime(events['event_date'])
upcoming_events['event_date'] = pd.to_datetime(upcoming_events['event_date'])

logger.info(f"\nBEFORE CONSISTENCY MANAGEMENT:")
logger.info(f"  Completed events: {len(events)}")
logger.info(f"  Upcoming events: {len(upcoming_events)}")
logger.info(f"  Completed fights: {len(fights)}")
logger.info(f"  Upcoming fights: {len(upcoming_fights)}")

# Find today's events in completed
from datetime import date
today = date.today()
logger.info(f"\nToday's date: {today}")

todays_events = events[events['event_date'].dt.date == today]
if len(todays_events) > 0:
    logger.info(f"\nEvents dated today in COMPLETED data:")
    for _, evt in todays_events.iterrows():
        event_id = evt['event_id']
        fight_count = len(fights[fights['event_id'] == event_id])
        logger.info(f"  - {evt['event_name']} ({evt['event_date'].date()}): {fight_count} fights")

# Run consistency management
logger.info(f"\n{'='*70}")
logger.info("RUNNING CONSISTENCY MANAGEMENT...")
logger.info(f"{'='*70}\n")

manager = DataConsistencyManager(data_folder)
manager.manage_event_consistency()

# Check after
events_after = pd.read_csv(data_folder / "event_data.csv")
upcoming_events_after = pd.read_csv(data_folder / "upcoming_event_data.csv")
fights_after = pd.read_csv(data_folder / "fight_data.csv")
upcoming_fights_after = pd.read_csv(data_folder / "upcoming_fight_data.csv")

logger.info(f"\n{'='*70}")
logger.info("AFTER CONSISTENCY MANAGEMENT:")
logger.info(f"{'='*70}")
logger.info(f"  Completed events: {len(events_after)} (was {len(events)})")
logger.info(f"  Upcoming events: {len(upcoming_events_after)} (was {len(upcoming_events)})")
logger.info(f"  Completed fights: {len(fights_after)} (was {len(fights)})")
logger.info(f"  Upcoming fights: {len(upcoming_fights_after)} (was {len(upcoming_fights)})")

# Check if today's events moved
events_after['event_date'] = pd.to_datetime(events_after['event_date'])
todays_events_after = events_after[events_after['event_date'].dt.date == today]

if len(todays_events_after) == 0:
    logger.info(f"\n[SUCCESS] No events dated today in completed data")
else:
    logger.info(f"\n[WARNING] Still {len(todays_events_after)} events dated today in completed data:")
    for _, evt in todays_events_after.iterrows():
        logger.info(f"  - {evt['event_name']} ({evt['event_date'].date()})")

logger.info(f"\n{'='*70}")
