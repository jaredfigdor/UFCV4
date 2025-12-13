"""Scrape fight data for the 2 events that lost their fight data."""
import logging
import sys
from pathlib import Path
import pandas as pd

from ufcscraper.ufc_scraper import UFCScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

data_folder = Path("ufc_data")

# Load events to find the ones without fights
events = pd.read_csv(data_folder / "event_data.csv")
fights = pd.read_csv(data_folder / "fight_data.csv")

# Find events with 0 fights (recent ones)
events['event_date'] = pd.to_datetime(events['event_date'])
recent_events = events[events['event_date'] >= '2025-11-01'].sort_values('event_date')

events_needing_scraping = []
for _, evt in recent_events.iterrows():
    event_id = evt['event_id']
    fight_count = len(fights[fights['event_id'] == event_id])
    if fight_count == 0:
        events_needing_scraping.append(evt)
        logger.info(f"Event needs fight data: {evt['event_name']} ({evt['event_date'].date()})")

if len(events_needing_scraping) == 0:
    logger.info("No events need scraping - all events have fight data")
    sys.exit(0)

logger.info(f"\n{'='*70}")
logger.info(f"Found {len(events_needing_scraping)} events needing fight data")
logger.info(f"{'='*70}\n")

# Initialize scraper
logger.info("Initializing UFC scraper...")
scraper = UFCScraper(
    data_folder=data_folder,
    n_sessions=1,  # Single session for Windows compatibility
    delay=0.5
)

# Scrape completed events (this will update recent events)
logger.info("\nScraping completed events to get latest data...")
scraper.event_scraper.scrape_events()

# Scrape fighters (in case there are new fighters)
logger.info("\nUpdating fighter data...")
scraper.fighter_scraper.scrape_fighters()

# Scrape fights for completed events
logger.info("\nScraping completed fights...")
scraper.fight_scraper.scrape_fights()

# Scrape round data
logger.info("\nScraping round data...")
scraper.fight_scraper.scrape_rounds()

# Verify the fix
logger.info(f"\n{'='*70}")
logger.info("VERIFICATION")
logger.info(f"{'='*70}\n")

fights_updated = pd.read_csv(data_folder / "fight_data.csv")
logger.info(f"Total fights after scraping: {len(fights_updated)} (was {len(fights)})")

for evt_series in events_needing_scraping:
    event_id = evt_series['event_id']
    fight_count = len(fights_updated[fights_updated['event_id'] == event_id])
    logger.info(f"  {evt_series['event_name']:45s} {fight_count:3d} fights")

logger.info(f"\n{'='*70}")
logger.info("NEXT STEPS:")
logger.info(f"{'='*70}")
logger.info("1. Delete training cache files:")
logger.info("   rm ufc_data/training_cache_metadata.json")
logger.info("   rm ufc_data/training_dataset_cache.csv")
logger.info("2. Run: python app.py")
logger.info(f"{'='*70}\n")
