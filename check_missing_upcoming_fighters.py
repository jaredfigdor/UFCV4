"""Check for fighters missing from upcoming fights."""
import pandas as pd
from pathlib import Path

data_folder = Path("ufc_data")

# Load data
upcoming_fights = pd.read_csv(data_folder / "upcoming_fight_data.csv")
fighters = pd.read_csv(data_folder / "fighter_data.csv")

print("=" * 70)
print("CHECKING FOR MISSING FIGHTER DATA IN UPCOMING FIGHTS")
print("=" * 70)

# Get all unique fighter IDs from upcoming fights
fighter_ids_in_upcoming = set(upcoming_fights['fighter_1'].unique()) | set(upcoming_fights['fighter_2'].unique())
fighter_ids_in_db = set(fighters['fighter_id'].unique())

missing_fighter_ids = fighter_ids_in_upcoming - fighter_ids_in_db

print(f"\nTotal unique fighters in upcoming fights: {len(fighter_ids_in_upcoming)}")
print(f"Total fighters in fighter_data.csv: {len(fighter_ids_in_db)}")
print(f"Missing fighters: {len(missing_fighter_ids)}")

if len(missing_fighter_ids) > 0:
    print(f"\n[ERROR] Found {len(missing_fighter_ids)} fighter IDs in upcoming fights but not in fighter database!")
    print(f"\nMissing fighter IDs:")
    for fid in list(missing_fighter_ids)[:20]:
        print(f"  {fid}")

    # Find which fights have missing fighters
    fights_with_missing = upcoming_fights[
        upcoming_fights['fighter_1'].isin(missing_fighter_ids) |
        upcoming_fights['fighter_2'].isin(missing_fighter_ids)
    ]

    print(f"\nAffected upcoming fights: {len(fights_with_missing)}")

    # Merge with events to get event names
    upcoming_events = pd.read_csv(data_folder / "upcoming_event_data.csv")
    fights_with_events = fights_with_missing.merge(
        upcoming_events[['event_id', 'event_name', 'event_date']],
        on='event_id',
        how='left'
    )

    print("\nAffected upcoming fights:")
    for _, fight in fights_with_events.head(10).iterrows():
        print(f"\n  Event: {fight.get('event_name', 'Unknown')} ({fight.get('event_date', 'Unknown')})")
        print(f"  Fight ID: {fight['fight_id']}")
        if fight['fighter_1'] in missing_fighter_ids:
            print(f"  Missing Fighter 1: {fight['fighter_1']}")
        if fight['fighter_2'] in missing_fighter_ids:
            print(f"  Missing Fighter 2: {fight['fighter_2']}")

    print("\n" + "=" * 70)
    print("SOLUTION:")
    print("=" * 70)
    print("Run fighter scraper to get missing fighter data:")
    print("  python -c \"from ufcscraper.ufc_scraper import UFCScraper; from pathlib import Path; s = UFCScraper(Path('ufc_data'), 1, 0.5); s.fighter_scraper.scrape_fighters()\"")
else:
    print("\n[OK] All fighters in upcoming fights have corresponding fighter data")

print("=" * 70)
