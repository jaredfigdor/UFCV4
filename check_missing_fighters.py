"""Check for fights with missing fighter data."""
import pandas as pd
from pathlib import Path

data_folder = Path("ufc_data")

# Load data
fights = pd.read_csv(data_folder / "fight_data.csv")
fighters = pd.read_csv(data_folder / "fighter_data.csv")

print("=" * 70)
print("CHECKING FOR MISSING FIGHTER DATA")
print("=" * 70)

# Get all unique fighter IDs from fights
fighter_ids_in_fights = set(fights['fighter_1'].unique()) | set(fights['fighter_2'].unique())
fighter_ids_in_db = set(fighters['fighter_id'].unique())

missing_fighter_ids = fighter_ids_in_fights - fighter_ids_in_db

print(f"\nTotal unique fighters in fights: {len(fighter_ids_in_fights)}")
print(f"Total fighters in fighter_data.csv: {len(fighter_ids_in_db)}")
print(f"Missing fighters: {len(missing_fighter_ids)}")

if len(missing_fighter_ids) > 0:
    print(f"\n[ERROR] Found {len(missing_fighter_ids)} fighter IDs in fights but not in fighter database!")

    # Find which fights have missing fighters
    fights_with_missing = fights[
        fights['fighter_1'].isin(missing_fighter_ids) |
        fights['fighter_2'].isin(missing_fighter_ids)
    ]

    print(f"\nAffected fights: {len(fights_with_missing)}")
    print("\nRecent affected fights:")

    # Merge with events to get event names
    events = pd.read_csv(data_folder / "event_data.csv")
    fights_with_events = fights_with_missing.merge(
        events[['event_id', 'event_name', 'event_date']],
        on='event_id',
        how='left'
    )

    recent_affected = fights_with_events.sort_values('event_date', ascending=False).head(10)

    for _, fight in recent_affected.iterrows():
        print(f"\n  Event: {fight['event_name']} ({fight['event_date']})")
        print(f"  Fight ID: {fight['fight_id']}")

        if fight['fighter_1'] in missing_fighter_ids:
            print(f"  Missing Fighter 1: {fight['fighter_1']}")
        if fight['fighter_2'] in missing_fighter_ids:
            print(f"  Missing Fighter 2: {fight['fighter_2']}")

    print("\n" + "=" * 70)
    print("SOLUTION:")
    print("=" * 70)
    print("Need to scrape fighter data for these missing fighters.")
    print("This likely happened because fights were moved but fighter data wasn't scraped.")
    print("\nRecommendation: Run the fighter scraper to update fighter_data.csv")
else:
    print("\n[OK] All fighters in fights have corresponding fighter data")

print("=" * 70)
