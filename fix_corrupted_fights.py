"""Fix corrupted fight data by removing bad entries."""
import pandas as pd
from pathlib import Path

data_folder = Path("ufc_data")

# Load fight data
fights = pd.read_csv(data_folder / "fight_data.csv")

print("=" * 70)
print("FIXING CORRUPTED FIGHT DATA")
print("=" * 70)

# Find corrupted fights (those with 'F' as a fighter ID)
corrupted_fights = fights[(fights['fighter_1'] == 'F') | (fights['fighter_2'] == 'F')]

print(f"\nTotal fights: {len(fights)}")
print(f"Corrupted fights: {len(corrupted_fights)}")

if len(corrupted_fights) > 0:
    # Get event IDs of corrupted fights
    corrupted_event_ids = corrupted_fights['event_id'].unique()
    print(f"\nAffected events: {len(corrupted_event_ids)}")

    # Load events to show which ones
    events = pd.read_csv(data_folder / "event_data.csv")
    affected_events = events[events['event_id'].isin(corrupted_event_ids)]

    print("\nAffected events:")
    for _, evt in affected_events.iterrows():
        count = len(corrupted_fights[corrupted_fights['event_id'] == evt['event_id']])
        print(f"  - {evt['event_name']} ({evt['event_date']}): {count} corrupted fights")

    # Remove corrupted fights
    fights_cleaned = fights[~((fights['fighter_1'] == 'F') | (fights['fighter_2'] == 'F'))]

    print(f"\nRemoving {len(corrupted_fights)} corrupted fights...")
    print(f"Fights after cleanup: {len(fights_cleaned)}")

    # Save cleaned data
    fights_cleaned.to_csv(data_folder / "fight_data.csv", index=False, encoding='utf-8')
    print("\n[OK] Saved cleaned fight_data.csv")

    # Also remove these events from event_data since they have no valid fights
    print(f"\nRemoving {len(corrupted_event_ids)} events with no valid fight data...")
    events_cleaned = events[~events['event_id'].isin(corrupted_event_ids)]
    events_cleaned.to_csv(data_folder / "event_data.csv", index=False, encoding='utf-8')
    print(f"[OK] Saved cleaned event_data.csv")

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"Removed {len(corrupted_fights)} corrupted fights")
    print(f"Removed {len(corrupted_event_ids)} events without valid fight data")
    print(f"Final fight count: {len(fights_cleaned)}")
    print(f"Final event count: {len(events_cleaned)}")
    print("\nThe events need to be re-scraped to get proper fight data.")
    print("They will be automatically scraped on the next run since they're")
    print("recent events that the scraper will check for updates.")

else:
    print("\n[OK] No corrupted fights found")

print("=" * 70)
