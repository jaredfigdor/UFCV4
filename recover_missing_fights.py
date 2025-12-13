"""
Recovery script to manually move fights for events that lost their fight data.
This fixes the 2 events (UFC 322 and UFC Fight Night Nov 22) that have 0 fights.
"""
import pandas as pd
from pathlib import Path

data_folder = Path("ufc_data")

# Load data
upcoming_fights = pd.read_csv(data_folder / "upcoming_fight_data.csv")
completed_fights = pd.read_csv(data_folder / "fight_data.csv")
events = pd.read_csv(data_folder / "event_data.csv")

print("=" * 70)
print("FIGHT DATA RECOVERY SCRIPT")
print("=" * 70)

# Find events with 0 fights
events['event_date'] = pd.to_datetime(events['event_date'])
recent_events = events[events['event_date'] >= '2025-11-01'].sort_values('event_date')

events_missing_fights = []
for _, evt in recent_events.iterrows():
    event_id = evt['event_id']
    fight_count = len(completed_fights[completed_fights['event_id'] == event_id])
    if fight_count == 0:
        events_missing_fights.append(event_id)
        print(f"\n[MISSING FIGHTS] {evt['event_name']} ({evt['event_date'].date()})")

        # Check if fights exist in upcoming data
        upcoming_count = len(upcoming_fights[upcoming_fights['event_id'] == event_id])
        print(f"  Fights in upcoming_fight_data.csv: {upcoming_count}")

        if upcoming_count > 0:
            # Move these fights
            fights_to_move = upcoming_fights[upcoming_fights['event_id'] == event_id]

            print(f"  RECOVERING {len(fights_to_move)} fights...")

            # Append to completed fights
            fights_to_move.to_csv(
                data_folder / "fight_data.csv",
                mode='a',
                header=False,
                index=False,
                encoding='utf-8'
            )

            # Remove from upcoming fights
            upcoming_fights = upcoming_fights[upcoming_fights['event_id'] != event_id]

            print(f"  [OK] Moved {len(fights_to_move)} fights to fight_data.csv")
        else:
            print(f"  [ERROR] No fights found in upcoming data either!")

if len(events_missing_fights) > 0:
    # Save cleaned upcoming fights
    upcoming_fights.to_csv(
        data_folder / "upcoming_fight_data.csv",
        index=False,
        encoding='utf-8'
    )
    print(f"\n[OK] Updated upcoming_fight_data.csv")

    # Verify recovery
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    completed_fights_new = pd.read_csv(data_folder / "fight_data.csv")
    print(f"Total fights in fight_data.csv: {len(completed_fights_new)}")
    print(f"Previous count: {len(completed_fights)}")
    print(f"Added: {len(completed_fights_new) - len(completed_fights)} fights")

    print("\nRecovered event fight counts:")
    for event_id in events_missing_fights:
        evt = events[events['event_id'] == event_id].iloc[0]
        fight_count = len(completed_fights_new[completed_fights_new['event_id'] == event_id])
        print(f"  {evt['event_name']:45s} {fight_count:3d} fights")
else:
    print("\n[OK] No events with missing fights found")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Delete training cache to force rebuild:")
print("   - ufc_data/training_cache_metadata.json")
print("   - ufc_data/training_dataset_cache.csv")
print("2. Run python app.py to rebuild with corrected data")
print("=" * 70)
