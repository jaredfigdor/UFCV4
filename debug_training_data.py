"""Debug script to understand training data caching issue."""
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

data_folder = Path("ufc_data")

# Load all data
events = pd.read_csv(data_folder / "event_data.csv")
fights = pd.read_csv(data_folder / "fight_data.csv")

with open(data_folder / "training_cache_metadata.json") as f:
    cache_metadata = json.load(f)

print("=" * 60)
print("TRAINING DATA DIAGNOSIS")
print("=" * 60)

# Check event counts
print(f"\n1. EVENT COUNTS:")
print(f"   Events in event_data.csv: {len(events)}")
print(f"   Events in cache metadata: {cache_metadata['num_events']}")
print(f"   Cache creation time: {datetime.fromtimestamp(cache_metadata['creation_timestamp'])}")

# Check fight counts
print(f"\n2. FIGHT COUNTS:")
print(f"   Fights in fight_data.csv: {len(fights)}")
print(f"   Fights in cache metadata: {cache_metadata['num_fights']}")
print(f"   Difference: {len(fights) - cache_metadata['num_fights']} fights")

# Check recent events
events['event_date'] = pd.to_datetime(events['event_date'])
recent_events = events.sort_values('event_date', ascending=False).head(10)

print(f"\n3. MOST RECENT 10 EVENTS IN event_data.csv:")
for idx, row in recent_events.iterrows():
    event_id = row['event_id']
    in_cache = event_id in cache_metadata['processed_events']
    print(f"   {'[CACHED]' if in_cache else '[NEW]   '} {row['event_name']:45s} {row['event_date'].date()}")

# Count fights per recent event
print(f"\n4. FIGHT COUNTS FOR RECENT EVENTS:")
for idx, row in recent_events.iterrows():
    event_id = row['event_id']
    fight_count = len(fights[fights['event_id'] == event_id])
    in_cache = event_id in cache_metadata['processed_events']
    print(f"   {'[CACHED]' if in_cache else '[NEW]   '} {row['event_name']:45s} {fight_count:3d} fights")

# Check what events are in cache but maybe not in event_data
cached_events = set(cache_metadata['processed_events'])
current_events = set(events['event_id'].tolist())

new_events = current_events - cached_events
removed_events = cached_events - current_events

print(f"\n5. CACHE DELTA:")
print(f"   Events in current data but not in cache: {len(new_events)}")
print(f"   Events in cache but not in current data: {len(removed_events)}")

if len(new_events) > 0:
    print(f"\n   NEW EVENTS NOT IN CACHE:")
    new_event_data = events[events['event_id'].isin(new_events)].sort_values('event_date')
    for idx, row in new_event_data.iterrows():
        print(f"     - {row['event_name']:45s} {row['event_date'].date()}")

if len(removed_events) > 0:
    print(f"\n   EVENTS IN CACHE BUT REMOVED FROM DATA:")
    print(f"     {len(removed_events)} events")

# Check upcoming data
upcoming_events = pd.read_csv(data_folder / "upcoming_event_data.csv")
upcoming_fights = pd.read_csv(data_folder / "upcoming_fight_data.csv")

print(f"\n6. UPCOMING DATA:")
print(f"   Upcoming events: {len(upcoming_events)}")
print(f"   Upcoming fights: {len(upcoming_fights)}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)

if len(fights) > cache_metadata['num_fights']:
    print(f"✗ Cache is STALE: {len(fights) - cache_metadata['num_fights']} new fights not in cache")
    print(f"  Recommendation: Delete training_cache_metadata.json and training_dataset_cache.csv")
elif len(new_events) > 0:
    print(f"✗ Cache is STALE: {len(new_events)} new events not in cache")
    print(f"  Recommendation: Delete training_cache_metadata.json and training_dataset_cache.csv")
else:
    print(f"✓ Cache appears up-to-date with current data")
    print(f"  The system is correctly using cached training data")

print("=" * 60)
