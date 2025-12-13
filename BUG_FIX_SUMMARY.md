# Training Data Not Growing - Bug Fix Summary

## Problem

Training dataset was stuck at **8,440 fights** despite completed events being moved from upcoming to completed data. The log showed:

```
Creating features for 8440 training fights
Creating features for 8440 fights (prediction=False)
```

This number never increased even though 6-8 events had completed and been "migrated."

## Root Cause

The `DataConsistencyManager` had a critical bug in the `_move_events_to_completed()` method:

1. **Events were moved** from `upcoming_event_data.csv` → `event_data.csv` ✓
2. **Fights were NOT moved** from `upcoming_fight_data.csv` → `fight_data.csv` ✗
3. The `_migrate_fights()` method only **logged** that fights were migrated but didn't actually move them

Result: Events got added to `event_data.csv` but their fight data was left orphaned in `upcoming_fight_data.csv` and eventually deleted without being saved to `fight_data.csv`.

## Events Affected

Two recent events lost their fight data:
- **UFC 322: Della Maddalena vs. Makhachev** (2025-11-15) - 14 fights lost
- **UFC Fight Night** (2025-11-22) - 15 fights lost

**Total: 29 fights lost**

## Fix Applied

### 1. Fixed `DataConsistencyManager` ([data_consistency.py](ufcscraper/data_consistency.py))

**Modified `_move_events_to_completed()` (line 140-162):**
- Added call to `_move_fights_to_completed()` to move fight data when events are moved
- Now properly migrates both events AND their associated fights

**Created new `_move_fights_to_completed()` method (line 238-281):**
- Loads upcoming fights from `upcoming_fight_data.csv`
- Finds fights matching the event IDs being migrated
- Appends those fights to `fight_data.csv`
- Properly logs the migration

**Updated `_migrate_fights()` (line 228-236):**
- Now delegates to `_move_fights_to_completed()` for actual migration
- Maintains backward compatibility with existing code

### 2. Recovered Lost Fight Data

**Created and ran `recover_missing_fights.py`:**
- Identified 2 events with 0 fights in `fight_data.csv`
- Found the missing fights still in `upcoming_fight_data.csv`
- Moved 29 fights from upcoming to completed
- Updated `upcoming_fight_data.csv` to remove migrated fights

**Results:**
- Fight count increased from 8,440 → 8,469 fights
- Both affected events now have their proper fight data

### 3. Cleared Stale Cache

**Deleted cache files to force rebuild:**
- `ufc_data/training_cache_metadata.json`
- `ufc_data/training_dataset_cache.csv`

## Verification

**Before Fix:**
```
Total completed fights: 8440
UFC 322: Della Maddalena vs. Makhachev: 0 fights
UFC Fight Night (2025-11-22): 0 fights
```

**After Fix:**
```
Total completed fights: 8469
UFC 322: Della Maddalena vs. Makhachev: 14 fights
UFC Fight Night (2025-11-22): 15 fights
Average fights per event: 11.2
```

## Next Steps

1. **Run the automation:** `python app.py`
   - Training dataset will be rebuilt from scratch (cache deleted)
   - Should now show **~8469 fights** instead of 8440
   - Future event migrations will properly include fight data

2. **Monitor future runs:**
   - Check that fight count increases when events complete
   - Log should show: "Moved X fights to completed data"

## Files Modified

1. **ufcscraper/data_consistency.py** - Fixed event/fight migration logic
2. **recover_missing_fights.py** - One-time recovery script (can be deleted)
3. **BUG_FIX_SUMMARY.md** - This document

## Prevention

The bug is now fixed in the `DataConsistencyManager`. Future event migrations will automatically:
1. Move events from upcoming → completed
2. Move associated fights from upcoming → completed
3. Clean up upcoming files properly

The training dataset will now grow correctly as new events complete!
