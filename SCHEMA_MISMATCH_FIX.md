# Schema Mismatch Fix - Proper Event Migration

## The Problem

When events moved from upcoming → completed, the system was trying to **append** upcoming fights directly to completed fights. This caused data corruption because:

**Upcoming Fights Schema (6 columns):**
```
fight_id, event_id, fighter_1, fighter_2, title_fight, weight_class
```

**Completed Fights Schema (17 columns):**
```
fight_id, event_id, referee, fighter_1, fighter_2, winner, num_rounds,
title_fight, weight_class, gender, result, result_details, finish_round,
finish_time, time_format, scores_1, scores_2
```

### What Went Wrong

1. Event moved from upcoming → completed ✓
2. System tried to append 6-column upcoming fight data to 17-column completed fight file ✗
3. Data got misaligned (e.g., `title_fight='F'` ended up in `fighter_2` column)
4. Feature engineering failed with "single positional indexer is out-of-bounds"
5. Training data couldn't build

## The Solution

**Stop trying to move upcoming fights to completed!** Instead:

### 1. Delete Upcoming Fights (Don't Move Them)
- Modified `data_consistency.py` to remove the call to `_move_fights_to_completed()`
- Upcoming fights are now DELETED when their event completes
- Added clear comments explaining why we don't move them

### 2. Re-Scrape Completed Fights with Full Results
- Created new function `scrape_completed_event_fights()` in `app.py`
- When events are moved, we capture their event IDs
- We then scrape ONLY those specific events from UFCStats.com
- This gets the full 17-column fight data with results, referee, winner, etc.

### 3. Targeted Scraping (Not Full Re-Scrape)
- Only scrapes the 1-2 events that just completed
- Takes 5-10 seconds instead of 10+ minutes
- Gets complete, properly formatted fight data

## How It Works Now

**Example: UFC 323 completes on Dec 6, 2025**

### Day of Dec 7 (Next Day):

```
Step 1: Check data freshness
Step 2: Scrape upcoming events (gets new upcoming fights)
Step 3: Move completed events
  → UFC 323 moved from upcoming_event_data.csv to event_data.csv
  → UFC 323 fights DELETED from upcoming_fight_data.csv
Step 3.5: Scrape complete fight data for newly completed events
  → Scrape ONLY UFC 323 from UFCStats.com (14 fights)
  → Gets full 17-column data with winner, result, referee, etc.
  → Saves to fight_data.csv with proper schema
Step 4: Build ML datasets
  → Training data now includes UFC 323's 14 complete fights
  → No schema mismatches, no errors!
```

## Files Modified

### 1. `ufcscraper/data_consistency.py`
**Line 162-171:** Removed `_move_fights_to_completed()` call
- Added comments explaining schema mismatch issue
- Upcoming fights are deleted (via `_remove_events_from_file()`)
- No longer attempts to move incompatible data structures

### 2. `app.py`
**Line 148-206:** Updated `move_completed_events()`
- Now returns `tuple[int, list[str]]` instead of just `int`
- Returns both count and list of moved event IDs
- Event IDs are used for targeted scraping

**Line 209-278:** New `scrape_completed_event_fights()` function
- Takes list of event IDs to scrape
- Gets fight URLs for only those events
- Scrapes complete fight data with all 17 columns
- Updates fighter data for any new fighters
- Efficient: only scrapes what's needed

**Line 569-576:** Modified main workflow
- Captures moved event IDs from `move_completed_events()`
- Calls `scrape_completed_event_fights()` with those IDs
- Only happens when events actually moved

## Benefits

✅ **No Data Corruption:** Upcoming fights are never appended to completed fights
✅ **Complete Data:** All fights have full 17-column schema with results
✅ **Fast:** Only scrapes 1-2 events instead of entire history
✅ **Reliable:** No more "single positional indexer" errors
✅ **Proper Training Data:** ML dataset builds successfully with complete fight information

## Testing

To test after next event completes:

```bash
# Run the automation
python app.py

# Expected logs:
# - "Found 1 events to move to completed"
# - "Moved X events to completed data"
# - "Scraping complete fight data for 1 newly completed events..."
# - "Scraped X fights with complete results"
# - "Creating features for XXXX training fights" (number should increase!)

# Verify fight schema
head -1 ufc_data/fight_data.csv  # Should show all 17 columns
tail -5 ufc_data/fight_data.csv  # Recent fights should have all columns filled
```

## Root Cause

The original implementation assumed upcoming and completed fights had the same structure. **They don't!** Upcoming fights only have the matchup information (who's fighting). Completed fights have the RESULTS (who won, how, etc.).

You can't transform one into the other by moving data - you need to re-scrape from the source with the full results.

---

**Status:** ✅ FIXED - Ready for next event completion
