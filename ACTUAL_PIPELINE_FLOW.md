# Actual Pipeline Flow - How Events Get Processed

## The Complete Flow (When You Run `python app.py`)

### Step 1: Check Data Freshness
- Checks if data files are older than 12 hours
- Determines if scraping is needed

### Step 2: Scrape Data (If Needed)

**Scrape Upcoming:**
- Scrapes upcoming events from UFCStats.com
- Events dated >= today → `upcoming_event_data.csv`
- Events dated < today → SKIPPED (will be handled by completed scraper)
- Scrapes upcoming fights → `upcoming_fight_data.csv` (6 columns, no results)

**Scrape Completed:**
- Scrapes completed events from UFCStats.com
- Events dated >= today → SKIPPED (will be handled by upcoming scraper)
- Events dated < today → `event_data.csv`
- Scrapes completed fights → `fight_data.csv` (17 columns, WITH results)
- **Removes duplicates automatically** via `remove_duplicates_from_file()`

### Step 3: Move Completed Events (Data Consistency)

The DataConsistencyManager looks for events in `upcoming_event_data.csv` that are dated < today and:
1. **Checks if event is already in `event_data.csv`** (from Step 2 scraper)
   - If YES: Logs "already in completed" and skips adding it ✅
   - If NO: Appends to `event_data.csv`
2. **Deletes** the event from `upcoming_event_data.csv`
3. **Deletes** associated fights from `upcoming_fight_data.csv`
   - Does NOT try to move them (schema mismatch!)
   - Completed fights were already scraped in Step 2 with full results

### Step 4: Build Datasets
- Training dataset uses fights from `fight_data.csv`
- Prediction dataset uses fights from `upcoming_fight_data.csv`

---

## Scenario 1: Event Was Never in Upcoming

**Example: UFC 330 is announced and happens same week**

### Dec 21: Event Announced
- Not scraped yet (data is current)

### Dec 22: Event Happens
- Still not scraped (within 12-hour window)

### Dec 23: You Run `python app.py`

**Step 2 - Scrape Completed:**
- Scraper fetches events from UFCStats.com
- Sees UFC 330 dated Dec 22 (yesterday)
- Adds event to `event_data.csv` ✅
- Scrapes all 12 fights with full results → `fight_data.csv` ✅

**Step 3 - Move Completed Events:**
- Checks `upcoming_event_data.csv` for past events
- UFC 330 is NOT there (was never added to upcoming)
- Nothing to move ✅

**Result:**
- UFC 330 in `event_data.csv` ✅
- 12 fights in `fight_data.csv` with full 17-column results ✅
- Training data includes UFC 330 ✅

---

## Scenario 2: Event WAS in Upcoming

**Example: UFC 324 is currently in upcoming (dated Jan 24, 2026)**

### Jan 25, 2026: You Run `python app.py`

**Step 2 - Scrape Upcoming:**
- Scraper fetches upcoming events
- UFC 324 dated Jan 24 is in PAST
- Upcoming scraper SKIPS it (line 103 event_scraper.py) ✅
- UFC 324 stays in `upcoming_event_data.csv` from previous scrape

**Step 2 - Scrape Completed:**
- Scraper fetches completed events
- Sees UFC 324 dated Jan 24 (yesterday)
- Adds event to `event_data.csv` ✅
- Scrapes all 13 fights with full results → `fight_data.csv` ✅
- **UFC 324 is now in BOTH upcoming and completed!**

**Step 3 - Move Completed Events:**
- Checks `upcoming_event_data.csv` for past events
- Finds UFC 324 (dated Jan 24 < today)
- Checks if UFC 324 is already in `event_data.csv` ✅ (from Step 2!)
- **Logs "already in completed" and SKIPS adding it** ✅
- Deletes UFC 324 from `upcoming_event_data.csv` ✅
- Deletes UFC 324 fights from `upcoming_fight_data.csv` ✅

**Result:**
- UFC 324 in `event_data.csv` (NO DUPLICATE) ✅
- 13 fights in `fight_data.csv` with full 17-column results ✅
- UFC 324 removed from upcoming files ✅
- Training data includes UFC 324 ✅

---

## Key Points

### ✅ What Works:
1. **Step 2 scraper handles everything** - Gets events and fights with full results
2. **No schema mismatch** - Upcoming fights (6 cols) never appended to completed fights (17 cols)
3. **No duplicates** - Step 3 checks before adding, Step 2 removes duplicates
4. **Works for both scenarios** - Whether event was in upcoming or not

### ✅ Why the Cleanup Scripts Were Needed (One-Time Fix):
- UFC 323 got corrupted because the OLD CODE tried to move upcoming fights to completed
- That bug is NOW FIXED
- Future events will work correctly without cleanup scripts

### ✅ What Happens Next Week:
When the next event completes, you just run `python app.py` and:
1. Step 2 scrapes it as completed (with full fight results)
2. Step 3 removes it from upcoming (if it was there)
3. Training data grows automatically
4. **NO CLEANUP SCRIPTS NEEDED!**

---

## The Fixes Applied

### 1. Event Scraper Date Logic (event_scraper.py)
- Line 98: Completed scraper skips events dated >= today
- Line 103: Upcoming scraper skips events dated < today
- **Result:** Events go to correct file based on date

### 2. Data Consistency Manager (data_consistency.py)
- Line 103: Move events to upcoming if dated >= today
- Line 114: Move events to completed if dated < today
- **Line 154-178: CHECK FOR DUPLICATES before appending** ← NEW FIX
- Line 180-184: Comments explaining why we DON'T move fights
- **Result:** No duplicate events, no schema mismatches

### 3. App.py Workflow
- Removed broken Step 3.5 (scrape_completed_event_fights)
- Step 2 already handles everything
- **Result:** Clean, simple workflow

---

## Trust the Pipeline! 🎯

The pipeline is NOW PROPERLY IMPLEMENTED. Next week when an event completes:
- Just run `python app.py`
- No cleanup scripts needed
- Event and fights automatically processed
- Training data grows correctly

**The cleanup scripts were only needed to fix the historical corruption from the old buggy code!**
