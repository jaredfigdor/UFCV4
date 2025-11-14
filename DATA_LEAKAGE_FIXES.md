# Data Leakage Fixes - UFC ML Model

## Problem
Training accuracy was 96-98% while test accuracy was only 64-70%, indicating severe data leakage where the model was learning from future information.

## Root Causes Identified

### 1. **Static Career Records** (CRITICAL)
**Location**: `feature_engineering.py` - `_create_record_features()`

**Problem**: Used `fighter_w`, `fighter_l`, `fighter_d` from `fighter_data.csv` which are current cumulative career totals, not temporal snapshots.

**Example**: For a 2010 fight, the model was using a fighter's 2025 record (e.g., 25-5 instead of their 2010 record of 5-2).

**Fix**:
- Created `_calculate_record_at_time()` method that counts wins/losses/draws from historical fights BEFORE the current fight date
- Updated `_create_record_features()` to take `current_fight` and `all_fights` parameters
- Removed all usage of `fighter_w`, `fighter_l`, `fighter_d` from fighter_data.csv

### 2. **Missing Historical Context** (CRITICAL)
**Location**: `dataset_builder.py` - `_build_training_dataset()`

**Problem**: Passed `all_completed_fights=None` to `create_fight_features()`, causing temporal filtering to only use fights within the current batch instead of ALL fights in the dataset.

**Example**: When processing fight #500, it only had access to fights #1-499 from the training batch, not ALL fights from UFC history before fight #500's date.

**Fix**:
- Created `all_fights_with_dates` by merging training_fights with event dates
- Passed this to `create_fight_features()` as `all_completed_fights` parameter
- Updated `create_fight_features()` to properly use `all_completed_fights` for temporal filtering

### 3. **Advanced Matchup Features** (MEDIUM)
**Location**: `feature_engineering.py` - `_create_advanced_matchup_features()`

**Problem**: Used static `fighter_w/l/d` for experience gap and win rate calculations.

**Fix**:
- Updated to use `_calculate_record_at_time()` for temporal records
- Added `current_fight` parameter to enable temporal lookups
- Win rate gaps now calculated from historical records only

## Files Modified

1. **ufcscraper/feature_engineering.py**
   - `_create_record_features()` - Complete rewrite for temporal records
   - `_calculate_record_at_time()` - New helper method
   - `_create_advanced_matchup_features()` - Updated to use temporal records
   - `create_fight_features()` - Fixed historical_context logic

2. **ufcscraper/dataset_builder.py**
   - `_build_training_dataset()` - Fixed to pass all_completed_fights with dates

## Verification

### Temporal Filtering Test
Created `test_temporal_filtering.py` to verify:
- Records are calculated only from fights before cutoff date
- Future fights are correctly excluded
- event_date column properly propagated through the pipeline

**Test Result**: ✅ PASSED - All temporal filters working correctly

### Expected Results After Fix

**Before Fix:**
- Training accuracy: 96-98%
- Test accuracy: 64-70%
- Gap: ~30% (massive overfitting)

**After Fix (Expected):**
- Training accuracy: 60-75%
- Test accuracy: 60-70%
- Gap: <10% (healthy generalization)

## Key Lessons

1. **Never use cumulative statistics** without temporal awareness in time-series ML
2. **Always pass complete historical context** to feature engineering functions
3. **Test temporal filtering** explicitly with unit tests
4. **High training accuracy** (>90%) in sports prediction is usually a red flag for data leakage

## What We Didn't Change

These areas were already correctly implementing temporal filtering:
- `_create_historical_features()` - Already filtered by date ✅
- `_create_opponent_quality_features()` - Already filtered by date ✅
- `_create_style_features()` - Already filtered by date ✅
- `_create_elo_rating_features()` - Already filtered by date ✅
- All round statistics - Already filtered via fighter_fights ✅

## How to Run

1. Delete old cached data:
   ```bash
   rm ufc_data/training_dataset_cache.csv ufc_data/prediction_dataset_cache.csv
   rm ufc_data/training_cache_metadata.json ufc_data/prediction_cache_metadata.json
   rm ufc_data/ufc_model.pkl ufc_data/ufc_scaler.pkl ufc_data/ufc_features.pkl
   ```

2. Rebuild and retrain:
   ```bash
   python app.py --force-rebuild --retrain-model --no-web
   ```

3. Verify accuracy metrics in logs - training should be 60-75%, not 96%+

## Testing

```bash
# Run temporal filtering unit test
python test_temporal_filtering.py

# Expected output: [SUCCESS] ALL TESTS PASSED
```
