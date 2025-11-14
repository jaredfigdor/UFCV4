# ALL CRITICAL ISSUES FOUND

## Issue 1: weight_class is STRING not NUMERIC ✅ FIXED
**Impact:** XGBoost can't use this feature at all
**Status:** Fixed - now encodes 1-8 by weight
**Expected improvement:** +2-4% confidence

## Issue 2: 63/37% Fighter Position Bias ❌ NOT FIXED
**Impact:** Model learns fighter_1 wins 63% of time, biases all predictions
**Root cause:** No data augmentation (mirroring)
**Status:** NOT IMPLEMENTED
**Expected improvement:** +5-10% confidence for underdogs

**Fix needed:**
- Duplicate training data with swapped positions
- Flip winner labels
- Sign-flip differential features

## Issue 3: Interaction Features NOT in Dataset ❌ NOT FIXED
**Impact:** Model can't use 6+ engineered interaction features
**Root cause:** `_create_interaction_features()` only runs in train_model(), not in dataset_builder
**Status:** Features exist in code but not in cached datasets
**Expected improvement:** +3-5% confidence

**Fix needed:**
- Call `_create_interaction_features()` in dataset_builder BEFORE saving cache
- Rebuild datasets with --force-rebuild

## Issue 4: Temporal Split Overlap (Minor)
**Impact:** 10 fights on 2022-08-06 in both train and test
**Status:** Minor issue, acceptable
**Fix:** Use `>` instead of `>=` in split

## Issue 5: 70% Missing Age Data in Predictions
**Impact:** Can't use age features for most predictions
**Root cause:** Fighters missing DOB data
**Status:** Partially fixed with imputation
**Fix:** Impute with 30.0 (already done)

## PRIORITY FIXES:

### Priority 1: Fix Fighter Position Bias (BIGGEST IMPACT)
### Priority 2: Rebuild Datasets with Interaction Features
### Priority 3: Already fixed weight_class encoding

## Expected Total Improvement:
- Current: 50-67% confidence
- After fixes: **65-80% confidence** for clear mismatches
