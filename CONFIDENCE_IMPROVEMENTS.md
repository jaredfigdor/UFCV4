# Confidence Improvement Plan

## Current Situation
- Confidence range: 50.6% - 62.1% (avg 56.1%)
- Test AUC: ~0.765
- Issue: 30%+ missing data in fighter_2 features

## Identified Problems

### 1. High Missing Data (30%+ for fighter_2 features)
**Root Cause:** Fighters with <2-3 UFC fights don't have enough history
**Impact:** Model can't use 30% of feature data, reducing confidence

### 2. Zero-Importance Features
Features not contributing:
- fighter_2_decision_rate, submission_rate (0.0 importance)
- quality_wins, bad_losses (0.0 importance)
- southpaw_advantage, height/reach advantages (0.0 importance)

### 3. Missing Interaction Features
Top correlated features (record_quality_difference, age_advantage) not interacting

## Improvement Strategy

### Phase 1: Better Missing Data Handling (QUICK WIN)
1. **Smarter defaults for debut fighters:**
   - Use pre-UFC record stats if available
   - Default to weight class averages instead of 0
   - Interpolate from similar fighters

2. **Fill missing fighter_2 features:**
   - Currently 30% missing → reduce to <10%
   - Will immediately improve model confidence

### Phase 2: Add High-Value Interaction Features (QUICK WIN)
1. **Create polynomial interactions:**
   ```python
   record_quality_diff × age_advantage
   wins × win_percentage
   late_finish_rate × age_vs_prime
   ```

2. **Add momentum indicators:**
   ```python
   recent_form_trend (last 3 fights weighted)
   streak_pressure (regression to mean risk)
   ```

### Phase 3: Hyperparameter Tuning (MODERATE WIN)
1. **Reduce regularization** slightly (currently very aggressive):
   - reg_alpha: [1, 3] → [0.5, 2]
   - reg_lambda: [8, 12] → [5, 10]

2. **Increase tree depth** slightly:
   - max_depth: [3, 4] → [4, 5]

3. **More estimators:**
   - n_estimators: [150, 200] → [200, 300]

### Phase 4: Probability Calibration (ADVANCED WIN)
1. **Apply isotonic regression** to calibrate probabilities
2. **Use CalibratedClassifierCV** wrapper
3. Will make confidence scores more accurate

## Expected Impact

### Conservative Estimate:
- Missing data fix: +3-5% confidence
- Interaction features: +2-4% confidence
- Hyperparameter tuning: +1-3% confidence
- **Total: 56% → 62-68% avg confidence**

### Optimistic Estimate:
- With all improvements: **65-72% avg confidence**
- Top predictions: **75-80% confidence**

## Implementation Order (Priority)

1. ✅ **Fix missing data** (30 min, biggest impact)
2. ✅ **Add interaction features** (20 min, high ROI)
3. ✅ **Tune hyperparameters** (40 min with grid search)
4. ⏳ **Probability calibration** (optional, if needed)

## Safety Checks
- ✅ Maintain temporal split (no leakage)
- ✅ Validate on recent test data
- ✅ Check for overfitting (train-test gap <10%)
