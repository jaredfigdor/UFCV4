# UFC ML Model Optimizations

## Summary of All Optimizations Applied

### 1. **Class Imbalance Handling** ⚠️ CRITICAL
**Problem**: Training data has 63.4% fighter_1 wins vs 36.6% fighter_2 wins
- This causes the model to learn a bias toward predicting fighter_1 wins
- Inflates training accuracy without improving real performance

**Solution**: Added class weighting to all gradient boosting models
- **XGBoost**: `scale_pos_weight=1.73` (automatically calculated from class ratio)
- **LightGBM**: `class_weight='balanced'`
- **CatBoost**: `auto_class_weights='Balanced'`

**Impact**: Forces model to give equal importance to both classes during training

---

### 2. **Feature Selection Pipeline** ✅
Implemented 3-stage feature selection to remove noise and redundancy:

**Stage 1 - Remove High-Missing Features** (>50% missing):
- `fighter_2_recent_volatility`: 62.6% missing
- `fighter_1_recent_volatility`: 49.1% missing
- `fighter_2_outcome_variance`, `fighter_2_performance_consistency`, `fighter_2_finish_variance`: 46% missing
- Total: ~10-15 features removed

**Stage 2 - Remove Zero/Low Variance Features** (variance < 0.01):
- `fighter_1_draws`, `fighter_2_draws`: Always 0
- `short_notice_fight`: Always 0
- `fighter_1/2_stance_open_stance`: Near constant
- Total: 5 features removed

**Stage 3 - Remove Perfectly Correlated Features** (correlation > 0.98):
- `avg_takedown_accuracy` = `takedown_success_rate` (1.000 correlation) → DUPLICATE
- `avg_opponent_win_rate` = `strength_of_schedule` (1.000 correlation) → DUPLICATE
- `avg_strikes_per_round` = `striking_volume` (1.000 correlation) → DUPLICATE
- `submission_rate` = `submission_threat` (1.000 correlation) → DUPLICATE
- `ring_rust_penalty` ≈ `age_x_rust` (0.99 correlation)
- Total: ~10 features removed

**Result**: ~186 features → ~155-160 features (removing 15-20% redundant/noisy features)

---

### 3. **Gradient Boosting Models** (Better for Combat Sports)
Replaced Random Forest with XGBoost/LightGBM/CatBoost as primary models

**Why Gradient Boosting?**
- Better bias-variance trade-off for tabular data
- Built-in regularization (L1/L2)
- Better probability calibration (important for AUC)
- Handles feature interactions better
- More resistant to overfitting with proper tuning

**Default Model**: XGBoost (best overall for sports prediction)

---

### 4. **Hyperparameter Optimization Improvements**

**Changed Scoring Metric**: `accuracy` → `roc_auc`
- ROC-AUC optimizes probability calibration
- Better for imbalanced datasets
- More informative than raw accuracy

**Stratified 5-Fold Cross-Validation**:
- Ensures balanced class distribution in each fold
- More reliable performance estimates
- Prevents overfitting to single train/test split

**XGBoost Hyperparameter Grid**:
```python
{
    'n_estimators': [200, 300, 500],  # More trees for complex patterns
    'max_depth': [4, 6, 8],  # CONTROLLED (was 25 → massive overfitting)
    'learning_rate': [0.01, 0.05, 0.1],  # Slower learning = better generalization
    'subsample': [0.7, 0.8, 0.9],  # Row sampling for regularization
    'colsample_bytree': [0.7, 0.8, 0.9],  # Feature sampling for regularization
    'min_child_weight': [3, 5, 7],  # Minimum samples per leaf
    'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
    'reg_lambda': [1, 2, 5],  # L2 regularization
}
```

**Key Changes**:
- `max_depth`: 15-25 → 4-8 (prevents memorization)
- Added `reg_alpha` and `reg_lambda` for regularization
- Added `subsample` and `colsample_bytree` for variance reduction

---

### 5. **Data Quality Improvements**

**Conservative Imputation** (only essential columns):
- Physical stats: filled with median (height, weight, reach, age)
- Record stats: filled with 0 (wins, losses for debut fighters)
- Binary features: filled with mode (gender, title_fight)
- Everything else: left to model to handle (gradient boosting handles missing values natively)

**Quality Filtering**:
- Training: Must have 80% of features + all critical features
- Prediction: Must have 60% of features (more lenient)
- Removes low-quality fights with too many missing values

---

## Expected Performance Impact

### Before Optimizations:
- Training Accuracy: **96-99%** ❌ (overfitting)
- Test Accuracy: **62-66%**
- Gap: **30-35%** (severe overfitting)

### After Optimizations:
- Training Accuracy: **68-75%** ✅ (realistic)
- Test Accuracy: **64-69%** ✅ (should improve)
- ROC-AUC: **0.70-0.75** ✅ (better probability calibration)
- Gap: **4-6%** ✅ (healthy generalization)

---

## What Each Optimization Fixes

1. **Class Weighting** → Removes fighter_1 bias (63/37 → 50/50 importance)
2. **Feature Selection** → Removes noise, reduces overfitting, faster training
3. **Gradient Boosting** → Better for complex interactions, built-in regularization
4. **ROC-AUC Scoring** → Optimizes probability calibration (important for betting odds)
5. **Controlled max_depth** → Prevents memorization (depth 25 → 4-8)
6. **L1/L2 Regularization** → Penalizes complex models, favors simplicity
7. **Subsampling** → Row/column sampling reduces variance
8. **Stratified CV** → Better performance estimates, prevents lucky splits

---

## How to Verify Improvements

After training with `python app.py --retrain-model --no-web`, check for:

### ✅ Good Signs:
- Training accuracy 68-75% (not 96%+)
- Test accuracy 64-69%
- Train-test gap < 8%
- ROC-AUC > 0.70
- Class balance warnings in logs showing scale_pos_weight calculation

### ❌ Bad Signs:
- Training accuracy > 85% (still overfitting)
- Train-test gap > 15% (poor generalization)
- ROC-AUC < 0.65 (poor probability calibration)

---

## Additional Recommendations (Not Implemented)

### Consider if still seeing issues:

1. **SMOTE for Data Augmentation**:
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
   ```
   - Creates synthetic minority class samples
   - Can improve performance on imbalanced data
   - May increase training time

2. **Ensemble Averaging**:
   - Train XGBoost, LightGBM, and CatBoost
   - Average their predictions
   - Often better than single model

3. **Feature Importance-Based Selection**:
   - After first training, remove bottom 20% importance features
   - Retrain on most important features only
   - Can reduce overfitting further

4. **Early Stopping**:
   - Stop training when validation performance plateaus
   - Prevents overfitting to training data
   - Already partially handled by GridSearchCV

---

## Files Modified

1. `ufcscraper/ml_predictor.py`:
   - Added XGBoost/LightGBM/CatBoost support
   - Implemented `_select_features()` method
   - Added class weighting
   - Changed to ROC-AUC optimization
   - Implemented stratified K-fold CV

2. `app.py`:
   - Changed default model from Random Forest to XGBoost
   - Fixed weight class mapping bug

3. `pyproject.toml`:
   - Added xgboost, lightgbm, catboost dependencies

---

## Next Steps

1. **Delete old model**: `rm ufc_data/ufc_model.pkl ufc_data/ufc_scaler.pkl ufc_data/ufc_features.pkl`
2. **Retrain**: `python app.py --retrain-model --no-web`
3. **Verify**: Check training logs for realistic accuracy (68-75%)
4. **Compare**: Old predictions vs new predictions - should be more balanced
5. **Monitor**: Track performance over multiple events to validate improvements

---

## Key Metrics to Watch

| Metric | Before | After (Target) | Status |
|--------|--------|----------------|--------|
| Training Accuracy | 96-99% | 68-75% | ⏳ Pending |
| Test Accuracy | 62-66% | 64-69% | ⏳ Pending |
| Train-Test Gap | 30-35% | 4-6% | ⏳ Pending |
| ROC-AUC | ~0.65 | 0.70-0.75 | ⏳ Pending |
| Features Used | 186 | 155-160 | ⏳ Pending |
| Class Balance | 63/37 | Weighted 50/50 | ⏳ Pending |

