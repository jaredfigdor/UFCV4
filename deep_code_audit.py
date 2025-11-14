"""Deep audit of all code for issues"""
import pandas as pd
import numpy as np
from pathlib import Path

data_folder = Path("ufc_data")

print("=" * 80)
print("DEEP CODE AUDIT - FINDING ALL ISSUES")
print("=" * 80)

# ISSUE 1: Check all columns for non-numeric types
print("\n1. CHECKING ALL FEATURES FOR NON-NUMERIC TYPES")
print("-" * 80)

training_df = pd.read_csv(data_folder / "training_dataset_cache.csv")
prediction_df = pd.read_csv(data_folder / "prediction_dataset_cache.csv")

metadata_cols = ['fight_id', 'fighter_1', 'fighter_2', 'event_id', 'winner', 'event_date', 'dataset_type', 'created_at']
feature_cols = [col for col in training_df.columns if col not in metadata_cols]

print(f"Checking {len(feature_cols)} features...")

non_numeric_features = []
for col in feature_cols:
    if col in training_df.columns:
        if training_df[col].dtype == 'object':
            non_numeric_features.append(col)
            print(f"  ISSUE: {col} is type 'object' (string)")
            print(f"    Sample values: {training_df[col].dropna().unique()[:5]}")

if len(non_numeric_features) == 0:
    print("  GOOD: All features are numeric")
else:
    print(f"\n  CRITICAL: {len(non_numeric_features)} non-numeric features found!")

# ISSUE 2: Check for inf or extreme values
print("\n\n2. CHECKING FOR INFINITY OR EXTREME VALUES")
print("-" * 80)

extreme_issues = []
for col in feature_cols:
    if col in training_df.columns and training_df[col].dtype in ['float64', 'int64']:
        # Check for inf
        inf_count = np.isinf(training_df[col]).sum()
        if inf_count > 0:
            extreme_issues.append(f"{col}: {inf_count} infinite values")

        # Check for extreme values (>1000 or <-1000 for most features)
        if 'height' not in col and 'weight' not in col and 'reach' not in col and 'days' not in col:
            extreme_vals = training_df[col].abs() > 1000
            if extreme_vals.sum() > 0:
                max_val = training_df[col].abs().max()
                extreme_issues.append(f"{col}: extreme value {max_val:.1f}")

if len(extreme_issues) > 0:
    print(f"  ISSUES FOUND:")
    for issue in extreme_issues[:10]:
        print(f"    {issue}")
else:
    print("  GOOD: No extreme values found")

# ISSUE 3: Check if features have zero variance
print("\n\n3. CHECKING FOR ZERO VARIANCE FEATURES")
print("-" * 80)

zero_var = []
for col in feature_cols:
    if col in training_df.columns and training_df[col].dtype in ['float64', 'int64']:
        var = training_df[col].var()
        if pd.notna(var) and var == 0:
            zero_var.append(col)

if len(zero_var) > 0:
    print(f"  ISSUE: {len(zero_var)} features have zero variance:")
    for col in zero_var:
        print(f"    {col}")
else:
    print("  GOOD: No zero variance features")

# ISSUE 4: Check if NaN percentage is too high for key features
print("\n\n4. CHECKING MISSING DATA IN KEY PREDICTIVE FEATURES")
print("-" * 80)

key_features = [
    'record_quality_difference',
    'fighter_1_win_percentage', 'fighter_2_win_percentage',
    'fighter_1_wins', 'fighter_1_losses',
    'fighter_2_wins', 'fighter_2_losses',
    'age_advantage',
    'fighter_1_last_3_win_rate', 'fighter_2_last_3_win_rate'
]

high_missing = []
for col in key_features:
    if col in prediction_df.columns:
        missing_pct = prediction_df[col].isna().sum() / len(prediction_df) * 100
        if missing_pct > 10:
            high_missing.append(f"{col}: {missing_pct:.1f}% missing in predictions")

if len(high_missing) > 0:
    print(f"  ISSUES FOUND:")
    for issue in high_missing:
        print(f"    {issue}")
else:
    print("  GOOD: Key features have low missing rates")

# ISSUE 5: Check if feature scaling is needed but not applied
print("\n\n5. CHECKING FEATURE SCALE RANGES")
print("-" * 80)

scale_issues = []
for col in feature_cols[:20]:  # Check first 20
    if col in training_df.columns and training_df[col].dtype in ['float64', 'int64']:
        col_min = training_df[col].min()
        col_max = training_df[col].max()
        if pd.notna(col_min) and pd.notna(col_max):
            scale_range = col_max - col_min
            # Check if scale is vastly different (some features 0-1, others 0-1000)
            if scale_range > 100:
                scale_issues.append(f"{col}: range {col_min:.1f} to {col_max:.1f}")

if len(scale_issues) > 0:
    print(f"  INFO: Some features have large scales (this is OK if using scaler):")
    for issue in scale_issues[:5]:
        print(f"    {issue}")
else:
    print("  GOOD: Feature scales look reasonable")

# ISSUE 6: Check fighter_1 vs fighter_2 bias in training
print("\n\n6. CHECKING FOR FIGHTER POSITION BIAS IN TRAINING")
print("-" * 80)

winner_dist = training_df['winner'].value_counts()
print(f"  Winner distribution in training:")
print(f"    Fighter 1 (winner=1): {winner_dist.get(1, 0)} ({winner_dist.get(1, 0)/len(training_df)*100:.1f}%)")
print(f"    Fighter 2 (winner=0): {winner_dist.get(0, 0)} ({winner_dist.get(0, 0)/len(training_df)*100:.1f}%)")

bias_pct = abs(winner_dist.get(1, 0) - winner_dist.get(0, 0)) / len(training_df) * 100
if bias_pct > 15:
    print(f"    WARNING: {bias_pct:.1f}% bias detected! This can affect confidence!")
else:
    print(f"    GOOD: Bias is only {bias_pct:.1f}%")

# ISSUE 7: Check if any features are perfectly correlated (redundant)
print("\n\n7. CHECKING FOR PERFECTLY CORRELATED FEATURES")
print("-" * 80)

numeric_features = [col for col in feature_cols if col in training_df.columns and training_df[col].dtype in ['float64', 'int64']]
if len(numeric_features) > 50:
    numeric_features = numeric_features[:50]  # Sample to speed up

corr_matrix = training_df[numeric_features].corr().abs()
perfect_corr = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.99:
            perfect_corr.append(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

if len(perfect_corr) > 0:
    print(f"  ISSUE: {len(perfect_corr)} pairs of nearly identical features:")
    for pair in perfect_corr[:5]:
        print(f"    {pair}")
else:
    print("  GOOD: No perfectly correlated features")

# ISSUE 8: Check record_quality_difference calculation
print("\n\n8. CHECKING RECORD_QUALITY_DIFFERENCE CALCULATION")
print("-" * 80)

if 'record_quality_difference' in prediction_df.columns:
    rqd = prediction_df['record_quality_difference'].dropna()
    if len(rqd) > 0:
        print(f"  Record quality difference stats:")
        print(f"    Mean: {rqd.mean():.3f}")
        print(f"    Std: {rqd.std():.3f}")
        print(f"    Min: {rqd.min():.3f}")
        print(f"    Max: {rqd.max():.3f}")

        # Check if it's all near zero (would indicate broken calculation)
        near_zero = (rqd.abs() < 0.01).sum() / len(rqd) * 100
        if near_zero > 50:
            print(f"    WARNING: {near_zero:.1f}% of values near zero!")
        else:
            print(f"    GOOD: Only {near_zero:.1f}% near zero")

# ISSUE 9: Check if interaction features are being created
print("\n\n9. CHECKING IF INTERACTION FEATURES EXIST")
print("-" * 80)

interaction_features = [col for col in feature_cols if 'interaction' in col or 'quality' in col and 'record' in col]
print(f"  Found {len(interaction_features)} interaction features:")
for feat in interaction_features[:10]:
    print(f"    {feat}")

if len(interaction_features) < 5:
    print("  WARNING: Expected more interaction features!")

# ISSUE 10: Check temporal split
print("\n\n10. CHECKING TEMPORAL SPLIT DATES")
print("-" * 80)

training_df['event_date'] = pd.to_datetime(training_df['event_date'])
sorted_dates = training_df.sort_values('event_date')['event_date']

split_idx = int(len(sorted_dates) * 0.8)
train_end = sorted_dates.iloc[split_idx]
test_start = sorted_dates.iloc[split_idx]

print(f"  Training ends: {train_end.date()}")
print(f"  Testing starts: {test_start.date()}")

overlap_fights = training_df[training_df['event_date'] == train_end]
print(f"  Fights on split date: {len(overlap_fights)}")

if train_end == test_start:
    print("  WARNING: Train and test share the same date! Could cause minor leakage.")
else:
    print("  GOOD: Clean temporal split")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
