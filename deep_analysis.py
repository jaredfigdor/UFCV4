"""Deep analysis of feature quality and data issues"""
import pandas as pd
import numpy as np
from pathlib import Path

data_folder = Path("ufc_data")

print("=" * 80)
print("DEEP FEATURE QUALITY ANALYSIS")
print("=" * 80)

# Load datasets
training_df = pd.read_csv(data_folder / "training_dataset_cache.csv")
prediction_df = pd.read_csv(data_folder / "prediction_dataset_cache.csv")
predictions = pd.read_csv(data_folder / "fight_predictions_summary.csv")

print(f"\nTraining data: {len(training_df)} fights")
print(f"Prediction data: {len(prediction_df)} fights")

# Analyze a clear mismatch example
print("\n" + "=" * 80)
print("ANALYZING SPECIFIC FIGHTS")
print("=" * 80)

# Get top predicted fight
top_pred = predictions.sort_values('confidence', ascending=False).iloc[0]
print(f"\nTop Prediction: {top_pred['fighter_1_name']} vs {top_pred['fighter_2_name']}")
print(f"Confidence: {top_pred['confidence']*100:.1f}%")
print(f"Winner: {top_pred['predicted_winner_name']}")

# Find this fight in prediction dataset
fight_data = prediction_df[prediction_df['fight_id'] == top_pred['fight_id']].iloc[0]

# Check key features
print("\n" + "-" * 80)
print("KEY FEATURES FOR THIS FIGHT:")
print("-" * 80)

key_features = [
    'record_quality_difference',
    'age_advantage',
    'fighter_1_wins', 'fighter_1_losses',
    'fighter_2_wins', 'fighter_2_losses',
    'fighter_1_last_3_win_rate', 'fighter_2_last_3_win_rate',
    'fighter_1_late_finish_rate', 'fighter_2_late_finish_rate',
    'experience_advantage',
    'fighter_1_win_percentage', 'fighter_2_win_percentage'
]

for feat in key_features:
    if feat in fight_data:
        val = fight_data[feat]
        if pd.isna(val):
            print(f"{feat:35s} = MISSING (NaN)")
        else:
            print(f"{feat:35s} = {val:.3f}")

# Check missing data in prediction dataset
print("\n" + "=" * 80)
print("MISSING DATA IN PREDICTION DATASET")
print("=" * 80)

metadata_cols = ['fight_id', 'fighter_1', 'fighter_2', 'event_id', 'winner', 'event_date', 'dataset_type', 'created_at']
feature_cols = [col for col in prediction_df.columns if col not in metadata_cols]

missing_analysis = []
for col in feature_cols:
    missing_count = prediction_df[col].isna().sum()
    if missing_count > 0:
        missing_pct = (missing_count / len(prediction_df)) * 100
        missing_analysis.append({
            'feature': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })

missing_df = pd.DataFrame(missing_analysis).sort_values('missing_pct', ascending=False)
print(f"\nFeatures with missing data: {len(missing_df)}/{len(feature_cols)}")
print("\nTop 20 features with most missing data in predictions:")
print(missing_df.head(20).to_string(index=False))

# Compare fighter stats for top prediction
print("\n" + "=" * 80)
print("FIGHTER COMPARISON (TOP PREDICTION)")
print("=" * 80)

fighters_df = pd.read_csv(data_folder / "fighter_data.csv")
f1_id = fight_data['fighter_1']
f2_id = fight_data['fighter_2']

f1_data = fighters_df[fighters_df['fighter_id'] == f1_id].iloc[0]
f2_data = fighters_df[fighters_df['fighter_id'] == f2_id].iloc[0]

f1_name = f1_data['fighter_f_name'] + ' ' + f1_data['fighter_l_name']
f2_name = f2_data['fighter_f_name'] + ' ' + f2_data['fighter_l_name']

print(f"\nFighter 1: {f1_name}")
print(f"  Record: {f1_data['fighter_w']}-{f1_data['fighter_l']}-{f1_data['fighter_d']}")
print(f"  Height: {f1_data['fighter_height_cm']:.1f} cm")
print(f"  Reach: {f1_data['fighter_reach_cm']:.1f} cm")

print(f"\nFighter 2: {f2_name}")
print(f"  Record: {f2_data['fighter_w']}-{f2_data['fighter_l']}-{f2_data['fighter_d']}")
print(f"  Height: {f2_data['fighter_height_cm']:.1f} cm")
print(f"  Reach: {f2_data['fighter_reach_cm']:.1f} cm")

# Check if this is actually a clear mismatch
f1_win_pct = f1_data['fighter_w'] / (f1_data['fighter_w'] + f1_data['fighter_l'] + f1_data['fighter_d'])
f2_win_pct = f2_data['fighter_w'] / (f2_data['fighter_w'] + f2_data['fighter_l'] + f2_data['fighter_d'])

print(f"\nWin Percentages:")
print(f"  Fighter 1: {f1_win_pct*100:.1f}%")
print(f"  Fighter 2: {f2_win_pct*100:.1f}%")

# Check for recent fight history
print("\n" + "=" * 80)
print("CHECKING RECENT FIGHT HISTORY FEATURES")
print("=" * 80)

recent_features = [col for col in feature_cols if 'last_' in col or 'recent' in col or 'momentum' in col]
print(f"\nFound {len(recent_features)} recent/momentum features")

for feat in recent_features[:10]:
    f1_feat = feat if 'fighter_1' in feat else feat.replace('fighter_2', 'fighter_1')
    f2_feat = feat.replace('fighter_1', 'fighter_2')

    if f1_feat in fight_data and f2_feat in fight_data:
        f1_val = fight_data[f1_feat]
        f2_val = fight_data[f2_feat]

        if pd.isna(f1_val) or pd.isna(f2_val):
            print(f"{feat:40s} F1: {'MISSING' if pd.isna(f1_val) else f'{f1_val:.3f}':8s}  F2: {'MISSING' if pd.isna(f2_val) else f'{f2_val:.3f}':8s}")

# Check training data quality
print("\n" + "=" * 80)
print("TRAINING DATA QUALITY CHECK")
print("=" * 80)

train_missing = training_df[feature_cols].isna().sum().sum()
train_total = len(training_df) * len(feature_cols)
train_complete_pct = ((train_total - train_missing) / train_total) * 100

print(f"\nTraining data completeness: {train_complete_pct:.1f}%")
print(f"Total missing values: {train_missing:,} / {train_total:,}")

# Identify the real issue
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

high_missing = missing_df[missing_df['missing_pct'] > 20]
print(f"\n❌ CRITICAL: {len(high_missing)} features have >20% missing data in predictions")
print("This severely limits model confidence!\n")

print("Likely causes:")
print("1. Fighters with few UFC fights lack historical statistics")
print("2. Recent form features missing for fighters coming back from layoff")
print("3. Style matchup features not calculated for debut fighters")
print("\nRecommended fixes:")
print("1. Better default values for missing features (use division averages)")
print("2. Interpolate from pre-UFC records when available")
print("3. Add 'experience level' feature to account for missing data")
print("4. Increase feature importance for features with low missing rates")
