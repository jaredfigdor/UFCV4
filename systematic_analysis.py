"""Systematic deep analysis to find fundamental issues"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

data_folder = Path("ufc_data")

print("=" * 80)
print("SYSTEMATIC DEEP ANALYSIS")
print("=" * 80)

# 1. CHECK TEMPORAL RECORD CALCULATION
print("\n1. TEMPORAL RECORD CALCULATION CHECK")
print("-" * 80)

training_df = pd.read_csv(data_folder / "training_dataset_cache.csv")
training_df['event_date'] = pd.to_datetime(training_df['event_date'])

# Sample some fights and check if records make sense
sample_fights = training_df.sort_values('event_date').iloc[::1000]  # Every 1000th fight

print("Sampling fights to check record accuracy:")
for idx, fight in sample_fights.head(5).iterrows():
    f1_record = f"{fight['fighter_1_wins']:.0f}-{fight['fighter_1_losses']:.0f}"
    f2_record = f"{fight['fighter_2_wins']:.0f}-{fight['fighter_2_losses']:.0f}"
    f1_win_pct = fight.get('fighter_1_win_percentage', 0)
    f2_win_pct = fight.get('fighter_2_win_percentage', 0)

    print(f"\nFight Date: {fight['event_date'].date()}")
    print(f"  F1 Record: {f1_record} ({f1_win_pct:.1%})")
    print(f"  F2 Record: {f2_record} ({f2_win_pct:.1%})")
    print(f"  Winner: {'F1' if fight['winner'] == 1 else 'F2'}")

    # Check if record makes sense
    if f1_win_pct == 1.0 and fight['fighter_1_losses'] > 0:
        print(f"  WARNING: Win % = 100% but has losses!")
    if f2_win_pct == 1.0 and fight['fighter_2_losses'] > 0:
        print(f"  WARNING: Win % = 100% but has losses!")

# 2. CHECK FEATURE CORRELATION WITH TARGET
print("\n\n2. FEATURE CORRELATION WITH TARGET")
print("-" * 80)

# Load model to get selected features
with open(data_folder / "ufc_features.pkl", 'rb') as f:
    feature_names = pickle.load(f)

print(f"Model uses {len(feature_names)} features")

# Calculate correlation of each feature with winner
correlations = []
for feat in feature_names:
    if feat in training_df.columns:
        corr = training_df[feat].corr(training_df['winner'])
        if not pd.isna(corr):
            correlations.append({'feature': feat, 'correlation': abs(corr)})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

print("\nTop 10 features by correlation:")
print(corr_df.head(10).to_string(index=False))

print("\nBottom 10 features by correlation:")
print(corr_df.tail(10).to_string(index=False))

# Check if we have weak features diluting the model
weak_features = corr_df[corr_df['correlation'] < 0.01]
print(f"\nFeatures with correlation < 0.01: {len(weak_features)}")
if len(weak_features) > 0:
    print("WARNING: Many weak features may be diluting model signal!")

# 3. CHECK PROBABILITY DISTRIBUTION
print("\n\n3. PROBABILITY DISTRIBUTION ANALYSIS")
print("-" * 80)

predictions = pd.read_csv(data_folder / "fight_predictions.csv")

print(f"Total predictions: {len(predictions)}")
print(f"\nFighter 1 win probability distribution:")
print(predictions['fighter_1_win_probability'].describe())

print(f"\nConfidence distribution:")
print(predictions['confidence'].describe())

# Check if probabilities are too centered around 0.5
centered = predictions[(predictions['confidence'] > 0.50) & (predictions['confidence'] < 0.60)]
print(f"\nPredictions with 50-60% confidence: {len(centered)} ({len(centered)/len(predictions)*100:.1f}%)")

if len(centered) / len(predictions) > 0.5:
    print("WARNING: More than 50% of predictions are 50-60% confidence!")
    print("Model is not separating clear favorites from toss-ups!")

# 4. CHECK RECORD QUALITY DIFFERENCE
print("\n\n4. RECORD QUALITY DIFFERENCE ANALYSIS")
print("-" * 80)

prediction_df = pd.read_csv(data_folder / "prediction_dataset_cache.csv")

# Check record_quality_difference for predictions
if 'record_quality_difference' in prediction_df.columns:
    rqd = prediction_df['record_quality_difference'].dropna()
    print(f"Record quality difference stats:")
    print(rqd.describe())

    # Merge with predictions to see relationship
    merged = prediction_df[['fight_id', 'record_quality_difference']].merge(
        predictions[['fight_id', 'confidence']], on='fight_id'
    )

    if len(merged) > 0:
        corr = merged['record_quality_difference'].corr(merged['confidence'])
        print(f"\nCorrelation between record_quality_diff and confidence: {corr:.3f}")

        if abs(corr) < 0.3:
            print("WARNING: Weak correlation! Model not using record quality effectively!")

# 5. CHECK FOR FIGHTER BIAS
print("\n\n5. FIGHTER POSITION BIAS CHECK")
print("-" * 80)

# Check if model favors fighter_1 or fighter_2
f1_wins_predicted = (predictions['predicted_winner'] == 1).sum()
f2_wins_predicted = (predictions['predicted_winner'] == 0).sum()

print(f"Fighter 1 predicted to win: {f1_wins_predicted} ({f1_wins_predicted/len(predictions)*100:.1f}%)")
print(f"Fighter 2 predicted to win: {f2_wins_predicted} ({f2_wins_predicted/len(predictions)*100:.1f}%)")

if abs(f1_wins_predicted - f2_wins_predicted) / len(predictions) > 0.2:
    print("WARNING: Significant positional bias detected!")

# 6. CHECK ACTUAL FIGHT EXAMPLES
print("\n\n6. SPECIFIC FIGHT ANALYSIS")
print("-" * 80)

# Get fighter data
fighters_df = pd.read_csv(data_folder / "fighter_data.csv")
fighters_df['full_name'] = fighters_df['fighter_f_name'] + ' ' + fighters_df['fighter_l_name']

# Analyze top 3 predictions
top_preds = predictions.sort_values('confidence', ascending=False).head(3)

for idx, pred in top_preds.iterrows():
    print(f"\n{'='*80}")
    fight_data = prediction_df[prediction_df['fight_id'] == pred['fight_id']].iloc[0]

    f1_id = pred['fighter_1']
    f2_id = pred['fighter_2']

    f1 = fighters_df[fighters_df['fighter_id'] == f1_id].iloc[0]
    f2 = fighters_df[fighters_df['fighter_id'] == f2_id].iloc[0]

    f1_name = f1['full_name']
    f2_name = f2['full_name']

    print(f"Fight: {f1_name} vs {f2_name}")
    print(f"Confidence: {pred['confidence']*100:.1f}%")
    print(f"Predicted winner: {'F1' if pred['predicted_winner']==1 else 'F2'}")

    # Show key stats
    print(f"\nFighter 1 ({f1_name}):")
    print(f"  Actual Record: {f1['fighter_w']}-{f1['fighter_l']}-{f1['fighter_d']}")
    print(f"  Model Record: {fight_data['fighter_1_wins']:.0f}-{fight_data['fighter_1_losses']:.0f}")
    print(f"  Win %: {fight_data.get('fighter_1_win_percentage', 0):.1%}")

    print(f"\nFighter 2 ({f2_name}):")
    print(f"  Actual Record: {f2['fighter_w']}-{f2['fighter_l']}-{f2['fighter_d']}")
    print(f"  Model Record: {fight_data['fighter_2_wins']:.0f}-{fight_data['fighter_2_losses']:.0f}")
    print(f"  Win %: {fight_data.get('fighter_2_win_percentage', 0):.1%}")

    print(f"\nKey Features:")
    print(f"  Record Quality Diff: {fight_data.get('record_quality_difference', 0):.3f}")
    print(f"  Age Advantage: {fight_data.get('age_advantage', 0):.1f}")
    print(f"  Experience Advantage: {fight_data.get('experience_advantage', 0):.0f} fights")

# 7. FINAL DIAGNOSIS
print("\n\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nPotential Issues Found:")
print("1. Check if temporal record calculation is actually working")
print("2. Check if weak features are diluting model signal")
print("3. Check if probability calibration is compressing confidence")
print("4. Check if positional bias exists")
print("5. Check if record_quality_difference is being used effectively")
