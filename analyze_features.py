"""Analyze feature importance and quality"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

data_folder = Path("ufc_data")

# Load model and get feature importance
print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

model_file = data_folder / "ufc_model.pkl"
if model_file.exists():
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    features_file = data_folder / "ufc_features.pkl"
    if features_file.exists():
        with open(features_file, 'rb') as f:
            feature_names = pickle.load(f)

        # Get feature importances
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 20 Most Important Features:")
        print(feature_importance_df.head(20).to_string(index=False))

        print("\n\nBottom 10 Least Important Features:")
        print(feature_importance_df.tail(10).to_string(index=False))

# Analyze feature quality in training data
print("\n" + "=" * 60)
print("FEATURE QUALITY ANALYSIS")
print("=" * 60)

training_df = pd.read_csv(data_folder / "training_dataset_cache.csv")

# Exclude metadata columns
metadata_cols = ['fight_id', 'fighter_1', 'fighter_2', 'event_id', 'winner', 'event_date', 'dataset_type', 'created_at']
feature_cols = [col for col in training_df.columns if col not in metadata_cols]

print(f"\nTotal features: {len(feature_cols)}")

# Check for missing values
missing_stats = []
for col in feature_cols:
    missing_pct = training_df[col].isna().sum() / len(training_df) * 100
    if missing_pct > 0:
        missing_stats.append({
            'feature': col,
            'missing_pct': missing_pct,
            'missing_count': training_df[col].isna().sum()
        })

if missing_stats:
    missing_df = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)
    print(f"\nFeatures with missing values: {len(missing_df)}")
    print("\nTop 15 features with most missing data:")
    print(missing_df.head(15).to_string(index=False))

# Check for zero variance features
print("\n" + "=" * 60)
print("FEATURE VARIANCE ANALYSIS")
print("=" * 60)

low_variance = []
for col in feature_cols:
    if training_df[col].dtype in ['float64', 'int64']:
        var = training_df[col].var()
        if var < 0.01:
            low_variance.append({'feature': col, 'variance': var})

if low_variance:
    print(f"\nLow variance features (<0.01): {len(low_variance)}")
    for item in low_variance[:10]:
        print(f"  {item['feature']}: {item['variance']:.6f}")

# Analyze correlation with target
print("\n" + "=" * 60)
print("TARGET CORRELATION ANALYSIS")
print("=" * 60)

correlations = []
for col in feature_cols:
    if training_df[col].dtype in ['float64', 'int64'] and col != 'winner':
        corr = training_df[col].corr(training_df['winner'])
        if not pd.isna(corr):
            correlations.append({'feature': col, 'correlation': abs(corr)})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
print("\nTop 15 features most correlated with winner:")
print(corr_df.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("""
To improve model confidence (while maintaining leak-free design):

1. REDUCE MISSING DATA:
   - Improve historical feature calculation for fighters with sparse data
   - Better handling of debut fighters and early-career fights

2. ADD PREDICTIVE FEATURES:
   - Recent form trends (last 3 fights weighted heavily)
   - Head-to-head style matchups (striker vs grappler)
   - Fighter momentum and streak psychology
   - Betting odds integration (if available)

3. IMPROVE EXISTING FEATURES:
   - Better Elo implementation with decay
   - More sophisticated opponent quality metrics
   - Ring rust with non-linear penalty

4. FEATURE ENGINEERING:
   - Create interaction features between top predictors
   - Polynomial features for non-linear relationships

5. MODEL TUNING:
   - Adjust learning rate and max_depth
   - Try ensemble of multiple models
   - Calibrate probabilities with Platt scaling
""")
