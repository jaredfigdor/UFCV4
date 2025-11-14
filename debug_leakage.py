"""
Debug script to identify data leakage sources.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("[DEBUG] Loading training dataset...")
df = pd.read_csv("ufc_data/training_dataset_cache.csv")

print(f"[DEBUG] Dataset shape: {df.shape}")
print(f"[DEBUG] Dataset columns: {len(df.columns)}")

# Prepare data
print("\n[DEBUG] Preparing features...")

# Identify feature columns
metadata_cols = ['fight_id', 'event_date', 'fighter_1', 'fighter_2', 'event_id',
                 'dataset_type', 'created_at']
target_col = 'winner'

feature_cols = [c for c in df.columns if c not in metadata_cols + [target_col]]
print(f"[DEBUG] Number of features: {len(feature_cols)}")

# Prepare X and y
X = df[feature_cols].copy()
y = df[target_col].copy()

# Select only numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"[DEBUG] Numeric features: {len(numeric_cols)}")

# Use only numeric features
X = X[numeric_cols].copy()

# Handle missing values
X = X.fillna(X.median())

print(f"[DEBUG] X shape: {X.shape}")
print(f"[DEBUG] y shape: {y.shape}")
print(f"[DEBUG] y distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[DEBUG] Training set: {X_train.shape}")
print(f"[DEBUG] Test set: {X_test.shape}")

# Train simple model
print("\n[DEBUG] Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\n[RESULT] Training accuracy: {train_acc:.3f}")
print(f"[RESULT] Test accuracy: {test_acc:.3f}")
print(f"[RESULT] Overfitting gap: {train_acc - test_acc:.3f}")

if train_acc > 0.85:
    print("\n[WARNING] Training accuracy > 85% - likely data leakage!")
    print("[DEBUG] Analyzing feature importance...")

    # Get feature importance
    importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n[DEBUG] Top 20 most important features:")
    for idx, row in importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Check for suspiciously perfect features
    print("\n[DEBUG] Checking for perfect predictors...")
    for feat in importance.head(10)['feature']:
        # Check correlation with target
        if feat in numeric_cols and feat in X.columns:
            feat_data = X[feat].fillna(0)
            corr = abs(feat_data.corr(y))
            if not np.isnan(corr) and corr > 0.5:
                print(f"  [SUSPECT] {feat}: correlation = {corr:.3f}")

                # Show value distribution by winner
                print(f"    Winners (1): mean={feat_data.loc[y==1].mean():.2f}, std={feat_data.loc[y==1].std():.2f}")
                print(f"    Losers (0): mean={feat_data.loc[y==0].mean():.2f}, std={feat_data.loc[y==0].std():.2f}")
else:
    print("\n[SUCCESS] Training accuracy is reasonable!")
    print("[SUCCESS] No obvious data leakage detected.")

print("\n[DEBUG] Checking for leaked outcome information...")
# Check if any features are derived from fight outcomes
outcome_keywords = ['winner', 'result', 'finish', 'scores']
suspect_features = [f for f in feature_cols if any(kw in f.lower() for kw in outcome_keywords)]

if suspect_features:
    print(f"[WARNING] Found {len(suspect_features)} features with outcome-related names:")
    for feat in suspect_features[:10]:
        print(f"  - {feat}")
else:
    print("[OK] No obvious outcome-related features found")

print("\n[DEBUG] Analysis complete.")
