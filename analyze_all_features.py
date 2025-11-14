"""
Comprehensive analysis of ALL features to identify data leakage sources.
This script will test each feature's correlation with the target and identify suspicious patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_features():
    """Analyze all features for potential data leakage."""

    # Load training dataset
    logger.info("Loading training dataset...")
    df = pd.read_csv('ufc_data/training_dataset_cache.csv')

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Separate features and target
    target_col = 'winner'
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found!")
        return

    # Get feature columns (exclude metadata and target)
    metadata_cols = ['fight_id', 'event_id', 'event_date', 'fighter_1', 'fighter_2',
                     'fighter_1_id', 'fighter_2_id', 'winner', 'method', 'round', 'time',
                     'weight_class', 'dataset_type', 'created_at']  # Add non-numeric cols
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    logger.info(f"\n{'='*80}")
    logger.info(f"FEATURE ANALYSIS - Total features: {len(feature_cols)}")
    logger.info(f"{'='*80}\n")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Only keep numeric columns
    X = X.select_dtypes(include=[np.number])
    feature_cols = list(X.columns)

    logger.info(f"Numeric features only: {len(feature_cols)}")

    # Handle missing values for correlation analysis
    X_filled = X.fillna(X.median())

    # Calculate correlations with target
    correlations = []
    for col in feature_cols:
        if col in X_filled.columns:
            corr = abs(X_filled[col].corr(y))
            correlations.append({'feature': col, 'correlation': corr})

    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

    # Show top 30 most correlated features
    logger.info("\nTOP 30 FEATURES BY CORRELATION WITH WINNER:")
    logger.info("-" * 80)
    for idx, row in corr_df.head(30).iterrows():
        logger.info(f"{row['feature']:50s} | Correlation: {row['correlation']:.4f}")

    # Identify highly suspicious features (>15% correlation is very suspicious)
    suspicious = corr_df[corr_df['correlation'] > 0.15]
    if len(suspicious) > 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"HIGHLY SUSPICIOUS FEATURES (correlation > 15%):")
        logger.info(f"{'='*80}")
        for idx, row in suspicious.iterrows():
            logger.info(f"  {row['feature']}: {row['correlation']:.4f}")

    # Train a simple model to check feature importance
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SIMPLE MODEL TO CHECK FEATURE IMPORTANCE...")
    logger.info(f"{'='*80}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_filled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train simple model (shallow tree to avoid overfitting)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,  # Shallow to see main patterns
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    logger.info(f"Simple model (max_depth=5) results:")
    logger.info(f"  Training accuracy: {train_acc:.4f}")
    logger.info(f"  Test accuracy: {test_acc:.4f}")
    logger.info(f"  Overfitting gap: {train_acc - test_acc:.4f}")

    if train_acc > 0.80:
        logger.info("\n[WARNING] Even a SHALLOW tree has >80% train accuracy!")
        logger.info("This strongly indicates data leakage in the features.")

    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTOP 30 FEATURES BY IMPORTANCE (shallow tree):")
    logger.info("-" * 80)
    for idx, row in importances.head(30).iterrows():
        logger.info(f"{row['feature']:50s} | Importance: {row['importance']:.4f}")

    # Check for specific suspicious patterns in feature names
    logger.info(f"\n{'='*80}")
    logger.info("CHECKING FOR SUSPICIOUS FEATURE PATTERNS...")
    logger.info(f"{'='*80}\n")

    suspicious_patterns = {
        'total': [],
        'career': [],
        'overall': [],
        'cumulative': [],
        'all_time': []
    }

    for col in feature_cols:
        col_lower = col.lower()
        for pattern in suspicious_patterns.keys():
            if pattern in col_lower:
                suspicious_patterns[pattern].append(col)

    for pattern, features in suspicious_patterns.items():
        if features:
            logger.info(f"\nFeatures containing '{pattern}' ({len(features)}):")
            for f in features[:10]:  # Show first 10
                logger.info(f"  - {f}")
            if len(features) > 10:
                logger.info(f"  ... and {len(features) - 10} more")

    # Look for features with very low variance (might be constant)
    logger.info(f"\n{'='*80}")
    logger.info("CHECKING FOR LOW-VARIANCE FEATURES...")
    logger.info(f"{'='*80}\n")

    variances = X_filled.var()
    low_var = variances[variances < 0.01].sort_values()

    if len(low_var) > 0:
        logger.info(f"Found {len(low_var)} features with variance < 0.01:")
        for feat, var in low_var.items():
            logger.info(f"  {feat}: variance = {var:.6f}")

    # Save full correlation report
    corr_df.to_csv('ufc_data/feature_correlations.csv', index=False)
    importances.to_csv('ufc_data/feature_importances.csv', index=False)

    logger.info(f"\n{'='*80}")
    logger.info("Reports saved:")
    logger.info("  - ufc_data/feature_correlations.csv")
    logger.info("  - ufc_data/feature_importances.csv")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    analyze_features()
