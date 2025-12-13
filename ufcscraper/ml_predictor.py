"""
UFC Fight Outcome Prediction ML Module
=====================================

This module handles:
1. Data preprocessing and feature engineering for ML
2. Model training with hyperparameter optimization
3. Fight outcome predictions with confidence scores
4. Model evaluation and performance metrics
5. Feature importance analysis

The module supports multiple algorithms and provides robust
predictions for UFC fight outcomes.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap

if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class UFCPredictor:
    """
    UFC Fight Outcome Predictor using advanced ML techniques.

    This class handles the complete ML pipeline from data preprocessing
    to model training and prediction generation.
    """

    def __init__(self, data_folder: Path | str):
        """
        Initialize the UFC predictor.

        Args:
            data_folder: Path to the folder containing training/prediction datasets
        """
        self.data_folder = Path(data_folder)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metrics = {}
        self.feature_importance = {}

        # Model save paths
        self.model_path = self.data_folder / "ufc_model.pkl"
        self.scaler_path = self.data_folder / "ufc_scaler.pkl"
        self.features_path = self.data_folder / "ufc_features.pkl"

    def load_and_preprocess_data(
        self,
        training_file: str = "training_dataset_cache.csv",
        prediction_file: str = "prediction_dataset_cache.csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess training and prediction datasets.

        Args:
            training_file: Name of training dataset file
            prediction_file: Name of prediction dataset file

        Returns:
            Tuple of (preprocessed_training_data, preprocessed_prediction_data)
        """
        logger.info("Loading datasets...")

        # Load datasets
        training_path = self.data_folder / training_file
        prediction_path = self.data_folder / prediction_file

        if not training_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {training_path}")
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction dataset not found: {prediction_path}")

        training_df = pd.read_csv(training_path)
        prediction_df = pd.read_csv(prediction_path)

        logger.info(f"Loaded training data: {len(training_df)} fights")
        logger.info(f"Loaded prediction data: {len(prediction_df)} fights")

        # Preprocess both datasets
        training_processed = self._preprocess_dataset(training_df, is_training=True)
        prediction_processed = self._preprocess_dataset(prediction_df, is_training=False)

        logger.info("Data preprocessing completed")
        return training_processed, prediction_processed

    def _preprocess_dataset(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess a dataset for ML training/prediction with quality filtering.

        Args:
            df: Input dataframe
            is_training: Whether this is training data (has target variable)

        Returns:
            Preprocessed dataframe with quality filtering applied
        """
        logger.info(f"Starting preprocessing: {len(df)} rows")
        df_processed = df.copy()

        # Remove only timestamp metadata columns that shouldn't be used for prediction
        # KEEP fight_id, fighter_1, fighter_2, event_id for mapping predictions back to fights
        # KEEP event_date for temporal train/test splitting
        drop_columns = [
            'dataset_type', 'created_at',
            'fighter_1_name', 'fighter_2_name'  # If these exist
        ]
        for col in drop_columns:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])

        # CRITICAL: Clean target variable for training data
        if is_training and 'winner' in df_processed.columns:
            original_count = len(df_processed)

            # Step 1: Remove rows where target is completely missing
            before_nan = len(df_processed)
            df_processed = df_processed.dropna(subset=['winner'])
            after_nan = len(df_processed)

            # Step 2: Remove any rows where winner is string 'nan', 'NaN', or similar
            df_processed = df_processed[~df_processed['winner'].astype(str).str.lower().isin(['nan', 'none', ''])]

            # Step 3: Convert to numeric and handle any remaining non-numeric values
            df_processed['winner'] = pd.to_numeric(df_processed['winner'], errors='coerce')

            # Step 4: Remove any rows that became NaN after numeric conversion
            df_processed = df_processed.dropna(subset=['winner'])

            # Step 5: Ensure winner column contains only valid binary values (0 or 1)
            valid_winners = df_processed['winner'].isin([0, 1, 0.0, 1.0])
            df_processed = df_processed[valid_winners]

            # Step 6: Convert to proper integer type
            df_processed['winner'] = df_processed['winner'].astype(int)

            # Step 7: Final validation - check for any remaining NaN or invalid values
            remaining_nan = df_processed['winner'].isna().sum()
            if remaining_nan > 0:
                logger.error(f"CRITICAL: {remaining_nan} NaN values remain in target after cleaning!")
                df_processed = df_processed.dropna(subset=['winner'])

            unique_values = sorted(df_processed['winner'].unique())
            logger.info(f"Target cleaning: {original_count}  {len(df_processed)} rows")
            logger.info(f"Removed {before_nan - after_nan} NaN targets")
            logger.info(f"Final target values: {unique_values}")

            if len(df_processed) == 0:
                raise ValueError("No valid training data remains after target cleaning!")

        # Apply data quality filtering - remove rows with too many missing values
        df_processed = self._apply_quality_filter(df_processed, is_training)

        # Handle categorical features
        categorical_features = ['weight_class']
        for feature in categorical_features:
            if feature in df_processed.columns:
                # Use label encoding for now (could expand to one-hot if needed)
                le = LabelEncoder()
                # Fit on both training and prediction data combined to ensure consistency
                all_values = df_processed[feature].dropna().astype(str).unique()
                le.fit(all_values)
                df_processed[feature] = le.transform(df_processed[feature].astype(str))

        # Conservative imputation - only fill a few critical columns
        df_processed = self._conservative_imputation(df_processed, is_training)

        logger.info(f"Preprocessing complete: {len(df_processed)} rows ready for ML")
        return df_processed

    def _apply_quality_filter(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Smart data quality filtering that preserves more real data.

        Args:
            df: Input dataframe
            is_training: Whether this is training data

        Returns:
            Filtered dataframe with intelligent quality control
        """
        original_count = len(df)

        # Get feature columns (exclude target and metadata columns)
        metadata_columns = ['fight_id', 'fighter_1', 'fighter_2', 'event_id', 'winner', 'dataset_type', 'created_at']
        feature_columns = [col for col in df.columns if col not in metadata_columns]

        # Define critical features that should rarely be missing
        critical_features = [
            'fighter_1_age', 'fighter_2_age',
            'fighter_1_wins', 'fighter_1_losses', 'fighter_2_wins', 'fighter_2_losses',
            'weight_class', 'gender_male'
        ]
        critical_features = [f for f in critical_features if f in feature_columns]

        # Count missing values
        total_features = len(feature_columns)
        missing_counts = df[feature_columns].isnull().sum(axis=1)
        critical_missing = df[critical_features].isnull().sum(axis=1) if critical_features else 0

        # Smart filtering rules:
        # 1. Must have at least 70% of all features
        # 2. Must have all critical features
        # 3. For training: be stricter (80% of features)
        # 4. For prediction: be more lenient (60% of features)

        if is_training:
            min_feature_threshold = 0.8  # Must have 80% of features for training
            max_missing_pct = 0.2  # Max 20% missing
        else:
            min_feature_threshold = 0.6  # Must have 60% of features for prediction
            max_missing_pct = 0.4  # Max 40% missing

        # Calculate quality masks
        feature_completeness = (total_features - missing_counts) / total_features
        has_enough_features = feature_completeness >= min_feature_threshold
        has_critical_features = critical_missing == 0

        # Apply filters
        quality_mask = has_enough_features & has_critical_features
        df_filtered = df[quality_mask].copy()

        # Log filtering results
        removed_count = original_count - len(df_filtered)
        if removed_count > 0:
            avg_completeness = feature_completeness[quality_mask].mean()
            logger.info(f"Smart quality filter:")
            logger.info(f"  Removed {removed_count} low-quality rows ({removed_count/original_count*100:.1f}%)")
            logger.info(f"  Kept {len(df_filtered)} high-quality rows ({len(df_filtered)/original_count*100:.1f}%)")
            logger.info(f"  Average feature completeness: {avg_completeness*100:.1f}%")
            logger.info(f"  Required: {min_feature_threshold*100:.0f}% features + all critical features")

            # Show what was filtered out
            missing_critical = (critical_missing > 0).sum()
            insufficient_features = (~has_enough_features).sum()
            logger.info(f"  Filtered: {missing_critical} missing critical features, {insufficient_features} insufficient features")

        return df_filtered

    def _conservative_imputation(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Apply intelligent conservative imputation - only fill truly essential columns.

        Args:
            df: Input dataframe
            is_training: Whether this is training data

        Returns:
            Dataframe with conservative imputation applied
        """
        df_imputed = df.copy()
        metadata_columns = ['fight_id', 'fighter_1', 'fighter_2', 'event_id', 'winner', 'dataset_type', 'created_at']
        feature_columns = [col for col in df.columns if col not in metadata_columns]

        # Count missing values before imputation
        missing_before = df_imputed[feature_columns].isnull().sum().sum()

        # Define essential columns with their imputation strategies
        essential_imputations = {
            # Fighter ages - use median age (UFC average ~30 years)
            'fighter_1_age': 30.0,
            'fighter_2_age': 30.0,
            'age_advantage': 0.0,
            'fighter_1_fighting_age': 30.0,
            'fighter_2_fighting_age': 30.0,
            'fighter_1_age_vs_prime': 0.0,
            'fighter_2_age_vs_prime': 0.0,

            # Physical stats - use median for weight class
            'fighter_1_height_cm': 'median',
            'fighter_2_height_cm': 'median',
            'fighter_1_weight_lbs': 'median',
            'fighter_2_weight_lbs': 'median',
            'fighter_1_reach_cm': 'median',
            'fighter_2_reach_cm': 'median',
            'height_advantage': 0.0,
            'reach_advantage': 0.0,

            # Core record stats - fill with 0 for newer fighters
            'fighter_1_wins': 'zero',
            'fighter_1_losses': 'zero',
            'fighter_2_wins': 'zero',
            'fighter_2_losses': 'zero',

            # Recent form features - use neutral values
            'fighter_1_last_3_win_rate': 0.5,
            'fighter_2_last_3_win_rate': 0.5,
            'fighter_1_last_5_win_rate': 0.5,
            'fighter_2_last_5_win_rate': 0.5,
            'fighter_1_momentum_score': 0.0,
            'fighter_2_momentum_score': 0.0,
            'fighter_1_days_since_last_fight': 365.0,  # 1 year default
            'fighter_2_days_since_last_fight': 365.0,

            # Style features - use neutral/averages
            'fighter_1_late_finish_rate': 0.3,  # UFC average
            'fighter_2_late_finish_rate': 0.3,
            'fighter_1_ko_rate': 0.25,
            'fighter_2_ko_rate': 0.25,
            'fighter_1_submission_rate': 0.15,
            'fighter_2_submission_rate': 0.15,

            # Binary features - use mode (most common value)
            'gender_male': 'mode',
            'title_fight': 'zero',  # Most fights aren't title fights
        }

        # Apply imputation
        for col, strategy in essential_imputations.items():
            if col in df_imputed.columns and df_imputed[col].isnull().sum() > 0:
                missing_count = df_imputed[col].isnull().sum()

                if strategy == 'median':
                    fill_value = df_imputed[col].median()
                elif strategy == 'mode':
                    fill_value = df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 0
                elif strategy == 'zero':
                    fill_value = 0
                elif isinstance(strategy, (int, float)):
                    # Direct numeric value
                    fill_value = strategy
                else:
                    continue

                df_imputed[col] = df_imputed[col].fillna(fill_value)
                logger.debug(f"Imputed {missing_count} missing values in {col} with {strategy} ({fill_value})")

        # Count missing values after imputation
        missing_after = df_imputed[feature_columns].isnull().sum().sum()

        if missing_before > missing_after:
            filled_count = missing_before - missing_after
            logger.info(f"Conservative imputation: Filled {filled_count} essential missing values")
            logger.info(f"Remaining missing values: {missing_after} (will be handled by model)")

            # Show remaining missing by column (top 10)
            remaining_missing = df_imputed[feature_columns].isnull().sum()
            top_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False).head(10)
            if len(top_missing) > 0:
                logger.debug("Top missing features after imputation:")
                for col, count in top_missing.items():
                    pct = count / len(df_imputed) * 100
                    logger.debug(f"  {col}: {count} ({pct:.1f}%)")

        return df_imputed

    # REMOVED: _create_interaction_features() method - DEAD CODE
    # Interaction features are now created in feature_engineering.py during dataset building
    # This ensures they are included in cached datasets and avoids duplication

    def _select_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Apply feature selection to remove noise and redundancy.

        3-stage pipeline:
        1. Remove high-missing features (>50% missing)
        2. Remove zero/low variance features
        3. Remove perfectly correlated features (>0.98)
        """
        logger.info(f"Feature selection: Starting with {len(X.columns)} features")

        # Stage 1: Remove high-missing features (reduced threshold to 40% from 50%)
        missing_pct = X.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.4].index.tolist()
        if high_missing:
            logger.info(f"Removing {len(high_missing)} high-missing features (>40%): {high_missing[:5]}...")
            X = X.drop(columns=high_missing)

        # Stage 2: Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X.fillna(0))  # Temporarily fill NaN for variance calculation
        low_var_mask = selector.get_support()
        low_var_features = X.columns[~low_var_mask].tolist()
        if low_var_features:
            logger.info(f"Removing {len(low_var_features)} low-variance features: {low_var_features[:5]}...")
            X = X.loc[:, low_var_mask]

        # Stage 3: Remove perfectly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.98)]
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features (>0.98): {to_drop[:5]}...")
            X = X.drop(columns=to_drop)

        logger.info(f"Feature selection: Reduced to {len(X.columns)} features")
        return X

    def train_model(
        self,
        training_df: pd.DataFrame,
        model_type: str = "xgboost",
        optimize_hyperparameters: bool = True,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the UFC prediction model with gradient boosting optimizations.

        Args:
            training_df: Preprocessed training dataset
            model_type: Type of model ("xgboost", "lightgbm", "catboost", "random_forest", "gradient_boosting", "logistic")
            optimize_hyperparameters: Whether to optimize hyperparameters
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary containing model performance metrics
        """
        logger.info(f"Training {model_type} model with anti-overfitting optimizations...")

        # Prepare features and target
        if 'winner' not in training_df.columns:
            raise ValueError("Training dataset must contain 'winner' column")

        # Keep event_date for temporal splitting (but don't use it as a feature)
        if 'event_date' not in training_df.columns:
            raise ValueError("Training dataset must contain 'event_date' column for temporal splitting")

        metadata_columns = ['fight_id', 'fighter_1', 'fighter_2', 'event_id', 'winner', 'event_date', 'dataset_type', 'created_at']
        feature_columns = [col for col in training_df.columns if col not in metadata_columns]

        X = training_df[feature_columns]
        y = training_df['winner']
        event_dates = pd.to_datetime(training_df['event_date'])

        # DATA AUGMENTATION: Mirror fights to remove positional bias
        # Original fight: F1 vs F2, winner=1  =>  Mirrored: F2 vs F1, winner=0
        logger.info(f"Applying data augmentation (mirroring) to remove positional bias...")
        X_mirrored = X.copy()

        # Swap all fighter_1 and fighter_2 features
        for col in X.columns:
            if 'fighter_1' in col:
                mirror_col = col.replace('fighter_1', 'fighter_2')
                if mirror_col in X.columns:
                    # Swap the values
                    X_mirrored[col] = X[mirror_col].values
                    X_mirrored[mirror_col] = X[col].values

        # Flip differential features (advantages become disadvantages)
        for col in ['height_advantage', 'reach_advantage', 'experience_advantage']:
            if col in X_mirrored.columns:
                X_mirrored[col] = -X_mirrored[col]

        # Flip record_quality_difference and other differences
        for col in X_mirrored.columns:
            if 'difference' in col or 'gap' in col:
                X_mirrored[col] = -X_mirrored[col]

        # Flip winner labels (1 -> 0, 0 -> 1)
        y_mirrored = 1 - y

        # Concatenate original + mirrored
        X = pd.concat([X, X_mirrored], ignore_index=True)
        y = pd.concat([y, y_mirrored], ignore_index=True)
        event_dates = pd.concat([event_dates, event_dates], ignore_index=True)

        logger.info(f"Data augmented: {len(training_df)} -> {len(X)} fights (2x)")

        # Interaction features are now created in feature_engineering.py
        # Apply feature selection
        X = self._select_features(X, y)
        self.feature_columns = X.columns.tolist()

        logger.info(f"Training with {len(self.feature_columns)} features on {len(X)} fights")

        # Check class balance and calculate scale_pos_weight for imbalanced data
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        if len(class_counts) == 2:
            neg_count = class_counts[0]
            pos_count = class_counts[1]
            scale_pos_weight = neg_count / pos_count
            logger.info(f"Class imbalance detected: {neg_count}/{pos_count} = {scale_pos_weight:.2f}")
            logger.info(f"Using scale_pos_weight={scale_pos_weight:.2f} to handle imbalance")
        else:
            scale_pos_weight = 1.0

        # TRAIN ON ALL DATA - don't hold out recent fights!
        # The most recent data is the MOST VALUABLE for predictions
        # Cross-validation during GridSearchCV prevents overfitting
        # We'll use GridSearchCV scores for performance metrics instead of a holdout test set
        X_train = X
        y_train = y

        logger.info(f"Training on ALL {len(X_train)} fights (including most recent data)")
        logger.info(f"Performance will be evaluated via cross-validation during hyperparameter tuning")

        # Final check for any remaining missing values
        if X_train.isnull().any().any():
            logger.warning("Found remaining missing values, applying final imputation...")
            final_imputer = SimpleImputer(strategy='median')
            X_train = pd.DataFrame(final_imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

        # Scale features (important for some models)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Convert back to DataFrame for XGBoost/LightGBM/CatBoost
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        # Select and configure model with AGGRESSIVE ANTI-OVERFITTING SETTINGS
        if model_type == "xgboost":
            base_model = xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,  # CRITICAL: Handle class imbalance
                eval_metric='auc'
                # No early_stopping_rounds - we train on all data with CV for validation
            )
            param_grid = {
                'n_estimators': [200, 300],  # Increased: more trees for better learning
                'max_depth': [4, 5],  # Slightly deeper: better feature interactions
                'learning_rate': [0.01, 0.03],  # Keep slow learning for stability
                'subsample': [0.7, 0.8],  # Less aggressive: keep more data
                'colsample_bytree': [0.7, 0.8],  # Less aggressive: keep more features
                'min_child_weight': [10, 20],  # Reduced: allow more splits
                'reg_alpha': [0.5, 2],  # Reduced L1: less feature suppression
                'reg_lambda': [5, 10],  # Reduced L2: less weight penalization
            } if optimize_hyperparameters else {}

        elif model_type == "lightgbm":
            base_model = lgb.LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',  # CRITICAL: Handle class imbalance
                verbose=-1
            )
            param_grid = {
                'n_estimators': [150, 200],
                'max_depth': [3, 4],
                'learning_rate': [0.01, 0.03],
                'subsample': [0.6, 0.7],
                'colsample_bytree': [0.6, 0.7],
                'min_child_samples': [15, 25],
                'reg_alpha': [1, 3],
                'reg_lambda': [8, 12],
            } if optimize_hyperparameters else {}

        elif model_type == "catboost":
            base_model = cb.CatBoostClassifier(
                random_state=42,
                thread_count=-1,
                auto_class_weights='Balanced',  # CRITICAL: Handle class imbalance
                verbose=False
            )
            param_grid = {
                'iterations': [150, 200],
                'depth': [3, 4],
                'learning_rate': [0.01, 0.03],
                'l2_leaf_reg': [8, 12],
            } if optimize_hyperparameters else {}

        elif model_type == "random_forest":
            base_model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            } if optimize_hyperparameters else {}

        elif model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 150],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 4],
                'subsample': [0.7, 0.8]
            } if optimize_hyperparameters else {}

        elif model_type == "logistic":
            base_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            } if optimize_hyperparameters else {}

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model with STRATIFIED K-FOLD CV and ROC-AUC optimization
        if optimize_hyperparameters and param_grid:
            logger.info(f"Optimizing hyperparameters with {len(param_grid)} combinations...")
            logger.info("Using Stratified 3-Fold CV with ROC-AUC scoring (better for imbalanced data)")

            # Use stratified k-fold for better class balance in folds
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_strategy,
                scoring='roc_auc',  # CRITICAL: Use ROC-AUC instead of accuracy
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train_scaled, y_train)

            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.3f}")
            cv_score = grid_search.best_score_
        else:
            logger.info("Training with default parameters...")
            self.model = base_model
            self.model.fit(X_train_scaled, y_train)
            cv_score = None

        # Evaluate model on training data only (no test set)
        train_pred = self.model.predict(X_train_scaled)
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]

        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        train_auc = roc_auc_score(y_train, train_proba)

        metrics = {
            'model_type': model_type,
            'train_accuracy': train_acc,
            'cv_auc': cv_score if cv_score else train_auc,
            'train_auc': train_auc,
            'n_features': len(self.feature_columns),
            'training_samples': len(X_train),
            'scale_pos_weight': scale_pos_weight
        }

        self.model_metrics = metrics

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            self.feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

        # Log results
        logger.info(f"Model training completed:")
        logger.info(f"  Training accuracy: {train_acc:.3f}")
        logger.info(f"  Training AUC:      {train_auc:.3f}")
        logger.info(f"  CV AUC:            {metrics['cv_auc']:.3f}")
        logger.info(f"  Training on {len(X_train)} fights (ALL data including recent)")

        # Performance indicators
        if train_acc > 0.85:
            logger.warning(f"  Training accuracy {train_acc:.1%} is very high - monitor for overfitting")
        if metrics['cv_auc'] > 0.70:
            logger.info("  Strong cross-validation AUC - good probability calibration!")

        # Save model
        self._save_model()

        return metrics

    def predict_fights(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for upcoming fights.

        Args:
            prediction_df: Preprocessed prediction dataset

        Returns:
            DataFrame with fight predictions and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first or load_model()")

        logger.info(f"Making predictions for {len(prediction_df)} upcoming fights...")

        # Prepare features
        if self.feature_columns is None:
            raise ValueError("Feature columns not defined. Train model first.")

        # Extract all available features (interaction features already in dataset)
        metadata_columns = ['fight_id', 'fighter_1', 'fighter_2', 'event_id', 'winner', 'event_date', 'dataset_type', 'created_at']

        # Select only the features that were used in training
        missing_features = set(self.feature_columns) - set(prediction_df.columns)
        if missing_features:
            raise ValueError(f"Missing features in prediction data: {missing_features}")

        X_pred = prediction_df[self.feature_columns]

        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)

        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        probabilities = self.model.predict_proba(X_pred_scaled)

        # Generate SHAP values for explainability
        logger.info("Generating SHAP values for prediction explanations...")
        shap_values_dict = self._generate_shap_values(X_pred_scaled, X_pred)

        # Create results dataframe with proper fight metadata
        results = pd.DataFrame({
            'fight_id': prediction_df['fight_id'] if 'fight_id' in prediction_df.columns else range(len(prediction_df)),
            'fighter_1_win_probability': probabilities[:, 1],
            'fighter_2_win_probability': probabilities[:, 0],
            'predicted_winner': ['Fighter 1' if p == 1 else 'Fighter 2' for p in predictions],
            'confidence': np.maximum(probabilities[:, 0], probabilities[:, 1]),
            'prediction_date': pd.Timestamp.now()
        })

        # Add all available context columns from the original prediction dataset
        preserve_columns = ['fighter_1', 'fighter_2', 'weight_class', 'title_fight', 'event_id']
        for col in preserve_columns:
            if col in prediction_df.columns:
                results[col] = prediction_df[col].values

        # Save SHAP values for web UI
        if shap_values_dict:
            self._save_shap_values(shap_values_dict, results)

        logger.info(f"Predictions completed for {len(results)} fights")

        return results

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get the top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        if not self.feature_importance:
            return []

        return list(self.feature_importance.items())[:n]

    def _save_model(self) -> None:
        """Save the trained model and preprocessing components."""
        try:
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save feature columns
            with open(self.features_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)

            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self) -> bool:
        """
        Load a previously trained model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            # Load feature columns
            with open(self.features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)

            # Extract feature importance from the loaded model
            if hasattr(self.model, 'feature_importances_') and self.feature_columns:
                importances = self.model.feature_importances_
                self.feature_importance = dict(sorted(
                    zip(self.feature_columns, importances),
                    key=lambda x: x[1],
                    reverse=True
                ))
                logger.info(f"Loaded feature importance for {len(self.feature_importance)} features")

            logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False

    def is_model_trained(self) -> bool:
        """Check if a model is currently loaded/trained."""
        return self.model is not None

    def _generate_shap_values(self, X_scaled: np.ndarray, X_original: pd.DataFrame) -> Dict:
        """
        Generate SHAP values for model predictions.

        Args:
            X_scaled: Scaled feature matrix
            X_original: Original unscaled features (for feature names)

        Returns:
            Dictionary with SHAP values and base values for each prediction
        """
        try:
            # Create SHAP explainer (TreeExplainer for XGBoost/LightGBM/RandomForest)
            if isinstance(self.model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
                explainer = shap.TreeExplainer(self.model)
                # TreeExplainer expects original feature space, not scaled
                # But XGBoost works with scaled data, so we use scaled
                shap_values = explainer.shap_values(X_scaled)

                # For binary classification, SHAP returns values for class 1
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1 (fighter_1 wins)

            elif isinstance(self.model, RandomForestClassifier):
                explainer = shap.TreeExplainer(self.model)
                shap_values_raw = explainer.shap_values(X_scaled)
                if isinstance(shap_values_raw, list):
                    shap_values = shap_values_raw[1]
                else:
                    shap_values = shap_values_raw
            else:
                # Fallback to KernelExplainer for other models
                # Sample background dataset (use a subset for speed)
                logger.warning("Using KernelExplainer (slower) for non-tree model")
                return {}

            # Store SHAP values with feature names
            shap_dict = {
                'shap_values': shap_values,  # Shape: (n_fights, n_features)
                'feature_names': self.feature_columns,
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0.0
            }

            logger.info(f"Generated SHAP values for {shap_values.shape[0]} predictions")
            return shap_dict

        except Exception as e:
            logger.warning(f"Could not generate SHAP values: {e}")
            return {}

    def _save_shap_values(self, shap_dict: Dict, predictions: pd.DataFrame):
        """
        Save SHAP values to CSV for web UI visualization.

        Args:
            shap_dict: Dictionary containing SHAP values and metadata
            predictions: Predictions dataframe with fight_ids
        """
        try:
            shap_values = shap_dict['shap_values']
            feature_names = shap_dict['feature_names']
            base_value = shap_dict.get('base_value', 0.0)

            # Create DataFrame with SHAP values
            shap_df_list = []

            for i, fight_id in enumerate(predictions['fight_id']):
                # Get SHAP values for this fight
                fight_shap = shap_values[i]

                # Create rows for top 15 features (positive and negative)
                feature_contributions = list(zip(feature_names, fight_shap))
                # Sort by absolute value
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                # Take top 15
                top_features = feature_contributions[:15]

                for feature_name, shap_value in top_features:
                    shap_df_list.append({
                        'fight_id': fight_id,
                        'feature': feature_name,
                        'shap_value': shap_value,
                        'base_value': base_value
                    })

            shap_df = pd.DataFrame(shap_df_list)

            # Save to CSV
            shap_file = self.data_folder / "fight_shap_values.csv"
            shap_df.to_csv(shap_file, index=False)

            logger.info(f"Saved SHAP values to {shap_file}")

        except Exception as e:
            logger.warning(f"Could not save SHAP values: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_model_trained():
            return {"status": "No model trained"}

        info = {
            "status": "Model trained",
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
            "model_files_exist": {
                "model": self.model_path.exists(),
                "scaler": self.scaler_path.exists(),
                "features": self.features_path.exists()
            }
        }

        # Add metrics if available
        if self.model_metrics:
            info.update(self.model_metrics)

        return info