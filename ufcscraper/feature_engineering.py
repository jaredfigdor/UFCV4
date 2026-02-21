"""
Feature engineering module for UFC fight prediction.

This module provides comprehensive feature engineering capabilities for creating
leak-free training and prediction datasets for UFC fight outcome prediction.
All features are calculated using only information available before the fight date.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature engineering class for UFC fight prediction datasets.

    This class creates comprehensive features for predicting fight outcomes,
    ensuring temporal consistency and preventing data leakage.
    """

    def __init__(self):
        """Initialize the feature engineering pipeline."""
        self.fighter_cache = {}
        self.fight_cache = {}

    def create_fight_features(
        self,
        fights_df: pd.DataFrame,
        fighters_df: pd.DataFrame,
        rounds_df: pd.DataFrame,
        events_df: pd.DataFrame,
        is_prediction: bool = False,
        all_completed_fights: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set for fights.

        Args:
            fights_df: Fight data
            fighters_df: Fighter data
            rounds_df: Round-by-round data
            events_df: Event data
            is_prediction: Whether this is for prediction dataset
            all_completed_fights: All completed fights for historical context

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Creating features for {len(fights_df)} fights (prediction={is_prediction})")

        # Merge fight and event data to get fight dates
        fights_with_dates = fights_df.merge(
            events_df[['event_id', 'event_date']],
            on='event_id',
            how='left'
        )
        fights_with_dates['event_date'] = pd.to_datetime(fights_with_dates['event_date'])

        # Sort by date to ensure temporal consistency
        fights_with_dates = fights_with_dates.sort_values('event_date').reset_index(drop=True)

        features_list = []

        for idx, fight in fights_with_dates.iterrows():
            logger.debug(f"Processing fight {idx+1}/{len(fights_with_dates)}")

            # Get fighter data - SKIP FIGHT if fighters don't exist
            fighter_1_match = fighters_df[fighters_df['fighter_id'] == fight['fighter_1']]
            fighter_2_match = fighters_df[fighters_df['fighter_id'] == fight['fighter_2']]

            if len(fighter_1_match) == 0:
                logger.warning(f"Skipping fight {fight['fight_id']}: Fighter 1 ({fight['fighter_1']}) not found in fighter database")
                continue
            if len(fighter_2_match) == 0:
                logger.warning(f"Skipping fight {fight['fight_id']}: Fighter 2 ({fight['fighter_2']}) not found in fighter database")
                continue

            fighter_1_data = fighter_1_match.iloc[0]
            fighter_2_data = fighter_2_match.iloc[0]

            # For historical context, use appropriate dataset
            if is_prediction and all_completed_fights is not None:
                # For predictions, use all completed fights as historical context
                historical_context = all_completed_fights
            elif not is_prediction and all_completed_fights is not None:
                # For training, use ALL completed fights before current fight date to prevent leakage
                current_fight_date = fight['event_date']
                # Ensure dates are datetime for comparison
                all_completed_fights_copy = all_completed_fights.copy()
                all_completed_fights_copy['event_date'] = pd.to_datetime(all_completed_fights_copy['event_date'])
                historical_context = all_completed_fights_copy[
                    all_completed_fights_copy['event_date'] < current_fight_date
                ]
            else:
                # Fallback: use only fights in current dataset before current fight
                current_fight_date = fight['event_date']
                historical_context = fights_with_dates[
                    fights_with_dates['event_date'] < current_fight_date
                ]

            # Create features for this fight
            fight_features = self._create_single_fight_features(
                fight=fight,
                fighter_1_data=fighter_1_data,
                fighter_2_data=fighter_2_data,
                all_fights=historical_context,
                rounds_df=rounds_df,
                fighters_df=fighters_df,
                is_prediction=is_prediction
            )

            features_list.append(fight_features)

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        logger.info(f"Created {len(features_df.columns)} features for {len(features_df)} fights")
        return features_df

    def _create_single_fight_features(
        self,
        fight: pd.Series,
        fighter_1_data: pd.Series,
        fighter_2_data: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame,
        fighters_df: pd.DataFrame,
        is_prediction: bool = False
    ) -> Dict:
        """Create features for a single fight."""
        features = {}

        # Basic fight information
        features['fight_id'] = fight['fight_id']
        features['event_date'] = fight['event_date']

        # Fighter identifiers (essential for name mapping)
        features['fighter_1'] = fight['fighter_1']
        features['fighter_2'] = fight['fighter_2']

        # Event identifier (for event name mapping)
        if 'event_id' in fight:
            features['event_id'] = fight['event_id']

        # Target variable (empty for prediction)
        if not is_prediction and 'winner' in fight:
            features['winner'] = 1 if fight['winner'] == fight['fighter_1'] else 0
        else:
            features['winner'] = np.nan

        # Fight context features
        features.update(self._create_fight_context_features(fight))

        # Fighter physical features
        features.update(self._create_physical_features(fighter_1_data, fighter_2_data, fight))

        # Fighter record features (using temporal snapshots to prevent leakage)
        features.update(self._create_record_features(
            fighter_1_data, fighter_2_data,
            current_fight=fight,
            all_fights=all_fights,
            is_prediction=is_prediction
        ))

        # Historical performance features
        features.update(self._create_historical_features(
            fight, fighter_1_data, fighter_2_data, all_fights, rounds_df
        ))

        # Matchup features
        features.update(self._create_matchup_features(fighter_1_data, fighter_2_data))

        # Advanced career momentum features
        features.update(self._create_momentum_features(
            fight, fighter_1_data, fighter_2_data, all_fights
        ))

        # Fighting style and method features
        features.update(self._create_style_features(
            fight, fighter_1_data, fighter_2_data, all_fights, rounds_df
        ))

        # Career stage and prime features
        features.update(self._create_career_stage_features(
            fight, fighter_1_data, fighter_2_data, all_fights
        ))

        # Elo rating system features (combat-adapted with finish bonuses)
        features.update(self._create_elo_features(
            fight, fighter_1_data, fighter_2_data, all_fights
        ))

        # Advanced matchup analysis
        features.update(self._create_advanced_matchup_features(
            fighter_1_data, fighter_2_data, all_fights, rounds_df, fight
        ))

        # Opponent quality features
        features.update(self._create_opponent_quality_features(
            fight, fighter_1_data, fighter_2_data, all_fights, fighters_df
        ))

        # Damage history and chin erosion features
        features.update(self._create_damage_history_features(
            fight, fighter_1_data, fighter_2_data, all_fights, rounds_df
        ))

        # Pace sustainability and cardio decay features
        features.update(self._create_pace_sustainability_features(
            fight, fighter_1_data, fighter_2_data, all_fights, rounds_df
        ))

        # Late-round performance and cardio mismatch features
        features.update(self._create_late_round_features(
            fight, fighter_1_data, fighter_2_data, all_fights, rounds_df
        ))

        # Takedown defense under fatigue features
        features.update(self._create_fatigue_defense_features(
            fight, fighter_1_data, fighter_2_data, all_fights, rounds_df
        ))

        # Interaction features - create polynomial combinations of top predictors
        features.update(self._create_interaction_features_inline(features))

        return features

    def _create_interaction_features_inline(self, features: Dict) -> Dict:
        """Create interaction features from existing features."""
        interactions = {}

        # REMOVED: All interaction features that used positional features
        # These created the same dominance issues as the base features
        # Keeping only differential features prevents positional bias

        return interactions

    def _create_fight_context_features(self, fight: pd.Series) -> Dict:
        """Create fight context features."""
        features = {}

        # Weight class - encode as numeric for XGBoost
        # Ordered by weight (lightest to heaviest)
        weight_class_map = {
            "Women's Strawweight": 0.5,  # 115 lbs
            'Flyweight': 1,  # 125 lbs
            "Women's Flyweight": 1,  # 125 lbs
            'Bantamweight': 2,  # 135 lbs
            "Women's Bantamweight": 2,  # 135 lbs
            "Women's Featherweight": 3,  # 145 lbs
            'Featherweight': 3,  # 145 lbs
            'Lightweight': 4,  # 155 lbs
            'Welterweight': 5,  # 170 lbs
            'Middleweight': 6,  # 185 lbs
            'Light Heavyweight': 7,  # 205 lbs
            'Heavyweight': 8,  # 265 lbs
            'Catch Weight': 4.5,  # Between typical weight classes
            'Open Weight': 8.5,  # Heavier than heavyweight
            'Unknown': 5  # Default to middle
        }
        wc = fight.get('weight_class', 'Unknown')
        features['weight_class'] = weight_class_map.get(wc, 5)  # Default to 5 (middleweight)

        # Title fight
        features['title_fight'] = 1 if fight.get('title_fight') == 'T' else 0

        # Number of rounds
        if 'num_rounds' in fight and pd.notna(fight['num_rounds']):
            features['scheduled_rounds'] = int(fight['num_rounds'])
        else:
            features['scheduled_rounds'] = 3  # Default

        # Gender
        features['gender_male'] = 1 if fight.get('gender') == 'M' else 0

        return features

    def _create_physical_features(
        self,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        fight: pd.Series
    ) -> Dict:
        """Create physical attribute features."""
        features = {}

        # Calculate age at fight time
        fight_date = pd.to_datetime(fight['event_date'])

        for i, fighter in enumerate([fighter_1, fighter_2], 1):
            prefix = f'fighter_{i}_'

            # Age at fight
            if pd.notna(fighter.get('fighter_dob')):
                dob = pd.to_datetime(fighter['fighter_dob'])
                age = (fight_date - dob).days / 365.25
                features[f'{prefix}age'] = age
            else:
                features[f'{prefix}age'] = np.nan

            # Physical stats
            features[f'{prefix}height_cm'] = fighter.get('fighter_height_cm', np.nan)
            features[f'{prefix}weight_lbs'] = fighter.get('fighter_weight_lbs', np.nan)
            features[f'{prefix}reach_cm'] = fighter.get('fighter_reach_cm', np.nan)

            # Stance (one-hot)
            stances = ['Orthodox', 'Southpaw', 'Switch', 'Open Stance']
            for stance in stances:
                features[f'{prefix}stance_{stance.lower().replace(" ", "_")}'] = (
                    1 if fighter.get('fighter_stance') == stance else 0
                )

        # Physical advantages
        features['height_advantage'] = (
            features.get('fighter_1_height_cm', 0) - features.get('fighter_2_height_cm', 0)
        )
        features['reach_advantage'] = (
            features.get('fighter_1_reach_cm', 0) - features.get('fighter_2_reach_cm', 0)
        )

        # REMOVED: age_advantage - was dominating predictions even with sqrt transformation
        # Age information still available through fighter_1_age and fighter_2_age individual features
        # The model can learn age effects without being biased by explicit age gaps

        return features

    def _calculate_record_at_time(
        self,
        fighter_id: str,
        fighter_data: pd.Series,
        all_fights: pd.DataFrame,
        cutoff_date: pd.Timestamp
    ) -> Dict:
        """
        Calculate a fighter's win-loss-draw record at a specific point in time.

        Uses the fighter's current career totals from fighter_data.csv as baseline,
        then subtracts any UFC fights that occurred AFTER the cutoff date.

        This handles:
        - Pre-UFC career fights (included in fighter_data totals)
        - Fights outside our dataset date range
        - Temporal accuracy (no data leakage)
        """
        # Start with current career totals from fighter_data
        current_wins = fighter_data.get('fighter_w', 0)
        current_losses = fighter_data.get('fighter_l', 0)
        current_draws = fighter_data.get('fighter_d', 0)

        # Handle NaN values
        current_wins = 0 if pd.isna(current_wins) else int(current_wins)
        current_losses = 0 if pd.isna(current_losses) else int(current_losses)
        current_draws = 0 if pd.isna(current_draws) else int(current_draws)

        # Get all UFC fights AFTER cutoff date (these shouldn't be counted)
        future_fights = all_fights[
            ((all_fights['fighter_1'] == fighter_id) | (all_fights['fighter_2'] == fighter_id)) &
            (pd.to_datetime(all_fights['event_date']) >= cutoff_date)
        ].copy()

        # Subtract future fight results from current totals
        future_wins = 0
        future_losses = 0
        future_draws = 0

        for _, fight in future_fights.iterrows():
            if 'winner' in fight and pd.notna(fight['winner']):
                if fight['winner'] == fighter_id:
                    future_wins += 1
                else:
                    # Check if it's a draw
                    if fight.get('result') and 'Draw' in str(fight['result']):
                        future_draws += 1
                    else:
                        future_losses += 1
            elif fight.get('result') and 'Draw' in str(fight['result']):
                future_draws += 1

        # Calculate record at cutoff time (current - future)
        wins = max(0, current_wins - future_wins)
        losses = max(0, current_losses - future_losses)
        draws = max(0, current_draws - future_draws)

        total_fights = wins + losses + draws
        win_percentage = wins / total_fights if total_fights > 0 else 0.0

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'total_fights': total_fights,
            'win_percentage': win_percentage
        }

    def _create_record_features(
        self,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        current_fight: pd.Series = None,
        all_fights: pd.DataFrame = None,
        is_prediction: bool = False
    ) -> Dict:
        """
        Create fighter record features using temporal snapshots.

        For prediction fights (upcoming): Uses current career totals from fighter_data.csv
        For historical fights (training): Calculates record at that point in time to prevent leakage
        """
        features = {}

        # For upcoming fights (prediction), use current career totals from fighter_data
        if is_prediction:
            for i, fighter in enumerate([fighter_1, fighter_2], 1):
                prefix = f'fighter_{i}_'

                # Use actual career record from fighter_data.csv
                wins = fighter.get('fighter_w', 0)
                losses = fighter.get('fighter_l', 0)
                draws = fighter.get('fighter_d', 0)

                # Handle NaN values
                wins = 0 if pd.isna(wins) else int(wins)
                losses = 0 if pd.isna(losses) else int(losses)
                draws = 0 if pd.isna(draws) else int(draws)

                total_fights = wins + losses + draws

                # REMOVED individual W-L-D records (same reason as below)
                # Store temporarily for differential calculation
                if i == 1:
                    f1_total_pred = total_fights
                else:
                    f2_total_pred = total_fights

            # Experience advantage (differential feature only)
            features['experience_advantage'] = f1_total_pred - f2_total_pred

            return features

        # For historical fights (training), calculate record at that point in time
        if current_fight is None or all_fights is None or all_fights.empty:
            # Fallback: Set only differential features
            features['experience_advantage'] = 0
            return features

        current_fight_date = pd.to_datetime(current_fight['event_date'])

        for i, (fighter, fighter_id) in enumerate([
            (fighter_1, current_fight['fighter_1']),
            (fighter_2, current_fight['fighter_2'])
        ], 1):
            prefix = f'fighter_{i}_'

            # Calculate record from historical fights BEFORE current fight
            record = self._calculate_record_at_time(
                fighter_id=fighter_id,
                fighter_data=fighter,
                all_fights=all_fights,
                cutoff_date=current_fight_date
            )

            # REMOVED individual W-L-D records to prevent positional bias
            # These features (fighter_1_wins, fighter_2_wins, etc.) were dominating predictions
            # because the model learned "fighter_1 with 20 wins usually wins" regardless of opponent
            # Keep only differential features (experience_advantage) and aggregate metrics

            # Store temporarily for differential calculation
            if i == 1:
                f1_total = record['total_fights']
            else:
                f2_total = record['total_fights']

        # Experience advantage (differential feature - position-invariant)
        features['experience_advantage'] = f1_total - f2_total

        return features

    def _create_historical_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame
    ) -> Dict:
        """Create historical performance features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights (before current fight)
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                # No historical data - set defaults
                for window in [2, 3, 5]:
                    features[f'{prefix}last_{window}_win_rate'] = np.nan
                    features[f'{prefix}last_{window}_finish_rate'] = np.nan
                    features[f'{prefix}last_{window}_avg_fight_time'] = np.nan

                # Existing features
                features[f'{prefix}avg_striking_accuracy'] = np.nan
                features[f'{prefix}avg_takedown_accuracy'] = np.nan
                features[f'{prefix}avg_control_time'] = np.nan
                features[f'{prefix}avg_strikes_per_round'] = np.nan
                features[f'{prefix}days_since_last_fight'] = np.nan

                # Existing round stats features
                features[f'{prefix}takedown_defense'] = np.nan
                features[f'{prefix}striking_defense'] = np.nan
                features[f'{prefix}strikes_landed_per_min'] = np.nan
                features[f'{prefix}strikes_absorbed_per_min'] = np.nan
                features[f'{prefix}knockdowns_per_fight'] = np.nan
                features[f'{prefix}head_strike_pct'] = np.nan
                features[f'{prefix}body_strike_pct'] = np.nan
                features[f'{prefix}leg_strike_pct'] = np.nan
                features[f'{prefix}avg_takedowns_attempted'] = np.nan
                features[f'{prefix}submission_attempts_per_fight'] = np.nan
                features[f'{prefix}reversals_per_fight'] = np.nan
                # New absorption features
                features[f'{prefix}head_strikes_absorbed_per_min'] = np.nan
                features[f'{prefix}body_strikes_absorbed_per_min'] = np.nan
                features[f'{prefix}leg_strikes_absorbed_per_min'] = np.nan
                features[f'{prefix}head_absorption_rate'] = np.nan
                features[f'{prefix}knockdowns_absorbed_per_fight'] = np.nan
                features[f'{prefix}knockdowns_absorbed_per_min'] = np.nan
                features[f'{prefix}clinch_strike_defense'] = np.nan
                features[f'{prefix}ground_strike_defense'] = np.nan
                features[f'{prefix}distance_strike_defense'] = np.nan
                continue

            # Sort by date
            fighter_fights = fighter_fights.sort_values('event_date')

            # Rolling window features
            for window in [2, 3, 5]:
                recent_fights = fighter_fights.tail(window)

                if len(recent_fights) > 0:
                    # Win rate in window
                    wins = 0
                    finishes = 0
                    total_time = 0

                    for _, fight in recent_fights.iterrows():
                        if 'winner' in fight and pd.notna(fight['winner']):
                            if fight['winner'] == fighter_id:
                                wins += 1

                            # Check for finish
                            if fight.get('result') in ['KO/TKO', 'Submission']:
                                finishes += 1

                            # Fight time (approximate from finish round and time)
                            if pd.notna(fight.get('finish_round')) and pd.notna(fight.get('finish_time')):
                                try:
                                    round_num = int(fight['finish_round'])
                                    time_parts = str(fight['finish_time']).split(':')
                                    if len(time_parts) == 2:
                                        minutes = int(time_parts[0])
                                        seconds = int(time_parts[1])
                                        total_seconds = (round_num - 1) * 300 + minutes * 60 + seconds
                                        total_time += total_seconds
                                except:
                                    pass

                    features[f'{prefix}last_{window}_win_rate'] = wins / len(recent_fights)
                    features[f'{prefix}last_{window}_finish_rate'] = finishes / len(recent_fights)
                    if total_time > 0:
                        features[f'{prefix}last_{window}_avg_fight_time'] = total_time / len(recent_fights)
                    else:
                        features[f'{prefix}last_{window}_avg_fight_time'] = np.nan
                else:
                    features[f'{prefix}last_{window}_win_rate'] = np.nan
                    features[f'{prefix}last_{window}_finish_rate'] = np.nan
                    features[f'{prefix}last_{window}_avg_fight_time'] = np.nan

            # Round-level statistical features
            fighter_round_stats = self._get_fighter_round_stats(
                fighter_id, fighter_fights, rounds_df
            )

            # Existing features
            features[f'{prefix}avg_striking_accuracy'] = fighter_round_stats.get('striking_accuracy', np.nan)
            features[f'{prefix}avg_takedown_accuracy'] = fighter_round_stats.get('takedown_accuracy', np.nan)
            features[f'{prefix}avg_control_time'] = fighter_round_stats.get('control_time', np.nan)
            features[f'{prefix}avg_strikes_per_round'] = fighter_round_stats.get('strikes_per_round', np.nan)

            # Existing round stats features
            features[f'{prefix}takedown_defense'] = fighter_round_stats.get('takedown_defense', np.nan)
            features[f'{prefix}striking_defense'] = fighter_round_stats.get('striking_defense', np.nan)
            features[f'{prefix}strikes_landed_per_min'] = fighter_round_stats.get('strikes_landed_per_min', np.nan)
            features[f'{prefix}strikes_absorbed_per_min'] = fighter_round_stats.get('strikes_absorbed_per_min', np.nan)
            features[f'{prefix}knockdowns_per_fight'] = fighter_round_stats.get('knockdowns_per_fight', np.nan)
            features[f'{prefix}head_strike_pct'] = fighter_round_stats.get('head_strike_pct', np.nan)
            features[f'{prefix}body_strike_pct'] = fighter_round_stats.get('body_strike_pct', np.nan)
            features[f'{prefix}leg_strike_pct'] = fighter_round_stats.get('leg_strike_pct', np.nan)
            features[f'{prefix}avg_takedowns_attempted'] = fighter_round_stats.get('avg_takedowns_attempted', np.nan)
            features[f'{prefix}submission_attempts_per_fight'] = fighter_round_stats.get('submission_attempts_per_fight', np.nan)
            features[f'{prefix}reversals_per_fight'] = fighter_round_stats.get('reversals_per_fight', np.nan)

            # New absorption features
            features[f'{prefix}head_strikes_absorbed_per_min'] = fighter_round_stats.get('head_strikes_absorbed_per_min', np.nan)
            features[f'{prefix}body_strikes_absorbed_per_min'] = fighter_round_stats.get('body_strikes_absorbed_per_min', np.nan)
            features[f'{prefix}leg_strikes_absorbed_per_min'] = fighter_round_stats.get('leg_strikes_absorbed_per_min', np.nan)
            features[f'{prefix}head_absorption_rate'] = fighter_round_stats.get('head_absorption_rate', np.nan)
            features[f'{prefix}knockdowns_absorbed_per_fight'] = fighter_round_stats.get('knockdowns_absorbed_per_fight', np.nan)
            features[f'{prefix}knockdowns_absorbed_per_min'] = fighter_round_stats.get('knockdowns_absorbed_per_min', np.nan)
            features[f'{prefix}clinch_strike_defense'] = fighter_round_stats.get('clinch_strike_defense', np.nan)
            features[f'{prefix}ground_strike_defense'] = fighter_round_stats.get('ground_strike_defense', np.nan)
            features[f'{prefix}distance_strike_defense'] = fighter_round_stats.get('distance_strike_defense', np.nan)

            # Days since last fight
            if len(fighter_fights) > 0:
                last_fight_date = pd.to_datetime(fighter_fights.iloc[-1]['event_date'])
                current_fight_date = pd.to_datetime(current_fight['event_date'])
                days_since = (current_fight_date - last_fight_date).days
                features[f'{prefix}days_since_last_fight'] = days_since
            else:
                features[f'{prefix}days_since_last_fight'] = np.nan

        return features

    def _get_fighter_round_stats(
        self,
        fighter_id: str,
        fighter_fights: pd.DataFrame,
        rounds_df: pd.DataFrame
    ) -> Dict:
        """Calculate aggregate round statistics for a fighter."""
        if len(fighter_fights) == 0:
            return {
                'striking_accuracy': np.nan,
                'takedown_accuracy': np.nan,
                'control_time': np.nan,
                'strikes_per_round': np.nan,
                'takedown_defense': np.nan,
                'striking_defense': np.nan,
                'strikes_landed_per_min': np.nan,
                'strikes_absorbed_per_min': np.nan,
                'knockdowns_per_fight': np.nan,
                'head_strike_pct': np.nan,
                'body_strike_pct': np.nan,
                'leg_strike_pct': np.nan,
                'avg_takedowns_attempted': np.nan,
                'submission_attempts_per_fight': np.nan,
                'reversals_per_fight': np.nan
            }

        # Get all rounds for this fighter
        fight_ids = fighter_fights['fight_id'].tolist()
        fighter_rounds = rounds_df[
            (rounds_df['fight_id'].isin(fight_ids)) &
            (rounds_df['fighter_id'] == fighter_id)
        ]

        if len(fighter_rounds) == 0:
            return {
                'striking_accuracy': np.nan,
                'takedown_accuracy': np.nan,
                'control_time': np.nan,
                'strikes_per_round': np.nan,
                'takedown_defense': np.nan,
                'striking_defense': np.nan,
                'strikes_landed_per_min': np.nan,
                'strikes_absorbed_per_min': np.nan,
                'knockdowns_per_fight': np.nan,
                'head_strike_pct': np.nan,
                'body_strike_pct': np.nan,
                'leg_strike_pct': np.nan,
                'avg_takedowns_attempted': np.nan,
                'submission_attempts_per_fight': np.nan,
                'reversals_per_fight': np.nan
            }

        # Calculate existing averages
        total_strikes_att = fighter_rounds['strikes_att'].sum()
        total_strikes_succ = fighter_rounds['strikes_succ'].sum()
        striking_accuracy = total_strikes_succ / total_strikes_att if total_strikes_att > 0 else np.nan

        total_takedown_att = fighter_rounds['takedown_att'].sum()
        total_takedown_succ = fighter_rounds['takedown_succ'].sum()
        takedown_accuracy = total_takedown_succ / total_takedown_att if total_takedown_att > 0 else np.nan

        # Control time (convert from MM:SS to seconds)
        control_times = []
        for ctrl_time in fighter_rounds['ctrl_time']:
            if pd.notna(ctrl_time) and str(ctrl_time) != '0:00':
                try:
                    parts = str(ctrl_time).split(':')
                    if len(parts) == 2:
                        seconds = int(parts[0]) * 60 + int(parts[1])
                        control_times.append(seconds)
                except:
                    pass

        avg_control_time = np.mean(control_times) if control_times else np.nan
        avg_strikes_per_round = fighter_rounds['strikes_succ'].mean()

        # NEW FEATURES

        # 1. Takedown Defense (calculate from opponent's attempts against this fighter)
        # Get opponent rounds from same fights
        opponent_rounds = rounds_df[
            (rounds_df['fight_id'].isin(fight_ids)) &
            (rounds_df['fighter_id'] != fighter_id)
        ]
        if len(opponent_rounds) > 0:
            opp_takedown_att = opponent_rounds['takedown_att'].sum()
            opp_takedown_succ = opponent_rounds['takedown_succ'].sum()
            # Defense = blocked / attempted = (att - succ) / att
            takedown_defense = (opp_takedown_att - opp_takedown_succ) / opp_takedown_att if opp_takedown_att > 0 else np.nan
        else:
            takedown_defense = np.nan

        # 2. Striking Defense (calculate from opponent's strikes against this fighter)
        if len(opponent_rounds) > 0:
            opp_strikes_att = opponent_rounds['strikes_att'].sum()
            opp_strikes_succ = opponent_rounds['strikes_succ'].sum()
            # Defense = blocked / attempted = (att - succ) / att
            striking_defense = (opp_strikes_att - opp_strikes_succ) / opp_strikes_att if opp_strikes_att > 0 else np.nan
        else:
            striking_defense = np.nan

        # 3. Strikes Landed Per Minute (assuming 5-minute rounds)
        total_rounds = len(fighter_rounds)
        total_minutes = total_rounds * 5  # Each round is 5 minutes
        strikes_landed_per_min = total_strikes_succ / total_minutes if total_minutes > 0 else np.nan

        # 4. Strikes Absorbed Per Minute (opponent's successful strikes)
        if len(opponent_rounds) > 0:
            opp_total_strikes_succ = opponent_rounds['strikes_succ'].sum()
            strikes_absorbed_per_min = opp_total_strikes_succ / total_minutes if total_minutes > 0 else np.nan
        else:
            strikes_absorbed_per_min = np.nan

        # 5. Knockdowns Per Fight
        total_knockdowns = fighter_rounds['knockdowns'].sum()
        num_fights = len(fight_ids)
        knockdowns_per_fight = total_knockdowns / num_fights if num_fights > 0 else 0

        # 6. Strike Target Distribution (head/body/leg percentages)
        total_head_succ = fighter_rounds['head_strikes_succ'].sum()
        total_body_succ = fighter_rounds['body_strikes_succ'].sum()
        total_leg_succ = fighter_rounds['leg_strikes_succ'].sum()
        total_targeted_strikes = total_head_succ + total_body_succ + total_leg_succ

        head_strike_pct = total_head_succ / total_targeted_strikes if total_targeted_strikes > 0 else np.nan
        body_strike_pct = total_body_succ / total_targeted_strikes if total_targeted_strikes > 0 else np.nan
        leg_strike_pct = total_leg_succ / total_targeted_strikes if total_targeted_strikes > 0 else np.nan

        # 7. Average Takedowns Attempted Per Fight
        avg_takedowns_attempted = total_takedown_att / num_fights if num_fights > 0 else 0

        # 8. Submission Attempts Per Fight
        total_sub_attempts = fighter_rounds['submission_att'].sum()
        submission_attempts_per_fight = total_sub_attempts / num_fights if num_fights > 0 else 0

        # 9. Reversals Per Fight
        total_reversals = fighter_rounds['reversals'].sum()
        reversals_per_fight = total_reversals / num_fights if num_fights > 0 else 0

        # NEW ABSORPTION & DEFENSE FEATURES

        # 10. Strike absorption by zone (per minute)
        if len(opponent_rounds) > 0:
            opp_head_strikes = opponent_rounds['head_strikes_succ'].sum()
            opp_body_strikes = opponent_rounds['body_strikes_succ'].sum()
            opp_leg_strikes = opponent_rounds['leg_strikes_succ'].sum()

            head_strikes_absorbed_per_min = opp_head_strikes / total_minutes if total_minutes > 0 else np.nan
            body_strikes_absorbed_per_min = opp_body_strikes / total_minutes if total_minutes > 0 else np.nan
            leg_strikes_absorbed_per_min = opp_leg_strikes / total_minutes if total_minutes > 0 else np.nan

            # 11. Head absorption rate (% of absorbed strikes to head - KO vulnerability indicator)
            total_absorbed_strikes = opp_head_strikes + opp_body_strikes + opp_leg_strikes
            head_absorption_rate = opp_head_strikes / total_absorbed_strikes if total_absorbed_strikes > 0 else np.nan

            # 12. Knockdown absorption metrics
            opp_knockdowns = opponent_rounds['knockdowns'].sum()
            knockdowns_absorbed_per_fight = opp_knockdowns / num_fights if num_fights > 0 else 0
            knockdowns_absorbed_per_min = opp_knockdowns / total_minutes if total_minutes > 0 else np.nan

            # 13. Position-specific strike defense
            # Clinch defense
            opp_clinch_att = opponent_rounds['clinch_strikes_att'].sum()
            opp_clinch_succ = opponent_rounds['clinch_strikes_succ'].sum()
            clinch_strike_defense = (opp_clinch_att - opp_clinch_succ) / opp_clinch_att if opp_clinch_att > 0 else np.nan

            # Ground defense
            opp_ground_att = opponent_rounds['ground_strikes_att'].sum()
            opp_ground_succ = opponent_rounds['ground_strikes_succ'].sum()
            ground_strike_defense = (opp_ground_att - opp_ground_succ) / opp_ground_att if opp_ground_att > 0 else np.nan

            # Distance defense
            opp_distance_att = opponent_rounds['distance_strikes_att'].sum()
            opp_distance_succ = opponent_rounds['distance_strikes_succ'].sum()
            distance_strike_defense = (opp_distance_att - opp_distance_succ) / opp_distance_att if opp_distance_att > 0 else np.nan
        else:
            head_strikes_absorbed_per_min = np.nan
            body_strikes_absorbed_per_min = np.nan
            leg_strikes_absorbed_per_min = np.nan
            head_absorption_rate = np.nan
            knockdowns_absorbed_per_fight = 0
            knockdowns_absorbed_per_min = np.nan
            clinch_strike_defense = np.nan
            ground_strike_defense = np.nan
            distance_strike_defense = np.nan

        return {
            'striking_accuracy': striking_accuracy,
            'takedown_accuracy': takedown_accuracy,
            'control_time': avg_control_time,
            'strikes_per_round': avg_strikes_per_round,
            'takedown_defense': takedown_defense,
            'striking_defense': striking_defense,
            'strikes_landed_per_min': strikes_landed_per_min,
            'strikes_absorbed_per_min': strikes_absorbed_per_min,
            'knockdowns_per_fight': knockdowns_per_fight,
            'head_strike_pct': head_strike_pct,
            'body_strike_pct': body_strike_pct,
            'leg_strike_pct': leg_strike_pct,
            'avg_takedowns_attempted': avg_takedowns_attempted,
            'submission_attempts_per_fight': submission_attempts_per_fight,
            'reversals_per_fight': reversals_per_fight,
            # New absorption features
            'head_strikes_absorbed_per_min': head_strikes_absorbed_per_min,
            'body_strikes_absorbed_per_min': body_strikes_absorbed_per_min,
            'leg_strikes_absorbed_per_min': leg_strikes_absorbed_per_min,
            'head_absorption_rate': head_absorption_rate,
            'knockdowns_absorbed_per_fight': knockdowns_absorbed_per_fight,
            'knockdowns_absorbed_per_min': knockdowns_absorbed_per_min,
            'clinch_strike_defense': clinch_strike_defense,
            'ground_strike_defense': ground_strike_defense,
            'distance_strike_defense': distance_strike_defense
        }

    def _create_damage_history_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame
    ) -> Dict:
        """Create damage accumulation and chin erosion features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights (before current fight)
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                # No historical data - set defaults
                features[f'{prefix}recent_knockdowns_absorbed_l3'] = 0
                features[f'{prefix}recent_ko_losses'] = 0
                features[f'{prefix}recent_sub_losses'] = 0
                features[f'{prefix}cumulative_head_strikes_absorbed'] = np.nan
                features[f'{prefix}days_since_last_ko_loss'] = np.nan
                continue

            # Sort by date
            fighter_fights = fighter_fights.sort_values('event_date')

            # 1. Recent knockdowns absorbed (last 3 fights)
            last_3_fights = fighter_fights.tail(3)
            recent_knockdowns = 0
            for _, fight in last_3_fights.iterrows():
                fight_id = fight['fight_id']
                # Get opponent's knockdowns in this fight
                opponent_rounds = rounds_df[
                    (rounds_df['fight_id'] == fight_id) &
                    (rounds_df['fighter_id'] != fighter_id)
                ]
                if len(opponent_rounds) > 0:
                    recent_knockdowns += opponent_rounds['knockdowns'].sum()
            features[f'{prefix}recent_knockdowns_absorbed_l3'] = recent_knockdowns

            # 2. Recent KO/TKO losses (last 5 fights)
            last_5_fights = fighter_fights.tail(5)
            ko_losses = 0
            for _, fight in last_5_fights.iterrows():
                if 'result' in fight and 'winner' in fight:
                    if fight['result'] in ['KO/TKO'] and fight['winner'] != fighter_id:
                        ko_losses += 1
            features[f'{prefix}recent_ko_losses'] = ko_losses

            # 3. Recent submission losses (last 5 fights)
            sub_losses = 0
            for _, fight in last_5_fights.iterrows():
                if 'result' in fight and 'winner' in fight:
                    if fight['result'] == 'Submission' and fight['winner'] != fighter_id:
                        sub_losses += 1
            features[f'{prefix}recent_sub_losses'] = sub_losses

            # 4. Cumulative head strikes absorbed (career)
            fight_ids = fighter_fights['fight_id'].tolist()
            opponent_rounds = rounds_df[
                (rounds_df['fight_id'].isin(fight_ids)) &
                (rounds_df['fighter_id'] != fighter_id)
            ]
            if len(opponent_rounds) > 0:
                cumulative_head_strikes = opponent_rounds['head_strikes_succ'].sum()
                features[f'{prefix}cumulative_head_strikes_absorbed'] = cumulative_head_strikes
            else:
                features[f'{prefix}cumulative_head_strikes_absorbed'] = np.nan

            # 5. Days since last KO/TKO loss
            ko_loss_fights = fighter_fights[
                (fighter_fights['result'] == 'KO/TKO') &
                (fighter_fights['winner'] != fighter_id)
            ]
            if len(ko_loss_fights) > 0:
                last_ko_loss_date = pd.to_datetime(ko_loss_fights.iloc[-1]['event_date'])
                current_date = pd.to_datetime(current_fight['event_date'])
                days_since = (current_date - last_ko_loss_date).days
                features[f'{prefix}days_since_last_ko_loss'] = days_since
            else:
                features[f'{prefix}days_since_last_ko_loss'] = np.nan

        return features

    def _create_pace_sustainability_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame
    ) -> Dict:
        """Create pace decay and cardio sustainability features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                features[f'{prefix}round_1_strike_output'] = np.nan
                features[f'{prefix}round_2_strike_output'] = np.nan
                features[f'{prefix}round_3_plus_strike_output'] = np.nan
                features[f'{prefix}output_decay_rate'] = np.nan
                features[f'{prefix}strike_rate_delta_r1_to_r3'] = np.nan
                features[f'{prefix}pace_consistency_score'] = np.nan
                continue

            # Get all rounds for this fighter
            fight_ids = fighter_fights['fight_id'].tolist()
            fighter_rounds = rounds_df[
                (rounds_df['fight_id'].isin(fight_ids)) &
                (rounds_df['fighter_id'] == fighter_id)
            ]

            if len(fighter_rounds) == 0:
                features[f'{prefix}round_1_strike_output'] = np.nan
                features[f'{prefix}round_2_strike_output'] = np.nan
                features[f'{prefix}round_3_plus_strike_output'] = np.nan
                features[f'{prefix}output_decay_rate'] = np.nan
                features[f'{prefix}strike_rate_delta_r1_to_r3'] = np.nan
                features[f'{prefix}pace_consistency_score'] = np.nan
                continue

            # Group by round number
            r1_rounds = fighter_rounds[fighter_rounds['round'] == 1]
            r2_rounds = fighter_rounds[fighter_rounds['round'] == 2]
            r3_plus_rounds = fighter_rounds[fighter_rounds['round'] >= 3]

            # 1. Average strike output by round
            round_1_output = r1_rounds['strikes_succ'].mean() if len(r1_rounds) > 0 else np.nan
            round_2_output = r2_rounds['strikes_succ'].mean() if len(r2_rounds) > 0 else np.nan
            round_3_plus_output = r3_plus_rounds['strikes_succ'].mean() if len(r3_plus_rounds) > 0 else np.nan

            features[f'{prefix}round_1_strike_output'] = round_1_output
            features[f'{prefix}round_2_strike_output'] = round_2_output
            features[f'{prefix}round_3_plus_strike_output'] = round_3_plus_output

            # 2. Output decay rate (percentage drop from R1 to R3+)
            if pd.notna(round_1_output) and pd.notna(round_3_plus_output) and round_1_output > 0:
                decay_rate = (round_1_output - round_3_plus_output) / round_1_output
                features[f'{prefix}output_decay_rate'] = decay_rate
            else:
                features[f'{prefix}output_decay_rate'] = np.nan

            # 3. Strike rate delta (strikes per minute change from R1 to R3)
            # Each round is 5 minutes
            if pd.notna(round_1_output) and pd.notna(round_3_plus_output):
                r1_per_min = round_1_output / 5.0
                r3_per_min = round_3_plus_output / 5.0
                features[f'{prefix}strike_rate_delta_r1_to_r3'] = r1_per_min - r3_per_min
            else:
                features[f'{prefix}strike_rate_delta_r1_to_r3'] = np.nan

            # 4. Pace consistency score (inverse of variance across rounds)
            # Lower variance = more consistent pace
            round_outputs = []
            if pd.notna(round_1_output):
                round_outputs.append(round_1_output)
            if pd.notna(round_2_output):
                round_outputs.append(round_2_output)
            if pd.notna(round_3_plus_output):
                round_outputs.append(round_3_plus_output)

            if len(round_outputs) >= 2:
                variance = np.var(round_outputs)
                # Normalize by mean to get coefficient of variation
                mean_output = np.mean(round_outputs)
                if mean_output > 0:
                    consistency = 1 / (1 + variance / mean_output)  # Higher = more consistent
                    features[f'{prefix}pace_consistency_score'] = consistency
                else:
                    features[f'{prefix}pace_consistency_score'] = np.nan
            else:
                features[f'{prefix}pace_consistency_score'] = np.nan

        return features

    def _create_late_round_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame
    ) -> Dict:
        """Create late-round performance and cardio mismatch features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                features[f'{prefix}win_rate_rounds_3_plus'] = np.nan
                features[f'{prefix}finish_rate_late_rounds'] = np.nan
                features[f'{prefix}late_round_win_rate_high_pace'] = np.nan
                features[f'{prefix}late_round_win_rate_low_pace'] = np.nan
                continue

            # Sort by date
            fighter_fights = fighter_fights.sort_values('event_date')

            # 1. Win rate in fights that reached round 3+
            long_fights = fighter_fights[fighter_fights['num_rounds'] >= 3]
            if len(long_fights) > 0:
                wins_in_long_fights = 0
                for _, fight in long_fights.iterrows():
                    if 'winner' in fight and pd.notna(fight['winner']):
                        if fight['winner'] == fighter_id:
                            wins_in_long_fights += 1
                features[f'{prefix}win_rate_rounds_3_plus'] = wins_in_long_fights / len(long_fights)
            else:
                features[f'{prefix}win_rate_rounds_3_plus'] = np.nan

            # 2. Finish rate in late rounds (R3+)
            finished_fights = fighter_fights[
                (fighter_fights['result'].isin(['KO/TKO', 'Submission'])) &
                (fighter_fights['winner'] == fighter_id)
            ]
            if len(finished_fights) > 0:
                late_finishes = len(finished_fights[finished_fights['finish_round'] >= 3])
                features[f'{prefix}finish_rate_late_rounds'] = late_finishes / len(finished_fights)
            else:
                features[f'{prefix}finish_rate_late_rounds'] = np.nan

            # 3. Late-round win rate conditional on R1 pace
            # Get fighter's round stats
            fight_ids = fighter_fights['fight_id'].tolist()
            fighter_rounds = rounds_df[
                (rounds_df['fight_id'].isin(fight_ids)) &
                (rounds_df['fighter_id'] == fighter_id)
            ]

            if len(fighter_rounds) > 0:
                # Calculate median R1 output
                r1_rounds = fighter_rounds[fighter_rounds['round'] == 1]
                if len(r1_rounds) > 0:
                    median_r1_output = r1_rounds['strikes_succ'].median()

                    # Classify each fight as high or low R1 pace
                    high_pace_fights = []
                    low_pace_fights = []

                    for _, fight in long_fights.iterrows():
                        fight_r1 = r1_rounds[r1_rounds['fight_id'] == fight['fight_id']]
                        if len(fight_r1) > 0:
                            r1_output = fight_r1.iloc[0]['strikes_succ']
                            if r1_output >= median_r1_output:
                                high_pace_fights.append(fight)
                            else:
                                low_pace_fights.append(fight)

                    # Win rate in high-pace long fights
                    if len(high_pace_fights) > 0:
                        wins_high_pace = sum(
                            1 for fight in high_pace_fights
                            if 'winner' in fight and fight['winner'] == fighter_id
                        )
                        features[f'{prefix}late_round_win_rate_high_pace'] = wins_high_pace / len(high_pace_fights)
                    else:
                        features[f'{prefix}late_round_win_rate_high_pace'] = np.nan

                    # Win rate in low-pace long fights
                    if len(low_pace_fights) > 0:
                        wins_low_pace = sum(
                            1 for fight in low_pace_fights
                            if 'winner' in fight and fight['winner'] == fighter_id
                        )
                        features[f'{prefix}late_round_win_rate_low_pace'] = wins_low_pace / len(low_pace_fights)
                    else:
                        features[f'{prefix}late_round_win_rate_low_pace'] = np.nan
                else:
                    features[f'{prefix}late_round_win_rate_high_pace'] = np.nan
                    features[f'{prefix}late_round_win_rate_low_pace'] = np.nan
            else:
                features[f'{prefix}late_round_win_rate_high_pace'] = np.nan
                features[f'{prefix}late_round_win_rate_low_pace'] = np.nan

        return features

    def _create_fatigue_defense_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame
    ) -> Dict:
        """Create takedown defense under fatigue features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                features[f'{prefix}td_defense_round_1'] = np.nan
                features[f'{prefix}td_defense_rounds_2_3'] = np.nan
                features[f'{prefix}td_defense_degradation'] = np.nan
                continue

            # Get all rounds for this fighter
            fight_ids = fighter_fights['fight_id'].tolist()

            # Get opponent's takedown attempts against this fighter
            opponent_rounds = rounds_df[
                (rounds_df['fight_id'].isin(fight_ids)) &
                (rounds_df['fighter_id'] != fighter_id)
            ]

            if len(opponent_rounds) == 0:
                features[f'{prefix}td_defense_round_1'] = np.nan
                features[f'{prefix}td_defense_rounds_2_3'] = np.nan
                features[f'{prefix}td_defense_degradation'] = np.nan
                continue

            # Split by round number
            r1_opponent = opponent_rounds[opponent_rounds['round'] == 1]
            r2_3_opponent = opponent_rounds[opponent_rounds['round'].isin([2, 3])]

            # 1. TD defense in round 1
            if len(r1_opponent) > 0:
                r1_td_att = r1_opponent['takedown_att'].sum()
                r1_td_succ = r1_opponent['takedown_succ'].sum()
                td_defense_r1 = (r1_td_att - r1_td_succ) / r1_td_att if r1_td_att > 0 else np.nan
                features[f'{prefix}td_defense_round_1'] = td_defense_r1
            else:
                features[f'{prefix}td_defense_round_1'] = np.nan
                td_defense_r1 = np.nan

            # 2. TD defense in rounds 2-3 (fatigue rounds)
            if len(r2_3_opponent) > 0:
                r2_3_td_att = r2_3_opponent['takedown_att'].sum()
                r2_3_td_succ = r2_3_opponent['takedown_succ'].sum()
                td_defense_r2_3 = (r2_3_td_att - r2_3_td_succ) / r2_3_td_att if r2_3_td_att > 0 else np.nan
                features[f'{prefix}td_defense_rounds_2_3'] = td_defense_r2_3
            else:
                features[f'{prefix}td_defense_rounds_2_3'] = np.nan
                td_defense_r2_3 = np.nan

            # 3. TD defense degradation (fatigue impact)
            # Positive = defense gets worse in later rounds
            if pd.notna(td_defense_r1) and pd.notna(td_defense_r2_3):
                degradation = td_defense_r1 - td_defense_r2_3
                features[f'{prefix}td_defense_degradation'] = degradation
            else:
                features[f'{prefix}td_defense_degradation'] = np.nan

        return features

    def _create_matchup_features(self, fighter_1: pd.Series, fighter_2: pd.Series) -> Dict:
        """Create matchup-specific features."""
        features = {}

        # Style indicators (simplified)
        f1_stance = fighter_1.get('fighter_stance', 'Orthodox')
        f2_stance = fighter_2.get('fighter_stance', 'Orthodox')

        # Stance matchup
        features['orthodox_vs_southpaw'] = 1 if (
            (f1_stance == 'Orthodox' and f2_stance == 'Southpaw') or
            (f1_stance == 'Southpaw' and f2_stance == 'Orthodox')
        ) else 0

        return features

    def _create_momentum_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame
    ) -> Dict:
        """Create advanced momentum and streak features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                features[f'{prefix}win_streak'] = 0
                features[f'{prefix}loss_streak'] = 0
                features[f'{prefix}current_streak'] = 0
                features[f'{prefix}momentum_score'] = 0
                features[f'{prefix}recent_activity'] = 0
                features[f'{prefix}fights_per_year'] = 0
                continue

            # Sort by date
            fighter_fights = fighter_fights.sort_values('event_date')

            # Calculate win/loss streaks
            recent_results = []
            win_streak = 0
            loss_streak = 0

            for _, fight in fighter_fights.iterrows():
                if 'winner' in fight and pd.notna(fight['winner']):
                    won = fight['winner'] == fighter_id
                    recent_results.append(1 if won else 0)

                    if won:
                        loss_streak = 0
                        win_streak += 1
                    else:
                        win_streak = 0
                        loss_streak += 1

            features[f'{prefix}win_streak'] = win_streak
            features[f'{prefix}loss_streak'] = loss_streak
            features[f'{prefix}current_streak'] = win_streak if win_streak > 0 else -loss_streak

            # Momentum score (weighted recent performance)
            if len(recent_results) >= 3:
                last_5 = recent_results[-5:] if len(recent_results) >= 5 else recent_results
                weights = [0.4, 0.3, 0.2, 0.1] if len(last_5) == 4 else [0.5, 0.3, 0.2] if len(last_5) == 3 else [1.0, 1.0]
                momentum = sum(w * r for w, r in zip(weights[:len(last_5)], last_5))
                features[f'{prefix}momentum_score'] = momentum
            else:
                features[f'{prefix}momentum_score'] = np.nan

            # Activity level
            if len(fighter_fights) > 1:
                first_fight = pd.to_datetime(fighter_fights.iloc[0]['event_date'])
                last_fight = pd.to_datetime(fighter_fights.iloc[-1]['event_date'])
                career_years = max((last_fight - first_fight).days / 365.25, 0.5)
                features[f'{prefix}fights_per_year'] = len(fighter_fights) / career_years

                # Recent activity (fights in last 2 years)
                current_date = pd.to_datetime(current_fight['event_date'])
                recent_fights = fighter_fights[
                    pd.to_datetime(fighter_fights['event_date']) >= (current_date - pd.Timedelta(days=730))
                ]
                features[f'{prefix}recent_activity'] = len(recent_fights)
            else:
                features[f'{prefix}fights_per_year'] = 0
                features[f'{prefix}recent_activity'] = 1

        return features

    def _create_style_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame
    ) -> Dict:
        """Create fighting style and method features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                features[f'{prefix}ko_rate'] = np.nan
                features[f'{prefix}submission_rate'] = np.nan
                features[f'{prefix}decision_rate'] = np.nan
                features[f'{prefix}first_round_finish_rate'] = np.nan
                features[f'{prefix}late_finish_rate'] = np.nan
                features[f'{prefix}average_fight_duration'] = np.nan
                # REMOVED: takedown_success_rate - 100% duplicate of avg_takedown_accuracy
                continue

            total_fights = len(fighter_fights)
            ko_wins = 0
            sub_wins = 0
            decision_wins = 0
            first_round_finishes = 0
            late_finishes = 0
            total_duration = 0
            valid_durations = 0

            for _, fight in fighter_fights.iterrows():
                if 'winner' in fight and pd.notna(fight['winner']) and fight['winner'] == fighter_id:
                    result = fight.get('result', '')

                    if 'KO' in str(result) or 'TKO' in str(result):
                        ko_wins += 1
                    elif 'Submission' in str(result):
                        sub_wins += 1
                    elif 'Decision' in str(result):
                        decision_wins += 1

                    # Round analysis
                    finish_round = fight.get('finish_round')
                    if pd.notna(finish_round):
                        if int(finish_round) == 1:
                            first_round_finishes += 1
                        elif int(finish_round) >= 3:
                            late_finishes += 1

                    # Fight duration
                    if pd.notna(fight.get('finish_round')) and pd.notna(fight.get('finish_time')):
                        try:
                            round_num = int(fight['finish_round'])
                            time_parts = str(fight['finish_time']).split(':')
                            if len(time_parts) == 2:
                                minutes = int(time_parts[0])
                                seconds = int(time_parts[1])
                                duration = (round_num - 1) * 300 + minutes * 60 + seconds
                                total_duration += duration
                                valid_durations += 1
                        except:
                            pass

            # Calculate rates
            features[f'{prefix}ko_rate'] = ko_wins / total_fights if total_fights > 0 else 0
            features[f'{prefix}submission_rate'] = sub_wins / total_fights if total_fights > 0 else 0
            features[f'{prefix}decision_rate'] = decision_wins / total_fights if total_fights > 0 else 0
            features[f'{prefix}first_round_finish_rate'] = first_round_finishes / total_fights if total_fights > 0 else 0
            features[f'{prefix}late_finish_rate'] = late_finishes / total_fights if total_fights > 0 else 0
            features[f'{prefix}average_fight_duration'] = total_duration / valid_durations if valid_durations > 0 else np.nan

            # Takedown defense approximation from rounds data
            fighter_rounds = rounds_df[
                (rounds_df['fight_id'].isin(fighter_fights['fight_id'])) &
                (rounds_df['fighter_id'] == fighter_id)
            ]

            # REMOVED: takedown_success_rate calculation
            # This was a 100% duplicate of avg_takedown_accuracy (from round stats above)
            # Keeping avg_takedown_accuracy as it's already calculated and more descriptive

        return features

    def _create_elo_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame
    ) -> Dict:
        """
        Create Elo rating system features (combat-adapted).

        Features:
        - Current Elo rating for each fighter
        - Elo delta (difference between fighters)
        - Elo uncertainty (Glicko-inspired rating deviation)
        - Peak Elo (career high rating)
        - Elo trend (momentum in ratings over last 3 fights)
        """
        features = {}

        # Elo parameters (combat-adapted)
        K_BASE = 32  # Base K-factor
        K_FINISH_BONUS = 16  # Extra K for finishes
        INITIAL_ELO = 1500
        INITIAL_RD = 350  # Rating deviation (uncertainty)

        current_fight_date = pd.to_datetime(current_fight['event_date'])
        fighter_1_id = current_fight['fighter_1']
        fighter_2_id = current_fight['fighter_2']

        # Calculate Elo for each fighter
        for i, (fighter_id, fighter_data) in enumerate([
            (fighter_1_id, fighter_1),
            (fighter_2_id, fighter_2)
        ], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights (before current fight)
            fighter_history = all_fights[
                ((all_fights['fighter_1'] == fighter_id) | (all_fights['fighter_2'] == fighter_id)) &
                (pd.to_datetime(all_fights['event_date']) < current_fight_date)
            ].copy()

            if len(fighter_history) == 0:
                # Debut fighter - use initial ratings
                features[f'{prefix}elo'] = INITIAL_ELO
                features[f'{prefix}elo_uncertainty'] = INITIAL_RD
                features[f'{prefix}peak_elo'] = INITIAL_ELO
                features[f'{prefix}elo_trend'] = 0.0
                continue

            # Sort by date
            fighter_history = fighter_history.sort_values('event_date')

            # Calculate Elo progression through all fights
            current_elo = INITIAL_ELO
            peak_elo = INITIAL_ELO
            elo_history = [INITIAL_ELO]
            rd = INITIAL_RD  # Rating deviation

            for _, hist_fight in fighter_history.iterrows():
                # Determine if this fighter won
                is_fighter_1 = hist_fight['fighter_1'] == fighter_id

                if 'winner' not in hist_fight or pd.isna(hist_fight['winner']):
                    continue

                won = (is_fighter_1 and hist_fight['winner'] == hist_fight['fighter_1']) or \
                      (not is_fighter_1 and hist_fight['winner'] == hist_fight['fighter_2'])

                # Calculate expected score (simplified Elo)
                # We don't have opponent Elo, so assume average opponent (1500)
                expected = 1 / (1 + 10 ** ((1500 - current_elo) / 400))

                # Actual score
                actual = 1.0 if won else 0.0

                # K-factor adjustments
                k = K_BASE

                # Finish bonus (more impressive wins = bigger rating change)
                if hist_fight.get('result') in ['KO/TKO', 'Submission']:
                    k += K_FINISH_BONUS

                # Uncertainty adjustment (higher RD = more volatile ratings)
                k_adjusted = k * (rd / INITIAL_RD)

                # Update Elo
                current_elo = current_elo + k_adjusted * (actual - expected)

                # Track peak
                peak_elo = max(peak_elo, current_elo)

                # Update rating deviation (decreases with more fights)
                rd = max(50, rd * 0.95)  # Decrease uncertainty over time

                # Store for trend calculation
                elo_history.append(current_elo)

            # Calculate Elo trend (momentum over last 3 fights)
            elo_trend = 0.0
            if len(elo_history) >= 4:
                recent_elos = elo_history[-4:]
                elo_trend = recent_elos[-1] - recent_elos[0]  # Change over last 3 fights

            # Store features
            features[f'{prefix}elo'] = current_elo
            features[f'{prefix}elo_uncertainty'] = rd
            features[f'{prefix}peak_elo'] = peak_elo
            features[f'{prefix}elo_trend'] = elo_trend

        # Elo delta (key feature - difference in ratings)
        features['elo_delta'] = features['fighter_1_elo'] - features['fighter_2_elo']

        # Elo confidence gap (considers uncertainty)
        f1_upper = features['fighter_1_elo'] + features['fighter_1_elo_uncertainty']
        f1_lower = features['fighter_1_elo'] - features['fighter_1_elo_uncertainty']
        f2_upper = features['fighter_2_elo'] + features['fighter_2_elo_uncertainty']
        f2_lower = features['fighter_2_elo'] - features['fighter_2_elo_uncertainty']

        # Overlap indicates uncertain matchup
        overlap = min(f1_upper, f2_upper) - max(f1_lower, f2_lower)
        features['elo_overlap'] = max(0, overlap)

        # Peak Elo difference
        features['peak_elo_delta'] = features['fighter_1_peak_elo'] - features['fighter_2_peak_elo']

        # Momentum comparison (trend difference)
        features['elo_momentum_delta'] = features['fighter_1_elo_trend'] - features['fighter_2_elo_trend']

        return features

    def _create_career_stage_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        STRETCH: int = 5
    ) -> Dict:
        """Create career stage and prime analysis features.
        
        Parameters
        ----------
        current_fight : pd.Series
            Row with current fight details.
        fighter_1 : pd.Series
            Metadata for fighter 1 (DOB, etc.).
        fighter_2 : pd.Series
            Metadata for fighter 2 (DOB, etc.).
        all_fights : pd.DataFrame
            Complete fight history for all fighters.
        STRETCH : int, optional
            Default window size for peak performance. Defaults to 5.
        
        Returns
        -------
        Dict
            Dictionary of career stage features for both fighters.
        """
        features = {}
        current_date = pd.to_datetime(current_fight['event_date'])

        for i, fighter_data in enumerate([fighter_1, fighter_2], 1):
            prefix = f'fighter_{i}_'
            fighter_id = current_fight[f'fighter_{i}']

            # Pull all fights for this fighter
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy().sort_values('event_date')

            total_fights = len(fighter_fights)

            # --- Fighting age ---
            if total_fights > 0:
                debut_date = pd.to_datetime(fighter_fights.iloc[0]['event_date'])
                fighting_age = (current_date - debut_date).days / 365.25
                features[f'{prefix}fighting_age'] = fighting_age
            else:
                features[f'{prefix}fighting_age'] = 0

            # --- Peak performance ---
            if total_fights >= 3:
                window = min(total_fights, STRETCH)
                best_stretch = 0
                for start in range(total_fights - (window - 1)):
                    stretch_fights = fighter_fights.iloc[start:start+window]
                    wins = sum(1 for _, f in stretch_fights.iterrows()
                            if f.get('winner') == fighter_id)
                    best_stretch = max(best_stretch, wins)
                features[f'{prefix}peak_performance'] = best_stretch / window
            elif total_fights in [1, 2]:
                features[f'{prefix}peak_performance'] = 0
            else:
                features[f'{prefix}peak_performance'] = 0

            # --- Physical age / prime analysis ---
            if pd.notna(fighter_data.get('fighter_dob')):
                dob = pd.to_datetime(fighter_data['fighter_dob'])
                age = (current_date - dob).days / 365.25

                weight_class = str(current_fight.get('weight_class', ''))
                if 'Heavyweight' in weight_class:
                    prime_age = 32
                elif any(wc in weight_class for wc in ['Light Heavyweight', 'Middleweight']):
                    prime_age = 30
                else:
                    prime_age = 28

                features[f'{prefix}age_vs_prime'] = age - prime_age
                features[f'{prefix}past_prime'] = 1 if age > prime_age + 3 else 0
            else:
                features[f'{prefix}age_vs_prime'] = np.nan
                features[f'{prefix}past_prime'] = np.nan

        return features

    def _create_advanced_matchup_features(
        self,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame = None,
        current_fight: pd.Series = None
    ) -> Dict:
        """Create advanced matchup and style clash features."""
        features = {}

        # Style clash indicators
        f1_stance = fighter_1.get('fighter_stance', 'Orthodox')
        f2_stance = fighter_2.get('fighter_stance', 'Orthodox')

        # Southpaw advantage
        features['southpaw_advantage'] = 1 if (
            f1_stance == 'Southpaw' and f2_stance == 'Orthodox'
        ) else -1 if (
            f1_stance == 'Orthodox' and f2_stance == 'Southpaw'
        ) else 0

        # Physical advantages categorized
        height_diff = fighter_1.get('fighter_height_cm', 0) - fighter_2.get('fighter_height_cm', 0)
        reach_diff = fighter_1.get('fighter_reach_cm', 0) - fighter_2.get('fighter_reach_cm', 0)

        features['significant_height_advantage'] = 1 if height_diff > 7.5 else -1 if height_diff < -7.5 else 0
        features['significant_reach_advantage'] = 1 if reach_diff > 12.5 else -1 if reach_diff < -12.5 else 0

        # REMOVED: Duplicate absolute value features that conflict with signed differentials
        # - win_rate_gap (absolute) conflicted with record_quality_difference (signed)
        # - major_experience_gap (absolute) conflicted with experience_advantage (signed)
        # These caused artificial feature importance where each fighter got "their own" feature
        # during data augmentation, creating biased predictions

        # REMOVED: record_quality_difference - was dominating all predictions
        # This feature had 0.324 correlation with winner (way too high)
        # It was the #1 SHAP feature in every single prediction, suppressing other features
        # Keep individual win percentages in fighter records instead

        # NEW: Enhanced style matchup features (Phase 6)
        if rounds_df is not None and current_fight is not None:
            # Get historical stats for both fighters
            f1_stats = self._get_fighter_style_stats(current_fight['fighter_1'], all_fights, rounds_df, current_fight)
            f2_stats = self._get_fighter_style_stats(current_fight['fighter_2'], all_fights, rounds_df, current_fight)

            # 1. Southpaw vs Orthodox head strike advantage (differential)
            if f1_stance == 'Southpaw' and f2_stance == 'Orthodox':
                # Fighter 1 has southpaw advantage - calculate differential
                f1_head_rate = f1_stats.get('head_strike_pct', 0) or 0
                f2_head_rate = f2_stats.get('head_strike_pct', 0) or 0
                features['southpaw_vs_orthodox_head_strike_advantage'] = f1_head_rate - f2_head_rate
            elif f1_stance == 'Orthodox' and f2_stance == 'Southpaw':
                # Fighter 2 has southpaw advantage
                f2_head_rate = f2_stats.get('head_strike_pct', 0) or 0
                f1_head_rate = f1_stats.get('head_strike_pct', 0) or 0
                features['southpaw_vs_orthodox_head_strike_advantage'] = f1_head_rate - f2_head_rate
            else:
                features['southpaw_vs_orthodox_head_strike_advantage'] = 0

            # 2. Striker anti-wrestling score (F1 strike defense vs F2 TD offense)
            f1_strike_def = f1_stats.get('striking_defense', 0) or 0
            f2_td_acc = f2_stats.get('takedown_accuracy', 0) or 0
            f2_strike_def = f2_stats.get('striking_defense', 0) or 0
            f1_td_acc = f1_stats.get('takedown_accuracy', 0) or 0
            features['striker_anti_wrestling_score'] = (f1_strike_def - f2_td_acc) - (f2_strike_def - f1_td_acc)

            # 3. Submission threat differential
            f1_sub_rate = f1_stats.get('submission_rate', 0) or 0
            f2_sub_rate = f2_stats.get('submission_rate', 0) or 0
            features['submission_threat_differential'] = f1_sub_rate - f2_sub_rate

            # 4. Grappler vs striker mismatch (TD accuracy differential)
            features['grappler_vs_striker_mismatch'] = f1_td_acc - f2_td_acc

            # 5. Clinch fighter vs distance fighter (clinch strike rate differential)
            f1_clinch_rate = f1_stats.get('clinch_strike_rate', 0) or 0
            f2_clinch_rate = f2_stats.get('clinch_strike_rate', 0) or 0
            f1_distance_rate = f1_stats.get('distance_strike_rate', 0) or 0
            f2_distance_rate = f2_stats.get('distance_strike_rate', 0) or 0
            features['clinch_fighter_vs_distance_fighter'] = (f1_clinch_rate - f1_distance_rate) - (f2_clinch_rate - f2_distance_rate)

            # 6. Pressure style effectiveness (strike volume vs opponent defense)
            f1_strike_volume = f1_stats.get('strikes_landed_per_min', 0) or 0
            f2_strike_volume = f2_stats.get('strikes_landed_per_min', 0) or 0
            features['pressure_style_effectiveness'] = (f1_strike_volume - f2_strike_def) - (f2_strike_volume - f1_strike_def)

            # 7. Defensive specialist score (defensive rating differential)
            f1_td_def = f1_stats.get('takedown_defense', 0) or 0
            f2_td_def = f2_stats.get('takedown_defense', 0) or 0
            f1_defensive = (f1_strike_def + f1_td_def) / 2
            f2_defensive = (f2_strike_def + f2_td_def) / 2
            features['defensive_specialist_score'] = f1_defensive - f2_defensive

            # 8. Finishing threat differential (finish rate comparison)
            f1_ko_rate = f1_stats.get('ko_rate', 0) or 0
            f2_ko_rate = f2_stats.get('ko_rate', 0) or 0
            f1_finish_rate = f1_ko_rate + f1_sub_rate
            f2_finish_rate = f2_ko_rate + f2_sub_rate
            features['finishing_threat_differential'] = f1_finish_rate - f2_finish_rate

        else:
            # No round data available - set defaults
            features['southpaw_vs_orthodox_head_strike_advantage'] = 0
            features['striker_anti_wrestling_score'] = 0
            features['submission_threat_differential'] = 0
            features['grappler_vs_striker_mismatch'] = 0
            features['clinch_fighter_vs_distance_fighter'] = 0
            features['pressure_style_effectiveness'] = 0
            features['defensive_specialist_score'] = 0
            features['finishing_threat_differential'] = 0

        return features

    def _get_fighter_style_stats(
        self,
        fighter_id: str,
        all_fights: pd.DataFrame,
        rounds_df: pd.DataFrame,
        current_fight: pd.Series
    ) -> Dict:
        """
        Calculate style statistics for a fighter from their historical data.
        Used by enhanced matchup features.
        """
        # Get fighter's historical fights (before current fight)
        fighter_fights = all_fights[
            ((all_fights['fighter_1'] == fighter_id) | (all_fights['fighter_2'] == fighter_id)) &
            (pd.to_datetime(all_fights['event_date']) < pd.to_datetime(current_fight['event_date']))
        ].copy()

        if len(fighter_fights) == 0:
            return {
                'head_strike_pct': 0,
                'striking_defense': 0,
                'takedown_accuracy': 0,
                'takedown_defense': 0,
                'submission_rate': 0,
                'ko_rate': 0,
                'clinch_strike_rate': 0,
                'distance_strike_rate': 0,
                'strikes_landed_per_min': 0
            }

        # Get round data
        fight_ids = fighter_fights['fight_id'].tolist()
        fighter_rounds = rounds_df[
            (rounds_df['fight_id'].isin(fight_ids)) &
            (rounds_df['fighter_id'] == fighter_id)
        ]

        opponent_rounds = rounds_df[
            (rounds_df['fight_id'].isin(fight_ids)) &
            (rounds_df['fighter_id'] != fighter_id)
        ]

        stats = {}

        # Calculate statistics from round data
        if len(fighter_rounds) > 0:
            # Head strike percentage
            total_head = fighter_rounds['head_strikes_succ'].sum()
            total_body = fighter_rounds['body_strikes_succ'].sum()
            total_leg = fighter_rounds['leg_strikes_succ'].sum()
            total_strikes = total_head + total_body + total_leg
            stats['head_strike_pct'] = total_head / total_strikes if total_strikes > 0 else 0

            # Striking defense (from opponent rounds)
            if len(opponent_rounds) > 0:
                opp_strikes_att = opponent_rounds['strikes_att'].sum()
                opp_strikes_succ = opponent_rounds['strikes_succ'].sum()
                stats['striking_defense'] = (opp_strikes_att - opp_strikes_succ) / opp_strikes_att if opp_strikes_att > 0 else 0
            else:
                stats['striking_defense'] = 0

            # Takedown accuracy
            total_td_att = fighter_rounds['takedown_att'].sum()
            total_td_succ = fighter_rounds['takedown_succ'].sum()
            stats['takedown_accuracy'] = total_td_succ / total_td_att if total_td_att > 0 else 0

            # Takedown defense (from opponent rounds)
            if len(opponent_rounds) > 0:
                opp_td_att = opponent_rounds['takedown_att'].sum()
                opp_td_succ = opponent_rounds['takedown_succ'].sum()
                stats['takedown_defense'] = (opp_td_att - opp_td_succ) / opp_td_att if opp_td_att > 0 else 0
            else:
                stats['takedown_defense'] = 0

            # Clinch and distance strike rates
            total_clinch = fighter_rounds['clinch_strikes_succ'].sum()
            total_distance = fighter_rounds['distance_strikes_succ'].sum()
            total_rounds = len(fighter_rounds)
            stats['clinch_strike_rate'] = total_clinch / total_rounds if total_rounds > 0 else 0
            stats['distance_strike_rate'] = total_distance / total_rounds if total_rounds > 0 else 0

            # Strikes landed per minute
            # Assume 5 minutes per round as we don't have exact time data
            total_minutes = len(fighter_rounds) * 5
            if total_minutes > 0:
                stats['strikes_landed_per_min'] = total_strikes / total_minutes
            else:
                stats['strikes_landed_per_min'] = 0

        else:
            stats['head_strike_pct'] = 0
            stats['striking_defense'] = 0
            stats['takedown_accuracy'] = 0
            stats['takedown_defense'] = 0
            stats['clinch_strike_rate'] = 0
            stats['distance_strike_rate'] = 0
            stats['strikes_landed_per_min'] = 0

        # Calculate finish rates from fight data
        total_fights = len(fighter_fights)
        if total_fights > 0:
            # Submission rate
            sub_wins = len(fighter_fights[
                (fighter_fights['result'] == 'Submission') &
                (fighter_fights['winner'] == fighter_id)
            ])
            stats['submission_rate'] = sub_wins / total_fights

            # KO rate
            ko_wins = len(fighter_fights[
                (fighter_fights['result'] == 'KO/TKO') &
                (fighter_fights['winner'] == fighter_id)
            ])
            stats['ko_rate'] = ko_wins / total_fights
        else:
            stats['submission_rate'] = 0
            stats['ko_rate'] = 0

        return stats

    def _create_opponent_quality_features(
        self,
        current_fight: pd.Series,
        fighter_1: pd.Series,
        fighter_2: pd.Series,
        all_fights: pd.DataFrame,
        fighters_df: pd.DataFrame
    ) -> Dict:
        """Create opponent quality and strength of schedule features."""
        features = {}

        for i, fighter_id in enumerate([current_fight['fighter_1'], current_fight['fighter_2']], 1):
            prefix = f'fighter_{i}_'

            # Get fighter's historical fights
            fighter_fights = all_fights[
                (all_fights['fighter_1'] == fighter_id) |
                (all_fights['fighter_2'] == fighter_id)
            ].copy()

            if len(fighter_fights) == 0:
                features[f'{prefix}avg_opponent_win_rate'] = np.nan
                features[f'{prefix}vs_winning_record'] = np.nan
                features[f'{prefix}vs_elite_opponents'] = np.nan
                # REMOVED: strength_of_schedule - duplicate
                features[f'{prefix}quality_wins'] = 0
                features[f'{prefix}bad_losses'] = 0
                continue

            opponent_win_rates = []
            quality_wins = 0
            bad_losses = 0
            vs_winning_record = 0

            for _, fight in fighter_fights.iterrows():
                # Determine opponent
                opponent_id = fight['fighter_2'] if fight['fighter_1'] == fighter_id else fight['fighter_1']

                # Get opponent's record at time of fight
                opponent_fights_before = all_fights[
                    ((all_fights['fighter_1'] == opponent_id) | (all_fights['fighter_2'] == opponent_id)) &
                    (pd.to_datetime(all_fights['event_date']) < pd.to_datetime(fight['event_date']))
                ]

                if len(opponent_fights_before) > 0:
                    # Calculate opponent's win rate at time of fight
                    opp_wins = sum(1 for _, f in opponent_fights_before.iterrows()
                                 if f.get('winner') == opponent_id)
                    opp_total = len(opponent_fights_before)
                    opp_win_rate = opp_wins / opp_total if opp_total > 0 else 0

                    opponent_win_rates.append(opp_win_rate)

                    # Quality metrics
                    if opp_win_rate > 0.5:
                        vs_winning_record += 1

                    # Elite opponent (>80% win rate with >5 fights)
                    if opp_win_rate > 0.8 and opp_total >= 5:
                        if fight.get('winner') == fighter_id:
                            quality_wins += 1

                    # Bad loss (losing to sub-.500 opponent)
                    if opp_win_rate < 0.5 and fight.get('winner') == opponent_id:
                        bad_losses += 1

            # Calculate features
            features[f'{prefix}avg_opponent_win_rate'] = np.mean(opponent_win_rates) if opponent_win_rates else np.nan
            features[f'{prefix}vs_winning_record'] = vs_winning_record / len(fighter_fights) if len(fighter_fights) > 0 else 0
            features[f'{prefix}vs_elite_opponents'] = len([wr for wr in opponent_win_rates if wr > 0.8]) / len(fighter_fights) if len(fighter_fights) > 0 else 0
            # REMOVED: strength_of_schedule - 100% duplicate of avg_opponent_win_rate
            features[f'{prefix}quality_wins'] = quality_wins
            features[f'{prefix}bad_losses'] = bad_losses

        return features

    def validate_features(self, training_df: pd.DataFrame, prediction_df: pd.DataFrame) -> bool:
        """
        Validate that training and prediction datasets have consistent features.

        Args:
            training_df: Training dataset
            prediction_df: Prediction dataset

        Returns:
            True if features are consistent
        """
        training_cols = set(training_df.columns) - {'winner'}  # Exclude target
        prediction_cols = set(prediction_df.columns) - {'winner'}

        if training_cols != prediction_cols:
            missing_in_prediction = training_cols - prediction_cols
            extra_in_prediction = prediction_cols - training_cols

            if missing_in_prediction:
                logger.error(f"Features missing in prediction: {missing_in_prediction}")
            if extra_in_prediction:
                logger.error(f"Extra features in prediction: {extra_in_prediction}")

            return False

        logger.info("Feature consistency validation passed")
        return True