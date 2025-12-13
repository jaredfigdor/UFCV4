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
            fighter_1_data, fighter_2_data, all_fights
        ))

        # Opponent quality features
        features.update(self._create_opponent_quality_features(
            fight, fighter_1_data, fighter_2_data, all_fights, fighters_df
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
            'Flyweight': 1,
            'Bantamweight': 2,
            'Featherweight': 3,
            'Lightweight': 4,
            'Welterweight': 5,
            'Middleweight': 6,
            'Light Heavyweight': 7,
            'Heavyweight': 8,
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

                features[f'{prefix}avg_striking_accuracy'] = np.nan
                features[f'{prefix}avg_takedown_accuracy'] = np.nan
                features[f'{prefix}avg_control_time'] = np.nan
                features[f'{prefix}avg_strikes_per_round'] = np.nan
                features[f'{prefix}days_since_last_fight'] = np.nan
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

            features[f'{prefix}avg_striking_accuracy'] = fighter_round_stats.get('striking_accuracy', np.nan)
            features[f'{prefix}avg_takedown_accuracy'] = fighter_round_stats.get('takedown_accuracy', np.nan)
            features[f'{prefix}avg_control_time'] = fighter_round_stats.get('control_time', np.nan)
            features[f'{prefix}avg_strikes_per_round'] = fighter_round_stats.get('strikes_per_round', np.nan)

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
                'strikes_per_round': np.nan
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
                'strikes_per_round': np.nan
            }

        # Calculate averages
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

        return {
            'striking_accuracy': striking_accuracy,
            'takedown_accuracy': takedown_accuracy,
            'control_time': avg_control_time,
            'strikes_per_round': avg_strikes_per_round
        }

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
        all_fights: pd.DataFrame
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

        return features

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