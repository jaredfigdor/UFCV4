# New Features Added to ML Pipeline

## Summary

Added **22 new features** (11 per fighter) to the training and prediction datasets, all derived from the `round_data.csv` file. These features provide detailed insights into fighting styles, defensive capabilities, and in-fight statistics.

**Total features increased from ~115 to ~137**

## New Features List

### 1. Takedown Defense (`takedown_defense`)
- **Description**: Percentage of opponent's takedown attempts that were successfully defended
- **Calculation**: `(opponent_takedown_att - opponent_takedown_succ) / opponent_takedown_att`
- **Example value**: 0.8333 (83.33% defense rate)
- **Use case**: Identifies fighters with strong takedown defense

### 2. Striking Defense (`striking_defense`)
- **Description**: Percentage of opponent's strikes that were successfully blocked/evaded
- **Calculation**: `(opponent_strikes_att - opponent_strikes_succ) / opponent_strikes_att`
- **Example value**: 0.6288 (62.88% defense rate)
- **Use case**: Measures defensive striking ability

### 3. Strikes Landed Per Minute (`strikes_landed_per_min`)
- **Description**: Average number of successful strikes landed per minute of fight time
- **Calculation**: `total_strikes_succ / (total_rounds * 5)`
- **Example value**: 2.9217 strikes/min
- **Use case**: Measures offensive output and pace

### 4. Strikes Absorbed Per Minute (`strikes_absorbed_per_min`)
- **Description**: Average number of strikes absorbed (taken) per minute of fight time
- **Calculation**: `opponent_total_strikes_succ / (total_rounds * 5)`
- **Example value**: 2.8696 strikes/min
- **Use case**: Measures durability and defensive effectiveness

### 5. Knockdowns Per Fight (`knockdowns_per_fight`)
- **Description**: Average number of knockdowns scored per fight
- **Calculation**: `total_knockdowns / num_fights`
- **Example value**: 0.3000 (0.3 knockdowns per fight on average)
- **Use case**: Identifies power strikers

### 6-8. Strike Target Distribution
Three features showing percentage of strikes targeting different areas:

#### Head Strike Percentage (`head_strike_pct`)
- **Calculation**: `total_head_strikes_succ / total_targeted_strikes`
- **Example value**: 0.6696 (66.96% of strikes target head)

#### Body Strike Percentage (`body_strike_pct`)
- **Calculation**: `total_body_strikes_succ / total_targeted_strikes`
- **Example value**: 0.1815 (18.15% of strikes target body)

#### Leg Strike Percentage (`leg_strike_pct`)
- **Calculation**: `total_leg_strikes_succ / total_targeted_strikes`
- **Example value**: 0.1488 (14.88% of strikes target legs)

**Use case**: Identifies fighting style patterns (head hunters vs leg kickers)

### 9. Average Takedowns Attempted (`avg_takedowns_attempted`)
- **Description**: Average number of takedown attempts per fight
- **Calculation**: `total_takedown_att / num_fights`
- **Example value**: 2.4000 takedowns attempted per fight
- **Use case**: Identifies wrestlers vs strikers

### 10. Submission Attempts Per Fight (`submission_attempts_per_fight`)
- **Description**: Average number of submission attempts per fight
- **Calculation**: `total_submission_att / num_fights`
- **Example value**: 0.0000 (no submission attempts)
- **Use case**: Identifies grapplers and submission specialists

### 11. Reversals Per Fight (`reversals_per_fight`)
- **Description**: Average number of position reversals per fight
- **Calculation**: `total_reversals / num_fights`
- **Example value**: 0.0000 (no reversals)
- **Use case**: Measures grappling control and scrambling ability

## Implementation Details

### Modified Files
1. **ufcscraper/feature_engineering.py**
   - Updated `_get_fighter_round_stats()` method (lines 566-721)
   - Updated `_create_historical_features()` method to use new stats (lines 545-567)
   - Added default values for fighters with no historical data (lines 485-511)

### Data Source
All features are calculated from the `round_data.csv` file which contains:
- `knockdowns`: Knockdowns scored in each round
- `strikes_att`, `strikes_succ`: Strike attempts and successful strikes
- `head_strikes_succ`, `body_strikes_succ`, `leg_strikes_succ`: Target-specific strikes
- `takedown_att`, `takedown_succ`: Takedown attempts and successes
- `submission_att`: Submission attempts
- `reversals`: Position reversals
- `ctrl_time`: Control time per round

### Defensive Stats Calculation
Defensive statistics are uniquely calculated by analyzing **opponent's** performance against the fighter:
- **Takedown Defense**: Based on how many of opponent's takedown attempts failed
- **Striking Defense**: Based on how many of opponent's strikes missed
- **Strikes Absorbed**: Direct count of opponent's successful strikes

This provides a true measure of defensive ability rather than just offensive stats.

## Feature Naming Convention

All features follow the pattern: `fighter_{1|2}_{feature_name}`

Examples:
- `fighter_1_takedown_defense`
- `fighter_2_strikes_landed_per_min`
- `fighter_1_head_strike_pct`

## Testing

Tested with 10 sample fights:
- ✓ All 22 features (11 per fighter) generated successfully
- ✓ Total features: 137 (increased from ~115)
- ✓ No features with all null values
- ✓ High data coverage (6-9 out of 10 fights had non-null values)

## Next Steps

1. **Cache cleared**: ML cache files deleted to force rebuild with new features
2. **Model retraining**: Run `python app.py --retrain-model` to train with new features
3. **Feature importance**: Check which new features are most predictive
4. **Performance evaluation**: Compare model accuracy with and without new features

## Expected Impact

These features should improve prediction accuracy by:
1. **Better style matchup analysis**: Strike distribution shows head hunters vs leg kickers
2. **Defensive capability measurement**: Striking/takedown defense weren't previously captured
3. **Pace and volume metrics**: Strikes per minute shows output rate
4. **Durability assessment**: Strikes absorbed indicates chin strength
5. **Grappling depth**: Submission attempts and reversals add grappling nuance

The model can now make more informed predictions based on style matchups (e.g., striker with good takedown defense vs wrestler).
