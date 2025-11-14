"""Fix weight classes in predictions summary."""
import pandas as pd
from pathlib import Path

# Load predictions
pred = pd.read_csv('ufc_data/fight_predictions.csv')

# Load prediction dataset which has original weight_class strings
pred_dataset = pd.read_csv('ufc_data/prediction_dataset_cache.csv')
weight_class_original = dict(zip(pred_dataset['fight_id'], pred_dataset['weight_class']))

# Load fighter data for names
fighters = pd.read_csv('ufc_data/fighter_data.csv')
fighters['full_name'] = (fighters['fighter_f_name'].fillna('') + ' ' + fighters['fighter_l_name'].fillna('')).str.strip()
fighter_names = dict(zip(fighters['fighter_id'], fighters['full_name']))

# Add fighter names
pred['fighter_1_name'] = pred['fighter_1'].map(fighter_names).fillna('Unknown Fighter')
pred['fighter_2_name'] = pred['fighter_2'].map(fighter_names).fillna('Unknown Fighter')

# Add event info
events = pd.read_csv('ufc_data/upcoming_event_data.csv')
event_data = dict(zip(events['event_id'], zip(events['event_name'], events['event_date'])))
pred['event_name'] = pred['event_id'].map(lambda x: event_data.get(x, ('Unknown Event', ''))[0])
pred['event_date'] = pred['event_id'].map(lambda x: event_data.get(x, ('', ''))[1])

# Get weight class from original dataset (has string values, not encoded)
pred['weight_class_name'] = pred['fight_id'].map(weight_class_original).fillna('Unknown')

# Add predicted winner name
pred['predicted_winner_name'] = pred.apply(
    lambda row: row['fighter_1_name'] if row['fighter_1_win_probability'] > row['fighter_2_win_probability']
    else row['fighter_2_name'], axis=1
)

# Create summary
summary_cols = [
    'fight_id', 'event_name', 'event_date', 'fighter_1_name', 'fighter_2_name',
    'predicted_winner_name', 'confidence', 'fighter_1_win_probability', 'fighter_2_win_probability',
    'weight_class_name'
]
summary = pred[summary_cols].sort_values('confidence', ascending=False)

# Save
summary.to_csv('ufc_data/fight_predictions_summary.csv', index=False)

print('Fixed weight classes!')
print('\nSample predictions:')
print(summary[['fighter_1_name', 'fighter_2_name', 'weight_class_name']].head(15))
