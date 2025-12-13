"""Remove orphaned UFC 323 fights from upcoming data."""
import pandas as pd

upcoming_fights = pd.read_csv('ufc_data/upcoming_fight_data.csv')

print(f'Before: {len(upcoming_fights)} upcoming fights')

ufc323_id = '00e11b5c8b7bfeeb'
cleaned = upcoming_fights[upcoming_fights['event_id'] != ufc323_id]

print(f'After: {len(cleaned)} upcoming fights')
print(f'Removed: {len(upcoming_fights) - len(cleaned)} UFC 323 fights')

cleaned.to_csv('ufc_data/upcoming_fight_data.csv', index=False)
print('\nCleaned upcoming_fight_data.csv')
print('\nUFC 323 event and fights are now completely removed.')
print('When you run app.py, it will:')
print('1. Re-scrape UFC 323 as a COMPLETED event')
print('2. Scrape all fights with full results (17 columns)')
print('3. Add to training data properly')
