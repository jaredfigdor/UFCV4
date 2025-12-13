import pandas as pd

fights = pd.read_csv('ufc_data/fight_data.csv')
print(f'Before: {len(fights)} fights')

fights_cleaned = fights[fights['event_id'] != 'bd92cf5da5413d2a']
print(f'After: {len(fights_cleaned)} fights (removed {len(fights) - len(fights_cleaned)})')

fights_cleaned.to_csv('ufc_data/fight_data.csv', index=False)
print('\nUFC 323 fights removed')
print('Now run: python app.py --force-rebuild')
print('It will re-scrape UFC 323 fights with complete results')
