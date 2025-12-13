"""Restore UFC 323 to upcoming events so it can be properly migrated."""
import pandas as pd

# UFC 323 details
ufc323_data = {
    'event_id': '00e11b5c8b7bfeeb',
    'event_name': 'UFC 323: Dvalishvili vs. Yan 2',
    'event_date': '2025-12-06',
    'event_city': 'Las Vegas',
    'event_state': 'Nevada',
    'event_country': 'USA'
}

# Load current upcoming events
upcoming_events = pd.read_csv('ufc_data/upcoming_event_data.csv')

print(f'Current upcoming events: {len(upcoming_events)}')

# Check if UFC 323 is already there
if ufc323_data['event_id'] in upcoming_events['event_id'].values:
    print('UFC 323 already in upcoming events')
else:
    # Add UFC 323
    upcoming_events = pd.concat([
        pd.DataFrame([ufc323_data]),
        upcoming_events
    ], ignore_index=True)

    # Save
    upcoming_events.to_csv('ufc_data/upcoming_event_data.csv', index=False)
    print(f'\nAdded UFC 323 to upcoming events')
    print(f'Total upcoming events: {len(upcoming_events)}')

print('\nNow when you run app.py:')
print('1. Step 3 will detect UFC 323 is past due (dated 2025-12-06)')
print('2. It will move the event to completed')
print('3. Step 3.5 will scrape UFC 323 fights with full results')
print('4. Training data will include UFC 323!')
