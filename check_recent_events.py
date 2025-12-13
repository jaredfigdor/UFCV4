import pandas as pd

events = pd.read_csv('ufc_data/event_data.csv')
fights = pd.read_csv('ufc_data/fight_data.csv')

events['event_date'] = pd.to_datetime(events['event_date'])
recent_events = events[events['event_date'] >= '2025-09-01'].sort_values('event_date')

print('Events since Sept 2025 and their fight counts:')
print('-' * 70)

events_without_fights = []
for _, evt in recent_events.iterrows():
    fc = len(fights[fights['event_id'] == evt['event_id']])
    status = 'OK' if fc > 0 else 'MISSING FIGHTS!'
    print(f'{str(evt["event_date"].date()):12s} {evt["event_name"]:40s} {fc:3d} fights  {status}')
    if fc == 0:
        events_without_fights.append(evt['event_id'])

print(f'\nTotal events without fights: {len(events_without_fights)}')
