import pandas as pd

events = pd.read_csv('ufc_data/event_data.csv')
events['event_date'] = pd.to_datetime(events['event_date'])

recent = events[events['event_date'] >= '2025-11-01'].sort_values('event_date')

print('Recent events in event_data.csv:')
for _, e in recent.iterrows():
    print(f'  {e["event_date"].date()} - {e["event_name"]}')

print(f'\nTotal events: {len(events)}')
print(f'Most recent event: {events["event_date"].max().date()}')
