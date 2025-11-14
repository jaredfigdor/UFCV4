"""Check what the LabelEncoder mapping is for weight classes."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load upcoming fights
fights = pd.read_csv('ufc_data/upcoming_fight_data.csv')

# Create label encoder like the dataset builder does
le = LabelEncoder()
encoded = le.fit_transform(fights['weight_class'])

# Get mapping
mapping = dict(enumerate(le.classes_))

print('LabelEncoder mapping (value -> weight class):')
print('='*60)
for k, v in sorted(mapping.items()):
    print(f'{k}: {v}')

print('\n\nSample fights with encoding:')
print('='*60)
sample = pd.DataFrame({
    'weight_class_original': fights['weight_class'].head(10),
    'weight_class_encoded': encoded[:10]
})
print(sample.to_string(index=False))
