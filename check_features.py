"""Check features used by model"""
import pandas as pd
import pickle
from pathlib import Path

data_folder = Path('ufc_data')

with open(data_folder / 'ufc_features.pkl', 'rb') as f:
    features = pickle.load(f)

training = pd.read_csv(data_folder / 'training_dataset_cache.csv')

print('Features used by model:', len(features))
print('\nChecking for non-numeric features in model...')

non_numeric = []
for f in features:
    if f in training.columns:
        if training[f].dtype == 'object':
            non_numeric.append(f)
            print(f'  {f}: {training[f].unique()[:5]}')

if len(non_numeric) > 0:
    print(f"\nCRITICAL ISSUE: {len(non_numeric)} non-numeric features in model!")
    print("XGBoost cannot handle string features directly!")
else:
    print("\nAll features are numeric - OK")

# Check weight_class specifically
if 'weight_class' in training.columns:
    print(f"\nweight_class dtype: {training['weight_class'].dtype}")
    print(f"weight_class values: {training['weight_class'].unique()[:10]}")
