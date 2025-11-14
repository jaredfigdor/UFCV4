"""Analyze training data quality for optimization opportunities."""
import pandas as pd
import numpy as np

print("="*80)
print("TRAINING DATA QUALITY ANALYSIS")
print("="*80)

train = pd.read_csv('ufc_data/training_dataset_cache.csv')
print(f'\nTraining shape: {train.shape[0]} rows, {train.shape[1]} columns')

# Missing values
print(f'\nMissing values per column (top 20):')
missing = train.isnull().sum().sort_values(ascending=False).head(20)
for col, count in missing.items():
    pct = count/len(train)*100
    print(f'  {col}: {count} ({pct:.1f}%)')

# Check class balance
if 'winner' in train.columns:
    print(f'\nClass balance:')
    winner_counts = train['winner'].value_counts()
    for val, count in winner_counts.items():
        pct = count/len(train)*100
        print(f'  Winner={val}: {count} ({pct:.1f}%)')

# Check for constant features
print(f'\nConstant or near-constant features (variance < 0.01):')
numeric_cols = train.select_dtypes(include=[np.number]).columns
variances = train[numeric_cols].var()
low_var = variances[variances < 0.01].sort_values()
for col, var in low_var.items():
    unique_vals = train[col].nunique()
    print(f'  {col}: variance={var:.6f}, unique_values={unique_vals}')

# Check for highly correlated features
print(f'\nHighly correlated feature pairs (|correlation| > 0.95):')
corr_matrix = train[numeric_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []
for column in upper_tri.columns:
    for idx in upper_tri.index:
        value = upper_tri.loc[idx, column]
        if pd.notna(value) and value > 0.95:
            high_corr_pairs.append((idx, column, value))

if high_corr_pairs:
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
        print(f'  {feat1} <-> {feat2}: {corr:.3f}')
else:
    print('  None found')

# Check data types
print(f'\nData types:')
dtype_counts = train.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f'  {dtype}: {count} columns')

print(f'\n{"="*80}')
