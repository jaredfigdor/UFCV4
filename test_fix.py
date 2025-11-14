"""Test the win/loss feature fix."""
import pandas as pd
from ufcscraper.dataset_builder import DatasetBuilder

print("Building datasets to test win/loss fix...")
builder = DatasetBuilder('ufc_data')
_, pred_df = builder.build_datasets(force_rebuild=True)

print(f"\nPrediction dataset: {len(pred_df)} fights")

# Find the Susurkaev vs McConico fight
# First, let's see what fights we have
print("\nAll upcoming fights:")
print(pred_df[['fight_id', 'fighter_1', 'fighter_2']].head(10))

# Look for the specific fighters
susurkaev_fights = pred_df[
    (pred_df['fighter_1'].str.contains('Susurkaev', na=False, case=False)) |
    (pred_df['fighter_2'].str.contains('Susurkaev', na=False, case=False))
]

if len(susurkaev_fights) > 0:
    print("\n\nSusurkaev fight found!")
    fight = susurkaev_fights.iloc[0]
    print(f"\nFight ID: {fight['fight_id']}")
    print(f"Fighter 1: {fight['fighter_1']}")
    print(f"Fighter 2: {fight['fighter_2']}")
    print(f"\nFighter 1 record:")
    print(f"  Wins: {fight['fighter_1_wins']}")
    print(f"  Losses: {fight['fighter_1_losses']}")
    print(f"  Total fights: {fight['fighter_1_total_fights']}")
    print(f"\nFighter 2 record:")
    print(f"  Wins: {fight['fighter_2_wins']}")
    print(f"  Losses: {fight['fighter_2_losses']}")
    print(f"  Total fights: {fight['fighter_2_total_fights']}")
else:
    print("\n\nSusurkaev fight not found. Checking fighter_data.csv...")
    fighters_df = pd.read_csv('ufc_data/fighter_data.csv')
    susurkaev = fighters_df[fighters_df['fighter_l_name'].str.contains('Susurkaev', na=False, case=False)]
    print(susurkaev[['fighter_f_name', 'fighter_l_name', 'fighter_w', 'fighter_l']])
