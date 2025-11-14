"""Test web app data loading"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask
from ufcscraper.web_app import load_predictions_data, load_fighter_data, load_fight_details

# Setup Flask app config
from ufcscraper import web_app
web_app.app.config['DATA_FOLDER'] = 'ufc_data'

# Test data loading
print("=" * 50)
print("Testing Web App Data Loading")
print("=" * 50)

# Load predictions
pred = load_predictions_data()
print(f"\nLoaded {len(pred)} predictions")
print(f"Columns: {list(pred.columns)[:10]}")

if len(pred) > 0:
    print(f"\nFirst fight ID: {pred.iloc[0]['fight_id']}")
    print(f"Fighter 1: {pred.iloc[0].get('fighter_1', 'NOT FOUND')}")
    print(f"Fighter 2: {pred.iloc[0].get('fighter_2', 'NOT FOUND')}")

    # Test fight details
    fight_id = pred.iloc[0]['fight_id']
    fight_data = load_fight_details(fight_id)

    if fight_data:
        print(f"\nFighter 1 Data:")
        print(f"  Full Name: {fight_data['fighter_1'].get('full_name', 'NOT FOUND')}")
        print(f"  Wins: {fight_data['fighter_1'].get('fighter_w', 'NOT FOUND')}")
        print(f"  Losses: {fight_data['fighter_1'].get('fighter_l', 'NOT FOUND')}")
        print(f"  Height: {fight_data['fighter_1'].get('fighter_height_cm', 'NOT FOUND')}")

        print(f"\nFighter 2 Data:")
        print(f"  Full Name: {fight_data['fighter_2'].get('full_name', 'NOT FOUND')}")
        print(f"  Wins: {fight_data['fighter_2'].get('fighter_w', 'NOT FOUND')}")
        print(f"  Losses: {fight_data['fighter_2'].get('fighter_l', 'NOT FOUND')}")
        print(f"  Height: {fight_data['fighter_2'].get('fighter_height_cm', 'NOT FOUND')}")
    else:
        print("\nERROR: Could not load fight details")
else:
    print("\nERROR: No predictions found")
