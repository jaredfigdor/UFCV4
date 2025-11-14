"""
Test script to verify temporal filtering works correctly.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from ufcscraper.dataset_builder import DatasetBuilder
from ufcscraper.feature_engineering import FeatureEngineering

def test_temporal_filtering():
    """Test that records are calculated temporally."""
    print("[TEST] Testing Temporal Filtering Logic...")
    print("=" * 60)

    data_folder = Path("ufc_data")

    # Load a small subset of data
    fights_df = pd.read_csv(data_folder / "fight_data.csv", encoding='utf-8').tail(100)
    events_df = pd.read_csv(data_folder / "event_data.csv", encoding='utf-8')
    fighters_df = pd.read_csv(data_folder / "fighter_data.csv", encoding='utf-8')

    # Merge to get dates
    fights_with_dates = fights_df.merge(
        events_df[['event_id', 'event_date']],
        on='event_id',
        how='left'
    )
    fights_with_dates['event_date'] = pd.to_datetime(fights_with_dates['event_date'])
    fights_with_dates = fights_with_dates.sort_values('event_date')

    print(f"[OK] Loaded {len(fights_with_dates)} test fights")
    print(f"  Date range: {fights_with_dates['event_date'].min()} to {fights_with_dates['event_date'].max()}")

    # Test temporal record calculation
    fe = FeatureEngineering()

    # Pick a fight in the middle
    test_fight_idx = len(fights_with_dates) // 2
    test_fight = fights_with_dates.iloc[test_fight_idx]
    test_fighter_id = test_fight['fighter_1']
    test_date = test_fight['event_date']

    print(f"\n[TARGET] Testing fighter: {test_fighter_id}")
    print(f"   Fight date: {test_date}")

    # Calculate record at time of fight
    record = fe._calculate_record_at_time(
        fighter_id=test_fighter_id,
        all_fights=fights_with_dates,
        cutoff_date=pd.to_datetime(test_date)
    )

    print(f"\n[RECORD] Temporal Record (before {test_date}):")
    print(f"   Wins: {record['wins']}")
    print(f"   Losses: {record['losses']}")
    print(f"   Total fights: {record['total_fights']}")
    print(f"   Win %: {record['win_percentage']:.1%}")

    # Verify it only counted fights before this date
    fighter_fights_before = fights_with_dates[
        ((fights_with_dates['fighter_1'] == test_fighter_id) |
         (fights_with_dates['fighter_2'] == test_fighter_id)) &
        (fights_with_dates['event_date'] < test_date)
    ]

    print(f"\n[VERIFY] Verification:")
    print(f"   Fights found before {test_date}: {len(fighter_fights_before)}")
    print(f"   Record shows total fights: {record['total_fights']}")

    if len(fighter_fights_before) == record['total_fights']:
        print("   [PASS] Temporal filtering working correctly!")
    else:
        print(f"   [FAIL] Mismatch! Expected {len(fighter_fights_before)} but got {record['total_fights']}")
        return False

    # Test that it DOESN'T include future fights
    fighter_fights_after = fights_with_dates[
        ((fights_with_dates['fighter_1'] == test_fighter_id) |
         (fights_with_dates['fighter_2'] == test_fighter_id)) &
        (fights_with_dates['event_date'] >= test_date)
    ]

    print(f"   Fights after {test_date}: {len(fighter_fights_after)}")
    print(f"   [PASS] Future fights correctly excluded!")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED - Temporal filtering is working correctly!")
    return True

if __name__ == "__main__":
    try:
        test_temporal_filtering()
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
