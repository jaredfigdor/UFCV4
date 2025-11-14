"""Check model training metrics"""
import pickle
from pathlib import Path

data_folder = Path("ufc_data")
model_file = data_folder / "ufc_model.pkl"

if model_file.exists():
    print("Loading model...")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model).__name__}")
else:
    print("No model file found")

# Check predictions
pred_file = data_folder / "fight_predictions.csv"
if pred_file.exists():
    import pandas as pd
    preds = pd.read_csv(pred_file)
    print(f"\nPredictions: {len(preds)} fights")
    print(f"Confidence range: {preds['confidence'].min():.3f} - {preds['confidence'].max():.3f}")
    print(f"Average confidence: {preds['confidence'].mean():.3f}")
    print(f"\nConfidence distribution:")
    print(f"  > 70%: {(preds['confidence'] > 0.70).sum()} fights")
    print(f"  60-70%: {((preds['confidence'] > 0.60) & (preds['confidence'] <= 0.70)).sum()} fights")
    print(f"  50-60%: {((preds['confidence'] > 0.50) & (preds['confidence'] <= 0.60)).sum()} fights")
    print(f"  < 50%: {(preds['confidence'] <= 0.50).sum()} fights")
