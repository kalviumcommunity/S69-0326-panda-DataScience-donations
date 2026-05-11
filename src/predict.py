import pickle
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

# Load model
with open(MODEL_DIR / "donor_model.pkl", "rb") as file:
    model = pickle.load(file)
with open(MODEL_DIR / "scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Example donor data
sample_data = np.array([[10, 5, 1200, 240]])

# Scale
sample_scaled = scaler.transform(sample_data)

# Predict
prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("Will Donate Again")
else:
    print("Will Not Donate Again")
