# python id="pvt7u4"
import pickle
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

with open(MODEL_DIR / "donor_model.pkl", "rb") as file:
    model = pickle.load(file)
with open(MODEL_DIR / "scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

test_cases = [
    [5, 8, 2000, 250],
    [400, 1, 50, 50]
]

for case in test_cases:
    scaled = scaler.transform([case])
    prediction = model.predict(scaled)

    print("Input:", case)

    if prediction[0] == 1:
        print("Prediction: Will Donate Again")
    else:
        print("Prediction: Will Not Donate Again")
