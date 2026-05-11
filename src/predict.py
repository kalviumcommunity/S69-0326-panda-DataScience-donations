import pickle
import numpy as np

# Load model
model = pickle.load(open("models/donor_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

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
