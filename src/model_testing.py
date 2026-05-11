# python id="pvt7u4"
import pickle
import numpy as np

model = pickle.load(open("models/donor_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

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
