import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("models/donor_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("Donor Retention Predictor")

recency = st.number_input("Recency")
frequency = st.number_input("Frequency")
total_amount = st.number_input("Total Donation Amount")
avg_amount = st.number_input("Average Donation Amount")

if st.button("Predict"):

    data = np.array([[recency, frequency, total_amount, avg_amount]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.success("Donor Will Donate Again")
    else:
        st.error("Donor Will Not Donate Again")