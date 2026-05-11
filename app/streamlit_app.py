import streamlit as st
import pickle
import numpy as np

# Page settings
st.set_page_config(
    page_title="Donor Retention Predictor",
    page_icon="💙",
    layout="centered"
)

# Custom UI Styling
st.markdown("""
<style>

.main {
    background-color: #f4f7fb;
}

h1 {
    color: #0b1f3a;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
}

.stButton > button {
    background-color: #0b1f3a;
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #163d6b;
    color: white;
}

.stNumberInput label {
    font-weight: bold;
    color: #0b1f3a;
}

.sidebar .sidebar-content {
    background-color: #0b1f3a;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("models/donor_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Title
st.title("💙 Donor Retention Predictor")

st.write(
    "Predict whether a donor is likely to contribute again using Machine Learning."
)

# Input fields
recency = st.number_input("📅 Recency")
frequency = st.number_input("🔁 Frequency")
total_amount = st.number_input("💰 Total Donation Amount")
avg_amount = st.number_input("📊 Average Donation Amount")

# Prediction button
if st.button("Predict"):

    data = np.array([[recency, frequency, total_amount, avg_amount]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.divider()

    if prediction[0] == 1:
        st.success("✅ Donor Will Donate Again")
    else:
        st.error("❌ Donor Will Not Donate Again")

# Sidebar
st.sidebar.header("💙 About")

st.sidebar.write(
    """
This app predicts donor retention using Machine Learning.

### Features Used:
- Recency
- Frequency
- Total Donation Amount
- Average Donation Amount
"""
)