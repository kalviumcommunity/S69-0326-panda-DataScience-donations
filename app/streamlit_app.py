import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "raw_data" / "processed_donations.csv"

st.set_page_config(
    page_title="Donor Retention Predictor",
    page_icon="💙",
    layout="centered",
)

st.markdown("""
<style>
    .main {
        background-color: #f4f7fb;
    }

    h1 {
        color: #0b1f3a;
        text-align: center;
        font-size: 42px;
        font-weight: 700;
    }

    .stButton > button {
        background-color: #0b1f3a;
        color: white;
        border-radius: 12px;
        height: 48px;
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
        font-weight: 700;
        color: #0b1f3a;
    }

    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }

    .sidebar .sidebar-content {
        background-color: #0b1f3a;
        color: white;
    }

    .sidebar .sidebar-content p,
    .sidebar .sidebar-content li,
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3 {
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_artifacts():
    model_path = MODEL_DIR / "donor_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        missing = [path.name for path in [model_path, scaler_path] if not path.exists()]
        raise FileNotFoundError(
            f"Missing saved files: {', '.join(missing)}. Run `src/save_model.py` to generate them."
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler

try:
    dataset = load_data()
    dataset['will_donate_again'] = (dataset['frequency'] >= 5).astype(int)
    model, scaler = load_artifacts()
except Exception as error:
    st.error(f"Unable to load app data: {error}")
    st.stop()

st.title("💙 Donor Retention Predictor")
st.write("Use this tool to estimate whether a donor is likely to give again based on past behavior.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Capture donor details")
    median_values = dataset[['recency', 'frequency', 'total_amount', 'avg_amount']].median()

    recency = st.number_input(
        "📅 Recency (days since last donation)",
        min_value=0,
        max_value=1000,
        value=int(median_values.recency),
        step=1,
    )
    frequency = st.number_input(
        "🔁 Frequency (number of donations)",
        min_value=0,
        max_value=100,
        value=int(median_values.frequency),
        step=1,
    )
    total_amount = st.number_input(
        "💰 Total Donation Amount",
        min_value=0.0,
        max_value=float(dataset['total_amount'].max() * 1.5),
        value=float(median_values.total_amount),
        step=0.01,
        format="%.2f",
    )
    avg_amount = st.number_input(
        "📊 Average Donation Amount",
        min_value=0.0,
        max_value=float(dataset['avg_amount'].max() * 1.5),
        value=float(median_values.avg_amount),
        step=0.01,
        format="%.2f",
    )

    if st.button("Predict donor retention"):
        donor_data = np.array([[recency, frequency, total_amount, avg_amount]])
        donor_data_scaled = scaler.transform(donor_data)
        prediction = model.predict(donor_data_scaled)[0]
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(donor_data_scaled)[0, 1]

        st.divider()
        if prediction == 1:
            st.success("✅ This donor is likely to donate again.")
        else:
            st.error("❌ This donor is unlikely to donate again.")

        if probability is not None:
            st.metric("Retention probability", f"{probability:.1%}")
            st.progress(int(round(probability * 100)))

with col2:
    st.subheader("Dataset snapshot")
    st.write(dataset.head(6))
    st.markdown("**Training dataset summary**")
    summary = dataset[['recency', 'frequency', 'total_amount', 'avg_amount']].describe().T
    summary = summary.rename(columns={"50%": "median"})[['mean', 'median', 'std']]
    st.table(summary)

st.sidebar.header("💙 About")
st.sidebar.write(
    """
    This app predicts donor retention using a logistic regression model.

    The model was trained on historical donation data and uses the following features:

    - Recency
    - Frequency
    - Total donation amount
    - Average donation amount
    """
)

with st.expander("Training dataset insights"):
    st.write("A donor is considered likely to donate again when frequency is 5 or higher in the dataset.")
    st.metric("Total donors", len(dataset))
    retention_rate = int((dataset['will_donate_again'].mean()) * 100)
    st.metric("Likely to donate again", f"{retention_rate}%")
    st.bar_chart(dataset[['frequency', 'total_amount', 'avg_amount']].median())
