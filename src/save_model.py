import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "raw_data" / "processed_donations.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Create target
df['will_donate_again'] = df['frequency'].apply(
    lambda x: 1 if x >= 5 else 0
)

print(df['will_donate_again'].value_counts())

# Features & target
X = df[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = df['will_donate_again']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model
with open(MODEL_DIR / "donor_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save scaler
with open(MODEL_DIR / "scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Model and scaler saved successfully")