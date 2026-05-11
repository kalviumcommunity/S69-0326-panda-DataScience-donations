import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("raw_data/processed_donations.csv")

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
with open("models/donor_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save scaler
with open("models/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Model and scaler saved successfully")