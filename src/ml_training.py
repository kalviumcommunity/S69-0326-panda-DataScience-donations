import pandas as pd

df = pd.read_csv("raw_data/processed_donations.csv")

print(df.head())

# ✅ TARGET VARIABLE
df['will_donate_again'] = df['frequency'].apply(lambda x: 1 if x > 1 else 0)

# Features & target
X = df[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = df['will_donate_again']

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled data:")
print(X_scaled[:5])

# Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_scaled, y)

print("Model trained")

# Predictions
predictions = model.predict(X_scaled)

print("Predictions:")
print(predictions[:10])