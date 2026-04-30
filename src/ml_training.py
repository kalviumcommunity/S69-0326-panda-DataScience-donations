import pandas as pd

df = pd.read_csv("data/processed_donations.csv")

print(df.head())

X = df[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = df['will_donate_again']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_scaled, y)

print("Model trained")