import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed_donations.csv")

print(df.head())

X = df[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = df['will_donate_again']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])