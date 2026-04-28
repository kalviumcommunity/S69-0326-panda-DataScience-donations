import pandas as pd

df = pd.read_csv("data/processed_donations.csv")

X = df[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = df['will_donate_again']