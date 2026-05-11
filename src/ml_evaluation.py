import pandas as pd

df = pd.read_csv("data/processed_donations.csv")

X = df[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = df['will_donate_again']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)