import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("raw_data/processed_donations.csv")

df['will_donate_again'] = df['frequency'].apply(lambda x: 1 if x > 1 else 0)

X = df[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = df['will_donate_again']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))