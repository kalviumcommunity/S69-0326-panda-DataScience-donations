from ml_preprocessing import load_data, clean_data, create_rfm_features

df = load_data()
df = clean_data(df)
rfm = create_rfm_features(df)

rfm['will_donate_again'] = rfm['frequency'].apply(lambda x: 1 if x > 1 else 0)

X = rfm[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = rfm['will_donate_again']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

print(rfm.head())