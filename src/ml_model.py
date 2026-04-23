from ml_preprocessing import load_data, clean_data, create_rfm_features

df = load_data()
df = clean_data(df)
rfm = create_rfm_features(df)

rfm['will_donate_again'] = rfm['frequency'].apply(lambda x: 1 if x > 1 else 0)

X = rfm[['recency', 'frequency', 'total_amount', 'avg_amount']]
y = rfm['will_donate_again']

print(rfm.head())