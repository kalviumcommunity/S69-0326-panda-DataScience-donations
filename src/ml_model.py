from ml_preprocessing import load_data, clean_data, create_rfm_features

df = load_data()
df = clean_data(df)
rfm = create_rfm_features(df)

print(rfm.head())