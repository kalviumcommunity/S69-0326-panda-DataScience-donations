import pandas as pd

# ✅ Load dataset
def load_data():
    df = pd.read_csv("raw_data/donations_raw.csv")
    return df

# ✅ Clean dataset
def clean_data(df):
    df['donation_date'] = pd.to_datetime(df['donation_date'])
    df = df.dropna()
    return df

# ✅ Feature Engineering (RFM)
def create_rfm_features(df):
    today = df['donation_date'].max()

    rfm = df.groupby('donor_id').agg({
        'donation_date': lambda x: (today - x.max()).days,
        'donor_id': 'count',
        'donation_amount': ['sum', 'mean']
    })

    rfm.columns = ['recency', 'frequency', 'total_amount', 'avg_amount']
    rfm = rfm.reset_index()

    return rfm

# ✅ Save processed data (VERY IMPORTANT)


# ✅ Main execution
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)

    rfm = create_rfm_features(df)
    print(rfm.head())

    save_processed_data(rfm)