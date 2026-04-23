import pandas as pd

def load_data():
    df = pd.read_csv("raw_data/donations_raw.csv")
    return df

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    print(df.info())

def clean_data(df):
    df['donation_date'] = pd.to_datetime(df['donation_date'])
    df = df.dropna()
    return df