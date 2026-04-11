import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
num_records = 2000
num_donors = 300

donor_ids = [f"D{str(i).zfill(4)}" for i in range(1, num_donors + 1)]

# Generate random dates within 2 years
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 1, 1)


def random_date():
    return start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

# Donation amounts (skewed realistic distribution)
def generate_amount():
    return round(np.random.exponential(scale=200) + 10, 2)

# Campaign types
campaigns = ["Education", "Health", "Disaster Relief", "Environment", "Animal Welfare"]

# Payment methods
payment_methods = ["UPI", "Credit Card", "Debit Card", "Net Banking"]

data = []

for i in range(num_records):
    donor = random.choice(donor_ids)

    record = {
        "donation_id": f"T{10000 + i}",
        "donor_id": donor,
        "donation_amount": generate_amount(),
        "donation_date": random_date(),
        "campaign_type": random.choice(campaigns),
        "payment_method": random.choice(payment_methods)
    }

    data.append(record)

df = pd.DataFrame(data)

# Introduce some missing values (for cleaning practice)
for col in ["campaign_type", "payment_method"]:
    df.loc[df.sample(frac=0.02).index, col] = None

# Introduce duplicates
df = pd.concat([df, df.sample(frac=0.02)], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save dataset
df.to_csv("../raw_data/donations_raw.csv", index=False)

print("Dataset generated successfully!")