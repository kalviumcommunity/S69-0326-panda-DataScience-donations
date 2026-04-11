import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("viridis")

# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
raw_data_path = os.path.join(base_dir, 'raw_data', 'donations_raw.csv')
output_dir = os.path.join(base_dir, 'output', 'charts')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(raw_data_path)

# Data Cleaning
df.drop_duplicates(inplace=True)
df['campaign_type'] = df['campaign_type'].fillna('Unknown')
df['payment_method'] = df['payment_method'].fillna('Unknown')
df['donation_date'] = pd.to_datetime(df['donation_date'])

# Analysis 1: Repeat Donors
donor_counts = df['donor_id'].value_counts()
repeat_donors = donor_counts[donor_counts > 1].index
one_time_donors = donor_counts[donor_counts == 1].index

repeat_contribution = df[df['donor_id'].isin(repeat_donors)]['donation_amount'].sum()
one_time_contribution = df[df['donor_id'].isin(one_time_donors)]['donation_amount'].sum()

# Visualization 1: Repeat vs One-time Donors
plt.figure(figsize=(10, 6))
plt.bar(['Repeat Donors', 'One-time Donors'], [repeat_contribution, one_time_contribution], color=['#4CAF50', '#2196F3'])
plt.title('Total Contribution: Repeat vs One-time Donors')
plt.ylabel('Total Donation Amount ($)')
plt.savefig(os.path.join(output_dir, 'repeat_donors.png'))
plt.close()