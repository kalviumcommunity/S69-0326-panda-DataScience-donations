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