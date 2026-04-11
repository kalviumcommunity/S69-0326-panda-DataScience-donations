import pandas as pd
import numpy as np
import random
import os
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