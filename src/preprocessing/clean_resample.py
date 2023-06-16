import pandas as pd

# Load the raw data from a CSV file
csv_path = '../../data/raw/raw_data.csv'
data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

# Remove any rows with missing or invalid values
data = data.dropna()

# Resample the data to a lower frequency (e.g., 0.1Hz)
data = data.resample('10S').mean()

# Save the cleaned and resampled data to a new CSV file
csv_path = '../../data/preprocessed/cleaned_data.csv'
data.to_csv(csv_path)
