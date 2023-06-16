import pandas as pd

# Load the raw data from a CSV file
data = pd.read_csv('raw/raw_data.csv', index_col=0, parse_dates=True)

# Remove any rows with missing or invalid values
data = data.dropna()

# Resample the data to a lower frequency (e.g., 0.1Hz)
data = data.resample('10S').mean()

# Save the cleaned and resampled data to a new CSV file
data.to_csv('preprocessed/cleaned_data.csv')
