import xarray as xr

# Open the .nc file as a xarray Dataset
nc_path = '../../data/raw/raw_data.nc'  # '/path/to/file.nc'
ds = xr.open_dataset(nc_path)

# Convert the Dataset to a pandas DataFrame
df = ds.to_dataframe()
print(df.head())

# Save the data to a new CSV file
csv_path = '../../data/raw/raw_data.csv'
df.to_csv(csv_path, sep=',', index=False)
