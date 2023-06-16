import xarray as xr

# Open the .nc file as a xarray Dataset
ds = xr.open_dataset('/path/to/file.nc')

# Convert the Dataset to a pandas DataFrame
df = ds.to_dataframe()
