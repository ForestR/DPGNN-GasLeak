import pandas as pd
import xarray as xr

# Create a sample DataFrame
data = {'time': ['2022-01-01', '2022-01-02', '2022-01-03'],
        'latitude': [45.0, 46.0, 47.0],
        'longitude': [-120.0, -121.0, -122.0],
        'temperature': [20.0, 21.0, 19.0]}
df = pd.DataFrame(data)

# Convert the DataFrame to a xarray Dataset and save as a .nc file
ds = xr.Dataset.from_dataframe(df)
nc_path = '../../data/raw/raw_data.nc'  # '/path/to/file.nc'
ds.to_netcdf(nc_path)
