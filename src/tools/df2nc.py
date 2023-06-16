import pandas as pd
import xarray as xr

# Create a sample DataFrame
data = {'time': ['2022-01-01 00:00:01', '2022-01-01 00:00:02', '2022-01-01 00:00:03'],
        'temperature': [26.2, 26.0, 25.9],
        'pressure': [120.0, 121.0, 120.0],
        'environment_pressure': [105.325, 105.425, 105.330]}
df = pd.DataFrame(data)

# Convert the DataFrame to a xarray Dataset and save as a .nc file
ds = xr.Dataset.from_dataframe(df)
nc_path = '../../data/raw/raw_data.nc'  # '/path/to/file.nc'
ds.to_netcdf(nc_path)
