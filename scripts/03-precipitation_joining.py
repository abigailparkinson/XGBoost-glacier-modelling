import xarray as xr
import pandas as pd
import zipfile
import os
import glob

# reading all ERA5 precipitation files
# input: ERA5 NetCDF files from the study area and period
all_dfs = []
for zip_file in sorted(glob.glob('ERA5_monthly_tp_*.nc')):
# extract archived ERA5 files
    extract_dir = zip_file.replace('.nc', '')
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_dir)
# open precipitation variable
    ds = xr.open_dataset(os.path.join(extract_dir, 'data_stream-mnth.nc'),
                         engine = 'netcdf4')
# convert precipitation from m to mm
    tp = ds['tp'] * 1000
# calculate catchment-wide mean precipitation - to avoid errors in high-resolution precipitation
    tp = tp.mean(dim=['latitude', 'longitude'])
#convert to dataframe
    df = tp.to_dataframe(name='mean_daily_precip')
    df = df.reset_index()
    df = df.rename(columns={'valid_time': 'month'})
    all_dfs.append(df)

# combine all monthly precipitation records to one dataset
monthly_df = pd.concat(all_dfs, ignore_index=True)
#convert mean daily precipitation to monthly totals
monthly_df['monthly_precip_mm'] = monthly_df['mean_daily_precip'] * monthly_df['month'].dt.days_in_month

monthly_df['month'] = (monthly_df['month'].dt.to_period('M').dt.to_timestamp())
monthly_df = monthly_df[
    ['month', 'monthly_precip_mm']].sort_values('month')
print(monthly_df.head())
print(monthly_df.tail())
print(monthly_df.info())

# glacier dataset with topographic and temperature variables
# can be substituted for any glacier dataset
glacier_df = pd.read_csv('temp_grid.csv')

# extracting the date for each input from the layer it came from
# raster layer name stores month and year - used to extract the date
ym = glacier_df['layer'].str.extract(
    r'(?P<year>\d{4})[-_](?P<month>\d{2})')
glacier_df['date'] = pd.to_datetime(ym['year'] + '-' + ym['month']+ '-01')

print(glacier_df[['layer', 'date']].head())

monthly_df['date'] = pd.to_datetime(monthly_df['month'])

# merging monthly precipitation to each grid cell
joined_df = glacier_df.merge(monthly_df[['date', 'monthly_precip_mm']],
                             on='date', how='left')

print(joined_df['monthly_precip_mm'].isnull().sum())

# partition precipitation into snowfall and rainfall based on temperature
# using a gradual transition not a hard transition
def snow_fraction(T):
    if T <= -1:
        return 1.0
    elif T >= 1:
        return 0.0
    else:
        return (1-T)/2

# apply the snow fraction to calculate snowfall and rainfall totals for each grid cell
joined_df['snow_fraction'] = (joined_df['temp_mean'].apply(snow_fraction))
joined_df['rain_fraction'] = 1 - (joined_df['snow_fraction'])

joined_df['snowfall_mm'] = joined_df['monthly_precip_mm'] * joined_df['snow_fraction']
joined_df['rainfall_mm'] = joined_df['monthly_precip_mm'] * joined_df['rain_fraction']


joined_df.to_csv('precip_grid.csv', index=False)
