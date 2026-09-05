import xarray as xr
import pandas as pd
import numpy as np

# load observed monthly precipitation for bias correction
obs_precip = pd.read_csv('monthly_precipitation.csv')
obs_precip['date'] = pd.to_datetime(obs_precip['month'], errors='coerce')
obs_precip.drop(columns=['month'], inplace=True)
obs_precip['month'] = obs_precip['date'].dt.month

# load PROMICE station and extract coordinates for CMIP6 grid cell extraction
df = pd.read_csv('PROMICE_a_daily.csv')
latitude = df['lat'].median()
longitude_raw = df['lon'].median()
longitude = 360 + longitude_raw

# load historical datasets and join them
ds_hist_a = xr.open_dataset('precip_2007_2014.nc')
ds_hist_b = xr.open_dataset('precip_2015_2023.nc')
ds_hist = xr.concat([ds_hist_a, ds_hist_b], dim = 'time')

# load projected datasets
# there is a limit on the size of dataset downloaded from CMIP6 so two datasets were downloadd
# individual files concatenated - continuous time series created
ds_a_126 = xr.open_dataset('126_precip_2024_2073.nc')
ds_b_126 = xr.open_dataset('126_precip_2074_2100.nc')
ds_126 = xr.concat([ds_a_126, ds_b_126], dim='time')

ds_a_245 = xr.open_dataset('245_precip_2024_2073.nc')
ds_b_245 = xr.open_dataset('245_precip_2074_2100.nc')
ds_245 = xr.concat([ds_a_245, ds_b_245], dim='time')

ds_a_585 = xr.open_dataset('585_precip_2024_2073.nc')
ds_b_585 = xr.open_dataset('585_precip_2074_2100.nc')
ds_585 = xr.concat([ds_a_585, ds_b_585], dim='time')

datasets = {
    'hist': ds_hist,
    'ssp126': ds_126,
    'ssp245': ds_245,
    'ssp585': ds_585
}

# extract precipitation from the CMIP6 grid nearest to the PROMICE station
for name, ds in datasets.items():
    pr = ds['pr'].sel(
        lat=latitude,
        lon=longitude,
        method='nearest')
# convert precipitation from kg m^-2 s^-1 (equivalent to mm s^-1) to mm day^-1
# aggregate to monthly totals
    pr_mm_day = pr * 86400
    monthly_precip_mm = pr_mm_day.resample(time='MS').sum()
    df = monthly_precip_mm.to_dataframe().reset_index()
    df.rename(columns={'pr': 'monthly_precip'}, inplace=True)
# export for dataset integrity
    df.to_csv(f'{name}_monthly_precip.csv', index=False)
    print(f'Saved {name}_monthly_precip.csv')

# name the monthly precipitation datasets for clarity
precip_hist = pd.read_csv('hist_monthly_precip.csv')
precip_126 = pd.read_csv('ssp126_monthly_precip.csv')
precip_245 = pd.read_csv('ssp245_monthly_precip.csv')
precip_585 = pd.read_csv('ssp585_monthly_precip.csv')

# extract calendar month from timestamps for climatology calculations
for df in [precip_hist, precip_126, precip_245, precip_585]:
    df['date'] = pd.to_datetime(df['time'], errors='coerce')
    df['month'] = df['date'].dt.month

# calculate observed monthly climatology
# monthly climatology represents mean precip for each calendar month across the observation period
obs_clim = obs_precip.groupby('month')['monthly_precip_mm'].mean().reset_index()
# calculate historical CMIP6 monthly precipitation climatology
hist_clim = precip_hist.groupby('month')['monthly_precip'].mean().reset_index()

bias_table = obs_clim.merge(hist_clim, on='month')
bias_table['factor'] = np.where(
    bias_table['monthly_precip'] > 0,
    bias_table['monthly_precip_mm']/bias_table['monthly_precip'], 1.0
)
print(bias_table)

# apply bias-correction factors to future CMIP6 projections
# all months receive correction factor corresponding to their month
ssp_data = {
    'ssp126': precip_126,
    'ssp245': precip_245,
    'ssp585': precip_585
}

# generate bias-corrected monthly precipitation projections
for name in ssp_data:
    ssp_data[name] = ssp_data[name].merge(
        bias_table[['month', 'factor']],
        on='month', how='left'
    )
    ssp_data[name]['monthly_precip_bc'] = (
        ssp_data[name]['monthly_precip']
        * ssp_data[name]['factor']
    )

# save the future monthly precipitation data
for name, df in ssp_data.items():
    df.to_csv(f'{name}_monthly_precip.csv', index=False)

df_126 = pd.read_csv('126_grid.csv')
df_245 = pd.read_csv('245_grid.csv')
df_585 = pd.read_csv('585_grid.csv')

grid_datasets = {
    '126': df_126,
    '245': df_245,
    '585': df_585
}

precip_datasets = {
    '126': ssp_data['ssp126'],
    '245': ssp_data['ssp245'],
    '585': ssp_data['ssp585']
}

#processing each scenario separately
for scenario, glacier_df in grid_datasets.items():

    monthly_df = precip_datasets[scenario].copy()
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

# extract scenario-specific dataset
    joined_df.to_csv(f'{scenario}_precip_grid.csv', index=False)