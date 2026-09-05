import pandas as pd
import xarray as xr

# finding the coordinates of the PROMICE stations for extracting CMIP6 projections
df = pd.read_csv('PROMICE_l.csv')
latitude = df['lat'].median()
longitude_raw = df['lon'].median()
longitude = 360 + longitude_raw
print(latitude)
print(longitude)

# loading CMIP6 temperature projection NCs
# for each of the selected SSP projections
# CMIP6 datasets were only able to be downloaded at 50 year intervals
# joining the two datasets to have a continuous temperature projection until 2100
l_ds_a = xr.open_dataset('126_temp_2024_2073.nc')
l_ds_b = xr.open_dataset('126_temp_2074_2100.nc')
l_ds = xr.concat([l_ds_a, l_ds_b], dim='time')

m_ds_a = xr.open_dataset('245_temp_2024_2073.nc')
m_ds_b = xr.open_dataset('245_temp_2074_2100.nc')
m_ds = xr.concat([m_ds_a, m_ds_b], dim='time')

u_ds_a = xr.open_dataset('585_temp_2024_2073.nc')
u_ds_b = xr.open_dataset('585_temp_2074_2100.nc')
u_ds = xr.concat([u_ds_a, u_ds_b], dim='time')

h_ds_a = xr.open_dataset('CMIP6_historical.nc')
h_ds_b = xr.open_dataset('CMIP6_historical_2.nc')
h_ds = xr.concat([h_ds_a, h_ds_b], dim='time')

datasets = {
    'historical': h_ds,
    'ssp126': l_ds,
    'ssp245': m_ds,
    'ssp585': u_ds
}

# extracting the grid cell that covers the PROMICE area
# converting temperature from kelvin to degrees centigrade
tas = {}
for name, ds in datasets.items():
    tas[name] = (
        ds['tas'].sel(
            lat = latitude,
            lon = longitude,
            method='nearest'
        ) - 273.15
    )
    print(name)
    print(tas[name])

tas_hist = tas['historical']
tas_126 = tas['ssp126']
tas_245 = tas['ssp245']
tas_585 = tas['ssp585']

# converting the datasets to dataframes
hist_df = tas_hist.to_dataframe().reset_index()
df_126 = tas_126.to_dataframe().reset_index()
df_245 = tas_245.to_dataframe().reset_index()
df_585 = tas_585.to_dataframe().reset_index()

# exporting the datasets to preserve dataset integrity
hist_df.to_csv('hist_temp.csv', index=False)
df_126.to_csv('ssp126.csv', index=False)
df_245.to_csv('ssp245.csv', index=False)
df_585.to_csv('ssp585.csv', index=False)

# loadin the observed and cleaned temperature datasets
df_a = pd.read_csv('TAS_a_clean_daily.csv')
df_u = pd.read_csv('TAS_u_clean_daily.csv')
df_l = pd.read_csv('TAS_l_clean_daily.csv')

# extracting dates from the CMIP6 projections dataframes
for df in [hist_df, df_126, df_245, df_585]:
    df['year'] = df['time'].str[:4].astype(int)
    df['month'] = df['time'].str[5:7].astype(int)
    df['day'] = df['time'].str[8:10].astype(int)
    df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-' + df['day'].astype(str).str.zfill(2)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# calculating monthly mean CMIP6 temperatures for historical and future scenarios
monthly_hist = (hist_df.groupby(
    ['year', 'month'])['tas'].mean().reset_index())
monthly_126 = (df_126.groupby(
    ['year', 'month'])['tas'].mean().reset_index())
monthly_245 = (df_245.groupby(
    ['year', 'month'])['tas'].mean().reset_index())
monthly_585 = (df_585.groupby(
    ['year', 'month'])['tas'].mean().reset_index())

# create monthly timestamps for all temperature datasets
monthly_hist['date'] = (
    monthly_hist['year'].astype(str)
    + '-'
    + monthly_hist['month'].astype(str).str.zfill(2)
)

monthly_126['date'] = (
    monthly_126['year'].astype(str)
    + '-'
    + monthly_126['month'].astype(str).str.zfill(2)
)

monthly_245['date'] = (
    monthly_245['year'].astype(str)
    + '-'
    + monthly_245['month'].astype(str).str.zfill(2)
)

monthly_585['date'] = (
    monthly_585['year'].astype(str)
    + '-'
    + monthly_585['month'].astype(str).str.zfill(2)
)

# convert projection timestamps to date time format
for df in [monthly_126, monthly_245, monthly_585]:
    df['date'] = pd.to_datetime(df['date'])

# extract month and day information from PROMICE observations
# for climatology and bias-correction calculations
for df in [df_a, df_l, df_u]:
    df['date'] = pd.to_datetime(df['time'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

# store PROMICE datasets for automated processing - can store as many datasets as necessary and the process is automated
aws_datasets = {
    'a' : df_a,
    'l' : df_l,
    'u' : df_u,
}

# calculate monthly temperature climatologies for each PROMICE station
# as the stations are stored, this automatically runs for every station
aws_clims = {}
for station, df in aws_datasets.items():
    aws_clims[station] = (
        df.groupby('month')['temp_final'].mean().reset_index())

# calculate historical CMIP6 monthly climatology
cmip_clim = hist_df.groupby('month')['tas'].mean().reset_index()

# calculate monthly bias between PROMICE observations and historical CMIP6 temperatures
# bias = PROMICE climatology - CMIP6 climatology
# positive value = CMIP6 underestimates temperature
# negative value = CMIP6 overestimates temperature
biases = {}
for station, clim in aws_clims.items():
    biases[station] = clim.merge(
        cmip_clim, on='month'
    )
    biases[station]['bias'] = (
        biases[station]['temp_final'] - biases[station]['tas']
    )

# apply monthly bias corrections to future CMIP6 projections
ssp_datasets = {
    '126' : df_126,
    '245' : df_245,
    '585' : df_585
}

# generate bias-corrected future temperature projections for each station and SSP scenario
future_datasets = {}
for station, bias in biases.items():
    for ssp, ssp_df in ssp_datasets.items():
        future = ssp_df.merge(
            bias[['month', 'bias']],
            on='month'
        )
        future['future_temp'] = (future['tas'] + future['bias'])
        future_datasets[f'{station}_{ssp}'] = future

# export bias-corrected future temperature projections
for name, df in future_datasets.items():
    df.to_csv(f'future_{name}.csv', index=False)

# this comes out with daily temperature projections for each station on the glacier
# these datasets can then go through the lapse rate calculations from the file 02-lapse_rate_calculations.py