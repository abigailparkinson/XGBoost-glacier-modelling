import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# for reproducibility - can put as many or as few weather stations and the respective ERA5 data as necessary
stations = {
    'l': {'promice': 'PROMICE_l_daily.csv',
          'era5': 'ERA5_l_daily.nc',
          'output': 'TAS_l_clean_daily.csv'},
    'u': {'promice': 'PROMICE_u_daily.csv',
          'era5': 'ERA5_u_daily.nc',
          'output': 'TAS_u_clean_daily.csv'},
    'a': {'promice': 'PROMICE_a_daily.csv',
          'era5': 'ERA5_a_daily.nc',
          'output': 'TAS_a_clean_daily.csv'}
}

#defining a cleaning function so that the cleaning function only needs to be defined once
def clean_station(station_name, files):
    print(f'\nProcessing station {station_name}')

    df_promice = pd.read_csv(files['promice'])
    df_promice['time'] = pd.to_datetime(df_promice['time'])
    df_promice = (df_promice.sort_values('time').set_index('time'))
#filtering - reduces the computational load
    df_filtered = df_promice[['t_u', 'alt']].copy()

#plots to ensure validity - raw data in isolation
    plt.figure(figsize=(10, 10))
    plt.plot(df_promice.index, df_promice['t_u'], linewidth=0.8, color='blue')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Station {station_name}: Raw Temperature')
    plt.show()
    plt.savefig(f'{station_name}_raw.png', dpi=300)

# reading and processing the NC file of ERA5 data
    ds = xr.open_dataset(files['era5'])
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
# era5 is in kelvin - convert to degrees C
    ds['t2m_c'] = ds['t2m'] - 273.15
    temp_series = ds['t2m_c'].mean(dim=['latitude', 'longitude'])

#converting to a dataframe so it can be joined onto the PROMICE dataset
    df_era5 = (temp_series.to_dataframe().reset_index())
    df_era5 = df_era5[['time', 't2m_c']]
# ensure integrity of temperature datasets - label the ERA5 data as being from ERA5 to ensure its origin is preserved
    df_era5 = df_era5.rename(columns={
        't2m_c' : 'T_era5'
    })
    df_era5['time'] = pd.to_datetime(df_era5['time']).dt.floor('D')
    df_era5 = df_era5.set_index('time')
    df_era5 = (df_era5.resample('D').mean())

# joining the datasets for interpolation calculation
    df = pd.concat([df_filtered, df_era5], axis=1)
# altitude of the PROMICE data is not always present - need to fill it with something
# median is a robust statistic as it is less skewed by outliers or incorrect altitude readings than mean
    df['alt'] = df['alt'].fillna(df['alt'].median())

# interpolating short gaps in PROMICE dataset without ERA5 data support - preserves local conditions
    df['t_u_interpolated'] = (
    df['t_u'].interpolate(method = 'time', limit=30
    ))

# PROMICE ERA-5 temperature bias
    df['bias'] = df['t_u_interpolated'] - df['T_era5']
# smooth bias with a year rolling mean
    df['bias_smooth'] = (
        df['bias'].rolling(window=365, min_periods=30).mean()
    )
    overall_bias = df['bias'].mean()
    df['bias_smooth'] = (df['bias_smooth'].fillna(overall_bias))

# bias correct ERA5 temperatures where gaps are long
    df['temp_filled'] = df['t_u_interpolated']
    missing = df['temp_filled'].isna()
    df.loc[missing, 'temp_filled'] = (df.loc[missing, 'T_era5'] + df.loc[missing, 'bias_smooth'])

# generate final temperature series
    df['temp_final'] = (
        df['temp_filled']
    )

# calculate daily positive temperature and postivie temperature exposure
    df['PDD_daily'] = (df['temp_final'].clip(lower=0))

# export processed dataset
    df_output = (
        df.reset_index()[['time',
                          'temp_final',
                          'T_era5',
                          'PDD_daily',
                          'alt']]
    )
    df_output.to_csv(files['output'], index=False)

    promice_start = df_promice.index.min()
    promice_end = df_promice.index.max()
    plot_df = df.loc[promice_start:promice_end].copy()

# plot to ensure validity of cleaning
    plt.figure(figsize=(10, 10))

    plt.plot(
        plot_df.index,
        plot_df['t_u'],
        color='black',
        alpha=0.8,
        linewidth = 0.8,
        label='PROMICE Raw'
    )

    plt.plot(
        plot_df.index,
        plot_df['T_era5'],
        color='orange',
        alpha=0.8,
        linewidth = 0.8,
        label='ERA5'
    )

    plt.plot(
        plot_df.index,
        plot_df['temp_final'],
        color='blue',
        linewidth=0.8,
        label='Cleaned / Filled'
    )

    plt.title(
        f'{station_name.upper()}: Raw vs ERA5 vs Cleaned'
    )

    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f'{station_name}_cleaning_check.png',
        dpi=300
    )

    plt.show()

# apply the cleaning function to each station
for station_name, files in stations.items():
    clean_station(station_name, files)
print('Finished')

# the function can be applied to as many or as few datasets as necessary