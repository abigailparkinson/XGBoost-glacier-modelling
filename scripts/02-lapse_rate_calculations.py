import pandas as pd
import numpy as np
import rasterio

# these datasets can be substituted for as many data sets as necessary
df_l = pd.read_csv('TAS_l_clean_daily.csv')
df_a = pd.read_csv('TAS_a_clean_daily.csv')
df_u =pd.read_csv('TAS_u_clean_daily.csv')

#standardising the date column
for df in [df_l, df_u, df_a]:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

#changing column names so this code can be reapplied to any temperature dataset
#substitute temp_final with the necessary column name and then the code can be run
df_l = df_l.rename(columns={'temp_final': 'T_l', 'alt' : 'z_l'})
df_a = df_a.rename(columns={'temp_final': 'T_a', 'alt' : 'z_a'})
df_u = df_u.rename(columns={'temp_final': 'T_u', 'alt' : 'z_u'})

# reference altitude - ensures reference altitude is not an outlier
Z_l = df_l['z_l'].median()
Z_a = df_a['z_a'].median()
Z_u = df_u['z_u'].median()

# merging the datasets so calculations can be done
df = df_l.merge(
    df_u[['date', 'T_u']],
    on='date',
    how='outer'
)

df = df.merge(
    df_a[['date', 'T_a']],
    on='date',
    how='outer'
)

# getting rid of any invalid data
df = df.dropna(subset=['date', 'T_l', 'T_u', 'T_a'])

# function to ensure the code only needs to be written once
# estimating daily lapse rates through linear regression of station
def calc_lapse(row):

    temps = np.array([
        row['T_l'],
        row['T_u'],
        row['T_a']], dtype=float)

    elevs = np.array([
        Z_l,
        Z_u,
        Z_a], dtype=float)

    mask = (np.isfinite(temps)
            & np.isfinite(elevs))
    temps = temps[mask]
    elevs = elevs[mask]
    if len(temps) < 2:
        return np.nan
    if len(np.unique(elevs)) < 2:
        return np.nan
    slope, _ = np.polyfit(elevs, temps, 1)
    return slope

df['lapse_rate_raw'] = df.apply(
    calc_lapse,
    axis=1
)

# clip unrealistic lapse rates
df['lapse_rate'] = df['lapse_rate'].clip(
    lower=-0.012,
    upper=-0.002
)

# station L chosen as the reference, because it has the longest record of unbroken readings
df['T_ref'] = df['T_l']
df['z_ref'] = df['z_l']

# load DEM (clipped to the catchment for computational efficiency)
# can be substituted for any DEM for the desired catchment
with rasterio.open('clipped_dem.tif') as src:
    dem = src.read(1)
    profile = src.profile.copy()

# initialise monthly temperature and PDD storage containers
monthly_pdd = {}
monthly_temp_sum = {}
monthly_temp_count = {}

# generate daily corrected temperature fields using lapse-rate correction
# and calculate PDD values
for _, row in df.iterrows():
    if pd.isna(row['lapse_rate']):
        continue
    month = row['date'].strftime('%Y-%m')
# apply the lapse rate calculated for that day
    temperature_raster = (
        row['T_ref'] + row['lapse_rate'] * (dem - row['z_ref'])
    )
# clipping to 0 - as PDD is only positive temperature
    pdd_raster = np.maximum(
        temperature_raster, 0
    )
    if month not in monthly_pdd:
        monthly_pdd[month] = pdd_raster.copy()
    else:
        monthly_pdd[month] += pdd_raster
    if month not in monthly_temp_sum:
        monthly_temp_sum[month] = temperature_raster.copy()
        monthly_temp_count[month] = 1
    else:
        monthly_temp_sum[month] += temperature_raster
        monthly_temp_count[month] += 1

#define output raster format
profile.update(dtype='float32', count=1, compress='lzw')

# export monthly PDD rasters
for month, raster in monthly_pdd.items():
    outfile = f'PDD_{month}.tif'
    with rasterio.open(outfile, 'w', **profile
        ) as dst:
        dst.write(raster.astype(np.float32), 1)
    print(f'Saved {outfile}')

#calculating monthly mean temperature and export
for month in monthly_temp_sum:
    mean_temp_raster = (
        monthly_temp_sum[month] / monthly_temp_count[month])
    outfile = f'monthly_temp_{month}.tif'
    with rasterio.open(outfile, 'w', **profile) as dst:
        dst.write(mean_temp_raster.astype(
            np.float32), 1
        )
    print(f'Saved {outfile}')