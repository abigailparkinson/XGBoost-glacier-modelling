import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading the grid dataset with relevant climate data and existing albedo data
df_full = pd.read_csv('precip_grid.csv')

# date time conversion and extracting the relevant date information
df_full['date'] = pd.to_datetime(df_full['date'])
df_full['year'] = df_full['date'].dt.year
df_full['month'] = df_full['date'].dt.month

#months are cyclical - encoding the seasonality to help RF understand
df_full['month_sin'] = np.sin(2*np.pi*df_full['month']/12)
df_full['month_cos'] = np.cos(2*np.pi*df_full['month']/12)

df = df_full.copy()

# loading future datasets
SSP_126 = pd.read_csv('126_precip_grid.csv')
SSP_245 = pd.read_csv('245_precip_grid.csv')
SSP_585 = pd.read_csv('585_precip_grid.csv')

# creating an albedo column in the future datasets
# creating relevant date columns and seasonality representations of months
for df in [SSP_126, SSP_245, SSP_585]:
    df['albedo_mean'] = np.nan
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)

# separating the observed dataset into training data (with observed albedo) and data to predict (no albedo data)
train_clean = df[df['albedo_mean'].notna()]
missing_clean = df[df['albedo_mean'].isna()]

print(df.info())

# choosing features to train albedo on - only using month information and climate data
# other features don't vastly affect climate data
features = ['month_sin', 'month_cos', 'PDD_mean', 'snowfall_mm', 'rainfall_mm']

# training random forest on the clean data
rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)
rf.fit(train_clean[features], train_clean['albedo_mean'])
# applying the trained model to inputs with missing albedo
df.loc[df['albedo_mean'].isna(), 'albedo_mean'] = rf.predict(missing_clean[features])

# preparing the dataset to plot a heatmap
# heatmap of glacier-wide monthly mean albedo to assess trends of albedo behaviour
monthly_clean = df.groupby(['year', 'month'])['albedo_mean'].mean().reset_index()

pivot_clean = monthly_clean.pivot(
    index='year', columns='month', values='albedo_mean'
)

plt.figure(figsize=(12,6))
sns.heatmap(pivot_clean, cmap='viridis', annot=False, fmt='.2f')
plt.title('Historic Monthly Mean Albedo')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()

# exporting cleaned dataset for training the XGBoost on
df.to_csv('albedo_grid.csv')

# running the albedo model on all future climate projections
for scenario_name, df in zip(
    ['SSP1-2.6', 'SSP2-4.5', 'SSP5-8.5'],
    [SSP_126, SSP_245, SSP_585]
):
    df['albedo_mean'] = rf.predict(df[features])
    monthly_clean = df.groupby(['year', 'month'])['albedo_mean'].mean().reset_index()

    pivot_clean = monthly_clean.pivot(
        index='year', columns='month', values='albedo_mean'
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_clean, cmap='viridis', annot=False, fmt='.2f')
    plt.title(f'{scenario_name} Monthly Mean Albedo')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.show()

# exporting the grids with albedo projections for use in glacier modelling
SSP_126.to_csv('126_albedo_grid.csv')
SSP_245.to_csv('245_albedo_grid.csv')
SSP_585.to_csv('585_albedo_grid.csv')
