import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import shap
import math
import os
import geopandas as gpd

# loading all datasets necessary
df = pd.read_csv('albedo_grid.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date', 'id_grid'])
df['northernness'] = np.cos(np.radians(df['aspect_mean']))
df_126 = pd.read_csv('126_albedo_grid.csv')
df_245 = pd.read_csv('245_albedo_grid.csv')
df_585 = pd.read_csv('585_albedo_grid.csv')

# batch parsing of date and derivation of the northernness variable
for d in [df_126, df_245, df_585]:
    d['date'] = pd.to_datetime(d['date'])
    d['northernness'] = np.cos(np.radians(d['aspect_mean']))
    d = d.sort_values(['date', 'id_grid'])

#khan is rate of change per year - convert to rate of change per month
df['khan_monthly_mb'] = (df['mass_change_mean']/1000)/12
khan_glacier_monthly = (df.groupby('date')['khan_monthly_mb'].mean().sort_index())
khan_glacier_cumulative = (khan_glacier_monthly.cumsum())

# DDF values assigned based on Braithwaite and Olesen (1989), Hock (1999; 2003) and Huss et al. (2008)
# DDF can be changed as desired
DDF_ice = 8.3
DDF_snow = 4

# defining a function to run the temperature index model - streamlines the calibration process as the code does not need to be rewritten
def run_model(model_df, snowfall_efficiency):
    model_df = model_df.copy()
    # albedo normalisation factor
    a = ((model_df['albedo_mean'] - 0.35)/(0.85-0.35)).clip(0,1)
    # assigning DDF to each grid cell based on albedo
    model_df['DDF'] = (DDF_ice + (DDF_snow - DDF_ice)*a)

    # calculating ablation and accumulation
    model_df['ablation'] = model_df['DDF'] * model_df['PDD_mean']
    # snowfall accumulation factor - used for calibrating snowfall
    model_df['accumulation'] = model_df['snowfall_mm'] * snowfall_efficiency

    # calculating mass balance initially in mm - as a direct difference between accumulation and ablation
    model_df['mass_balance_mm'] = model_df['accumulation'] - model_df['ablation']
    # converted to metres for ease of interpretability
    model_df['mass_balance_m'] = model_df['mass_balance_mm']/1000
    model_df = model_df.sort_values(['id_grid', 'date'])

    # calculating glacier-wide mean monthly mass balance
    glacier_monthly = (model_df.groupby('date')['mass_balance_m'].mean().sort_index())
    # calculating glacier-wide thickness change
    glacier_cumulative = glacier_monthly.cumsum()

    return model_df, glacier_monthly, glacier_cumulative

# defining the function to calibrate snowfall efficiency
# calibration is finding the accumulation feature minimising the error between modelled and observed mass balance and thickness change
def objective(params):
    # extract snowfall efficiency parameter
    snowfall_efficiency = params[0]
    _, model_monthly, model_cumulative = run_model(
        df, snowfall_efficiency
    )
    # creating a comparison dataframe to compare values
    comparison_monthly = pd.DataFrame({
        'obs': khan_glacier_monthly,
        'model': model_monthly}).dropna()
    comparison_cumulative = pd.DataFrame({
        'obs': khan_glacier_cumulative,
        'model': model_cumulative}).dropna()
    # calculating RMSE for monthly mass balance and thickness change
    rmse_monthly = np.sqrt(mean_squared_error(comparison_monthly['obs'], comparison_monthly['model']))
    rmse_cumulative = np.sqrt(mean_squared_error(comparison_cumulative['obs'], comparison_cumulative['model']))
    # objective function - finds accumulation value with minimised normalised monthly and cumulative RMSE
    objective_value = (
        rmse_monthly/comparison_monthly['obs'].std()
    + rmse_cumulative/comparison_cumulative['obs'].std())
    return objective_value

# define plausible snowfall efficiency search range
bounds = [
    (0.1, 1.0)
]

# optimise snowfall efficiency with differential evolution
# maximum iterations of 500 for efficiency but sufficient iterations
# random seed of 42 for reproducibility
result = differential_evolution(
    objective, bounds=bounds, strategy='best1bin', popsize=20, maxiter=500, seed=42, polish=True, disp=True
)

best_snow_efficiency = result.x[0]

print('Best Snow Efficiency:', best_snow_efficiency)

# running the temperature index with calibrated accumulation factor
best_df, best_monthly, best_cumulative = run_model(
    df, best_snow_efficiency
)

# plotting cumulatie thickness change against modelled thickness change for model performance assessment
# shows how the model performs for long term height change predictions
plt.plot(khan_glacier_cumulative.index, khan_glacier_cumulative.values, label='Observed', linewidth=1)
plt.plot(best_cumulative.index, best_cumulative.values, label='Best Model', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Cumulative Thickness Change (m)')
plt.title('Observed vs Modelled Cumulative Thickness Change')
plt.legend()
plt.tight_layout()
plt.close()
print('Plot done!')

# same concept as above but for plotting monthly mass balance
# shows how the model projects the scale of mass balance positivity and negativity
plt.plot(khan_glacier_monthly.index, khan_glacier_monthly.values, label='Observed', linewidth=1)
plt.plot(best_monthly.index, best_monthly.values, label='Best Model', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Monthly MB (m w.e./yr)')
plt.title('Observed vs Modelled Monthly MB')
plt.legend()
plt.tight_layout()
plt.close()
print('Plot done!')

#xgboost section

# starting with derivation of variables not already existing in the datasets
for col in ['PDD_mean', 'snowfall_mm', 'albedo_mean', 'khan_monthly_mb']:
    df[f'{col}_lag1'] = df.groupby('id_grid')[col].shift(1)
    df[f'{col}_lag3'] = df.groupby('id_grid')[col].shift(3)

#interaction features
df['rain_melt'] = (df['rainfall_mm'] * df['PDD_mean'])
df['snow_protection'] = (df['snowfall_mm'] * (1- df['albedo_mean']))

#rolling accumulation and ablation
df['pdd_3mo'] = (df.groupby('id_grid')['PDD_mean'].transform(lambda x: x.rolling(3).mean()))
df['snow_3mo'] = (df.groupby('id_grid')['snowfall_mm'].transform(lambda x: x.rolling(3).mean()))

df = df.replace([np.inf, -np.inf], np.nan)

# reading high resolution dataset and dropping unnecessary features to reduce computational load
hr_df = pd.read_csv('high_res_albedo_grid.csv')
hr_df = hr_df.drop(columns=['col_index', 'row_index', 'elev_mean', 'temp_mean', 'fid', 'layer'])
hr_df['date'] = pd.to_datetime(hr_df['date'])
hr_df['northernness'] = np.cos(np.radians(hr_df['aspect_mean']))
hr_df = hr_df.sort_values(['date', 'id_grid'])

# calculating necessary lagged variables for the high resolution and future projections datasets
for d in [hr_df, df_126, df_245, df_585]:
    for col in ['PDD_mean', 'snowfall_mm', 'albedo_mean']:
        d[f'{col}_lag1'] = (d.groupby('id_grid')[col].shift(1))
        d[f'{col}_lag3'] = (d.groupby('id_grid')[col].shift(3))

    d['rain_melt'] = (d['rainfall_mm'] * d['PDD_mean'])
    d['snow_protection'] = (d['snowfall_mm'] * (1 - d['albedo_mean']))

    d['pdd_3mo'] = d.groupby('id_grid')['PDD_mean'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    d['snow_3mo'] = d.groupby('id_grid')['snowfall_mm'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    d.replace([np.inf, -np.inf], np.nan, inplace=True)

last_mb_lookup = df.sort_values('date').groupby('id_grid')['khan_monthly_mb'].last().to_dict()

#defining all feature sets - any desired feature set can be defined
# substitute any feature sets required
climate_features = ['PDD_mean',
                    'snowfall_mm',
                    'rainfall_mm'
                    ]

climate_seas = ['PDD_mean',
            'snowfall_mm',
            'rainfall_mm',
            'month_sin',
            'month_cos']

some_features = ['PDD_mean',
            'snowfall_mm',
            'rainfall_mm',
            'albedo_mean',
            'northernness',
            'slope_mean',
            'month_sin',
            'month_cos']

no_seas = ['PDD_mean',
            'snowfall_mm',
            'rainfall_mm',
            'albedo_mean',
            'northernness',
            'slope_mean']

no_seas_lag = ['PDD_mean',
            'snowfall_mm',
            'rainfall_mm',
            'albedo_mean',
            'northernness',
            'slope_mean',

            'PDD_mean_lag1',
            'PDD_mean_lag3',
            'khan_monthly_mb_lag1',

            'snow_protection',
            'rain_melt',

            'pdd_3mo',
            'snow_3mo'
            ]

full_features = ['PDD_mean',
            'snowfall_mm',
            'rainfall_mm',
            'albedo_mean',
            'northernness',
            'slope_mean',
            'month_sin',
            'month_cos',

            'PDD_mean_lag1',
            'PDD_mean_lag3',
            'khan_monthly_mb_lag1',

            'snow_protection',
            'rain_melt',

            'pdd_3mo',
            'snow_3mo'
            ]

# defining and naming the feature sets
# container can be changed where necessary
feature_sets = {
    'Just climate': climate_features,
    'Climate and seasonality': climate_seas,
    'Climate, topography and albedo': no_seas,
    'All but lagged': some_features,
    'All but seasonality': no_seas_lag,
    'All features': full_features
}

# creating a results container
results = {}

# creating the mass balance plots - foundation for all model outputs to be plotted against
# start with plotting just observed - foundation for assessing model performance
fig_train_monthly, ax_train_monthly = plt.subplots(figsize=(10, 6))
ax_train_monthly.plot(
    khan_glacier_monthly.index, khan_glacier_monthly.values,
    color='black', linewidth=1, label='Observed'
)

fig_train_cum, ax_train_cum = plt.subplots(figsize=(10, 6))
ax_train_cum.plot(khan_glacier_cumulative.index, khan_glacier_cumulative.values,
         label='Observed', linewidth=1, color='black')

# creating the trimmed observed dataset to be plotted against, and compared to the high resolution model
hr_start = hr_df['date'].min()
hr_end = hr_df['date'].max()
hr_obs = (khan_glacier_monthly.loc[hr_start:hr_end].copy())
hr_obs_cum = hr_obs.cumsum()
hr_obs_cum = hr_obs_cum - hr_obs_cum.iloc[0]

# creating the high resolution mass balance and thickness change plots - foundation for all model outputs to be plotted against
fig_hr_monthly, ax_hr_monthly = plt.subplots(figsize=(10, 6))
ax_hr_monthly.plot(hr_obs.index, hr_obs.values,
                   label='Observed', linewidth=1, color='black')

obs_cum = hr_obs.cumsum()
obs_cum = obs_cum - obs_cum.iloc[0]

fig_hr_cum, ax_hr_cum = plt.subplots(figsize=(10, 6))
ax_hr_cum.plot(obs_cum.index, obs_cum.values,
               label='Observed', linewidth=1, color='black')

# creating a loop so that the cross-validation, and model training can be performed on as many feature sets as desired
# training and testing of models on all defined feature sets is automated - code only needs to be run once with defined feature sets all being trained against
for name, features in feature_sets.items():
    print(f'\nRunning {name}')
    df_model = df.dropna(subset=features+ ['khan_monthly_mb']).copy()

# features defined before the for loop
    X = df_model[features]
# unconverted monthly mass balance used as the target - direct predictions without assumptions in calculations
    y = df_model['khan_monthly_mb']
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_predictions = np.full(
        len(df_model),
        np.nan
    )

# cross validation section
# designed for testing each model for generalisability and increases confidence that the model is not overfit
    for train_idx, test_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        # 1000 estimators, relatively shallow trees and low learning rate reduces overfitting but ensures training isn't overly expensive
        cv_model = XGBRegressor(
            n_estimators=1000,
            max_depth=3,
            learning_rate=0.02,
            # ensures randomness of splitting data
            subsample=0.8,
            colsample_bytree=0.8,
            # ensures only meaningful splits are learned
            min_child_weight=5,
            # defining hyper parameters
            reg_alpha=0.1,
            reg_lambda=1,
            gamma=0.1,
            # model objective is to minimise squared error
            objective='reg:squarederror',
            random_state=42
        )
        cv_model.fit(X_train, y_train)

        cv_predictions[test_idx] = (
            cv_model.predict(X_test)
        )

# deriving cross validation to be used in prediction error
    cv_rmse = np.sqrt(mean_squared_error(
        y, cv_predictions
    ))
# deriving cross validation R2 scores to assess model generalisability
# and performance on unseen data
    cv_r2 = r2_score(
        y, cv_predictions
    )
    print(f'CV RMSE: {cv_rmse}, '
          f'CV R2: {cv_r2}')

# training the predictive model
    final_model = XGBRegressor(
        # 1000 estimators, relatively shallow trees and low learning rate reduces overfitting but ensures training isn't overly expensive
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.02,
        # ensures randomness of splitting data
        subsample=0.8,
        colsample_bytree=0.8,
        # ensures only meaningful splits are learned
        min_child_weight=5,
        # defining hyper parameters
        reg_alpha=0.1,
        reg_lambda=1,
        gamma=0.1,
        # model objective is to minimise squared error
        objective='reg:squarederror',
        random_state=42
    )

# fitting the final model
    final_model.fit(X, y)
    df_model['pred_mb'] = final_model.predict(X)

# creating glacier-wide monthly mass balance and cumulative thickness change
    train_monthly_mb = (df_model.groupby('date')['pred_mb'].mean().sort_index())
    train_cumulative_mb = train_monthly_mb.cumsum()

    train_compare = pd.DataFrame({
        'khan': khan_glacier_monthly,
        'pred': train_monthly_mb
    }).dropna()

# plotting model outputs for each feature set on the same axis
# allows for direct inter-model comparison and automates the plotting processes
    ax_train_monthly.plot(
        train_monthly_mb.index,
        train_monthly_mb.values,
        linewidth=1,
        label=name
    )

    ax_train_cum.plot(
        train_cumulative_mb.index,
        train_cumulative_mb.values,
        linewidth=1,
        label=name
    )

    #uncertainty from CV RMSE
    error_95 = 1.96 * cv_rmse
    monthly_upper = (
        train_monthly_mb + error_95
    )
    monthly_lower = (
        train_monthly_mb - error_95
    )
    cum_error = (
        error_95 * np.sqrt(
        np.arange(1, len(train_cumulative_mb)+1)
    ))
    # calculating the upper and lower bounds of predictions from cumulative error
    cum_upper = (
        train_cumulative_mb + cum_error
    )
    cum_lower = (
        train_cumulative_mb - cum_error
    )

    final_thickness_change = (
        train_cumulative_mb.iloc[-1]
    )
    final_thickness_error = (
        cum_error[-1]
    )

    rss = np.sum(
        (df_model['khan_monthly_mb'] - df_model['pred_mb'])**2
    )

    # creating predicted and lagged mass balance columns - will be filled in the following
    hr_df['predicted_mb'] = np.nan
    hr_df['khan_monthly_mb_lag1'] = np.nan
# creating a copy of the high resolution dataframe to preserve the original dataset
    hr_model = hr_df.copy()
    hr_model['predicted_mb'] = np.nan
    unique_dates = sorted(hr_model['date'].unique())

# creating lagged monthly mass balance feature for the high resolution model outputs
# based on the prior month's prediction of mass balance
    for n, date in enumerate(unique_dates):
        mask = hr_model['date'] == date
        current = hr_model.loc[mask].copy()

        if n == 0:
            current['khan_monthly_mb_lag1'] = (
                current['id_grid'].map(last_mb_lookup).fillna(
                    df['khan_monthly_mb'].mean()
                ))
        else:
            prev_date = unique_dates[n - 1]
            prev_predictions = (
                hr_model.loc[
                    hr_model['date'] == prev_date, ['id_grid', 'predicted_mb']
                ].set_index('id_grid')['predicted_mb']
            )
            current['khan_monthly_mb_lag1'] = (
                current['id_grid'].map(prev_predictions)
            )

        current['khan_monthly_mb_lag1'] = (
            current['khan_monthly_mb_lag1'].fillna(
                df['khan_monthly_mb'].mean()))

        preds = final_model.predict(current[features].fillna(0))
        hr_model.loc[mask, 'predicted_mb'] = preds

    hr_monthly_mb = (hr_model.groupby('date')['predicted_mb'].mean().sort_index())
    hr_cumulative_mb = hr_monthly_mb.cumsum()

    col_name = ('mb_' + name.replace(" ", "_").replace(",", ""))
    hr_df[col_name] = hr_model['predicted_mb'].values

# creating a dataframe that covers the time frame of the high resolution dataset
    compare = pd.DataFrame({
        'khan': hr_obs,
        'pred': hr_monthly_mb
    }).dropna()

    khan_cum = compare['khan'].cumsum()
    khan_cum = khan_cum - khan_cum.iloc[0]

    pred_cum = compare['pred'].cumsum()
    pred_cum = pred_cum - pred_cum.iloc[0]

# finishing the high resolution plot with all lines on one plot
# plotting predictions for each model output
    ax_hr_monthly.plot(
        compare.index,
        compare['pred'],
        linewidth=1,
        label=name
    )

    ax_hr_cum.plot(
        compare.index,
        pred_cum,
        linewidth=1,
        label=name
    )

# calculating RMSE and R2 score for the outputs of each high resolution model
    hr_rmse = np.sqrt(
        mean_squared_error(
            compare['khan'],
            compare['pred']
        )
    )

    hr_r2 = r2_score(
        compare['khan'],
        compare['pred']
    )

#storing the results for each model
    results[name] = {
        'model': final_model,
        'cv_rmse': cv_rmse,
        'cv_r2': cv_r2,
        'monthly_upper': monthly_upper,
        'monthly_lower': monthly_lower,
        'cum_upper': cum_upper,
        'cum_lower': cum_lower,
        'final_thickness_change': final_thickness_change,
        'final_thickness_error': final_thickness_error,

        'hr_rmse': hr_rmse,
        'hr_r2': hr_r2,
        'hr_monthly_mb': hr_monthly_mb,
        'hr_cumulative_mb': hr_cumulative_mb,

        'feature_importance': pd.Series(
            final_model.feature_importances_,
            index=features).sort_values(ascending=False),
        'train_compare': train_compare,
        'hr_compare': compare
    }
    print(results[name]['feature_importance'].sort_values(ascending=False))

# creating a results dataframe so that they can be easily accessed and plotted
summary = pd.DataFrame.from_dict({
        name: {
            'CV RMSE': res['cv_rmse'],
            'CV R2': res['cv_r2'],
            'Final Thickness': res['final_thickness_change'],
            'Thickness Error': res['final_thickness_error'],
            'HR RMSE: ': res['hr_rmse'],
            'HR R2': res['hr_r2'],
        }
    for name, res in results.items()
    }, orient='index')

# labellng the axes of the plots with all model outputs presented
ax_train_monthly.set_title('Observed vs Modelled Monthly MB')
ax_train_monthly.set_xlabel('Time')
ax_train_monthly.set_ylabel('MB (m w.e./yr)')
ax_train_cum.set_title('Observed vs Modelled Cumulative Thickness Change')
ax_train_cum.set_xlabel('Time')
ax_train_cum.set_ylabel('Cumulative Thickness Change (m)')
ax_hr_monthly.set_title('High Resolution Monthly MB')
ax_hr_monthly.set_xlabel('Time')
ax_hr_monthly.set_ylabel('MB (m w.e./yr)')
ax_hr_cum.set_title('High Resolution Thickness Change')
ax_hr_cum.set_xlabel('Time')
ax_hr_cum.set_ylabel('Cumulative Thickness Change (m)')

for ax in [ax_train_monthly, ax_hr_monthly, ax_train_cum, ax_hr_cum]:
    ax.legend()

# plotting the graphs with all model predictions on them
fig_train_monthly.show()
fig_train_cum.show()
fig_hr_monthly.show()
fig_hr_cum.show()

print(summary.round(3).to_string())
print('\nFINAL THICKNESS CHANGES')
for name, res in results.items():
    print(
        f"{name}: "
        f"{res['final_thickness_change']:.2f}"
        f" ± "
        f"{res['final_thickness_error']:.2f} m"
    )

# plotting modelled monthly mass balance against observed
fig, axes = plt.subplots(
    3, 2,
    figsize=(16, 16),
    sharex=True,
    sharey=True
)
axes = axes.flatten()
for ax, (name, res) in zip(
    axes, results.items()
):
    compare = res['train_compare']
    error = 1.96 * res['cv_rmse']
    ax.plot(
        compare.index,
        compare['khan'],
        color='black',
        linewidth=0.75,
        label='Observed'
    )
    ax.plot(
        compare.index,
        compare['pred'],
        color='blue',
        linewidth=0.75,
        label='Modelled'
    )
# fill between - translucent plot to show the prediction error and visualise how it varies between feature sets
    ax.fill_between(
        compare.index,
        compare['pred'] - error,
        compare['pred'] + error,
        alpha=0.2,
        color='red'
    )
    ax.set_title(
        f'{name}\n'
        f"CV R2={res['cv_r2']:.3f}",
        fontsize=10
    )

for ax in axes[len(results):]:
    ax.axis('off')

for ax in axes:
    ax.tick_params(axis='both', labelsize=6)

fig.suptitle('Observed vs Modelled Monthly MB')
fig.supxlabel('Date', fontsize=8)
fig.supylabel('Monthly Mass Balance (m w.e./yr)', fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='center',
    ncol=3
)
plt.tight_layout()
plt.show()

# same concept as the plot above, but plotting cumulative thickness change
fig, axes = plt.subplots(
    3, 2,
    figsize=(16, 16),
    sharex = True,
    sharey= True
)

axes = axes.flatten()
for ax, (name, res) in zip(
    axes, results.items()
):
    compare = res['train_compare']
    obs_cum = compare['khan'].cumsum()
    pred_cum = compare['pred'].cumsum()
    error = 1.96 * res['cv_rmse']
    cum_error = (
        error * np.sqrt(
        np.arange(
            1, len(pred_cum)+1)
    )
    )
    ax.plot(
        obs_cum.index,
        obs_cum.values,
        color='black',
        linewidth=0.75,
        label='Observed'
    )

    ax.plot(
        pred_cum.index,
        pred_cum.values,
        color='blue',
        linewidth=0.75,
        label='Modelled'
    )
# fill between this time showing cumulative error
    ax.fill_between(
        pred_cum.index,
        pred_cum - cum_error,
        pred_cum + cum_error,
        color='red',
        alpha=0.25
    )

    ax.set_title(
        f'{name}\n'
        f'{pred_cum.iloc[-1]:.2f}'
        f' ± '
        f'{cum_error[-1]:.2f} m',
        fontsize=10
    )

for ax in axes[len(results):]:
    ax.axis('off')

for ax in axes:
    ax.tick_params(axis='both', labelsize=6)

fig.suptitle('Observed vs Modelled Cumulative Thickness Change')
fig.supxlabel('Date', fontsize=8)
fig.supylabel('Cumulative Thickness Change (m)', fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='center',
    ncol=3
)
plt.tight_layout()
plt.show()

# renaming the features for the figure
feature_name_map = {
    'khan_monthly_mb_lag1': 'Lagged MB',
    'PDD_mean': 'PDD',
    'PDD_mean_lag3': 'Lagged PDD (3)',
    'pdd_3mo': '3 month PDD',
    'month_cos': 'Cosine Month',
    'month_sin': 'Sine Month',
    'snowfall_mm': 'Snowfall',
    'albedo_mean': 'Albedo',
    'PDD_mean_lag1': 'Lagged PDD (1)',
    'rainfall_mm': 'Rainfall',
    'snow_protection': 'Snow Cover',
    'northernness': 'Aspect',
    'slope_mean': 'Slope',
    'rain_melt': 'Rain Melt',
    'snow_3mo': '3 Month Snow'
}

# shap analysis - cutting off the number of predictions at 5000
# 5000 is an appropriate sample size to get a good representation, without overloading computational requirements
shap_results = {}

for name, features in feature_sets.items():
    print(f'\n{name}')
    model = results[name]['model']
    X_shap = (
        df.dropna(subset = features + ['khan_monthly_mb'])[features]
    )
    X_shap = X_shap.rename(columns=feature_name_map)
    if len(X_shap) > 5000:
        X_shap = X_shap.sample(5000, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)

    shap_results[name] = {
        'X': X_shap,
        'shap_values': shap_values
    }

n_models = len(shap_results)
ncols = 2
nrows = math.ceil(n_models / ncols)

# creating a shrink shap text function - ensures the axis labels on the shap plot can be set to the desired size
def shrink_shap_text(fontsize=6):
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)
    for text in ax.texts:
        text.set_fontsize(fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    for obj in ax.findobj():
        if hasattr(obj, 'set_fontsize'):
            try:
                obj.set_fontsize(fontsize)
            except Exception:
                pass

# shap plot - only doing shap analysis on the model with all features
# unnecessary to perform shap on all feature sets - this improves computational efficiency
res = shap_results['All features']
plt.figure(figsize=(8,6))

shap.plots.beeswarm(
    res['shap_values'],
    max_display=10,
    show=False
)
shrink_shap_text(8)
plt.title('SHAP Beeswarm - All Features', fontsize = 12)
plt.tight_layout()
plt.close()

best_model = results['All features']['model']
projection_rmse =  results['All features']['cv_rmse']
projection_error = 1.96* projection_rmse

ssp_results = {}

# scenario projection section
# creating a loop that can be used on any dataset necessary
for scenario_name, d in zip(
    ['SSP1-2.6', 'SSP2-4.5', 'SSP5-8.5'],
    [df_126, df_245, df_585]
):
# creating the lagged monthly mass balance variable for future projections
# lagged variable is the prior month's prediction
    d['predicted_mb'] = np.nan
    d['khan_monthly_mb_lag1'] = np.nan
    df_model = d.copy()
    unique_dates = sorted(df_model['date'].unique())

    for n, date in enumerate(unique_dates):
        mask = df_model['date'] == date
        current = df_model.loc[mask].copy()

        if n == 0:
            current['khan_monthly_mb_lag1'] = (
                current['id_grid'].map(last_mb_lookup).fillna(
                    df['khan_monthly_mb'].mean()
                ))
        else:
            prev_date = unique_dates[n - 1]
            prev_predictions = (
                df_model.loc[
                    df_model['date'] == prev_date, ['id_grid', 'predicted_mb']
                ].set_index('id_grid')['predicted_mb']
            )
            current['khan_monthly_mb_lag1'] = (
                current['id_grid'].map(prev_predictions)
            )

        current['khan_monthly_mb_lag1'] = (
            current['khan_monthly_mb_lag1'].fillna(
                df['khan_monthly_mb'].mean()))
# best model used to predict the future mass balance
        preds = best_model.predict(current[full_features].fillna(0))
        df_model.loc[mask, 'predicted_mb'] = preds

    future_monthly_mb = (df_model.groupby('date')['predicted_mb'].mean().sort_index())
    future_cumulative_mb = future_monthly_mb.cumsum()
# cumulative error calculation - based on projection error
    cum_error = (
        projection_error * np.sqrt(
        np.arange(
            1, len(future_cumulative_mb)+1
        )
    )
    )

# storing all necessary results for each scenario
# creating an SSP projection dataset
    ssp_results[scenario_name] = {
        'monthly_mb': future_monthly_mb,
        'cumulative_mb': future_cumulative_mb,
        'monthly_upper': future_monthly_mb + projection_error,
        'monthly_lower': future_monthly_mb - projection_error,
        'cum_upper': future_cumulative_mb + cum_error,
        'cum_lower': future_cumulative_mb - cum_error,
        'final_change': future_cumulative_mb.iloc[-1],
        'final_error': cum_error[-1],
        'df': df_model
    }

print('\nProjected cumulative thickness change')

for scenario, res in ssp_results.items():

    print(
        f'{scenario}: '
        f'{res["final_change"]:.2f}'
        f' ± '
        f'{res["final_error"]:.2f} m'
    )

fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharey=True)
axes = axes.flatten()

time_index = next(iter(ssp_results.values()))['monthly_mb'].index
split_dates = np.array_split(time_index, 4)

for ax, period in zip(axes, split_dates):
    start, end = period[0], period[-1]

    for scenario, res in ssp_results.items():
        data=res['monthly_mb'].loc[start:end]
        ax.plot(
            data.index,
            data.values,
            label=scenario,
            linewidth=0.75
        )
        ax.fill_between(
            data.index,
            res['monthly_lower'].loc[start:end],
            res['monthly_upper'].loc[start:end],
            alpha=0.15
        )
    ax.set_title(f'{start.year} - {end.year}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mass Balance (m w.e./yr)')
axes[0].legend()
fig.suptitle('Projected Monthly Mass Balance', fontsize=14)
plt.tight_layout()
plt.show()
print('Plot done!')

# plotting the cumulative thickness change for each scenario
# axes split into four for visual clarity
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

time_index = next(iter(ssp_results.values()))['cumulative_mb'].index
split_dates = np.array_split(time_index, 4)

for ax, period in zip(axes, split_dates):
    start, end = period[0], period[-1]
    all_vals=[]
    for scenario, res in ssp_results.items():
        data = res['cumulative_mb'].loc[start:end]
        all_vals.extend(res['cumulative_mb'].loc[start:end].values)
        ax.plot(
            data.index,
            data.values,
            label=scenario,
            linewidth=0.75)
        # showing the error for projections
        ax.fill_between(
            data.index,
            res['cum_lower'].loc[start:end],
            res['cum_upper'].loc[start:end],
            alpha=0.15
        )
    ax.set_title(f'{start.year}-{end.year}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Thickness Change (m)')
    ymin = min(all_vals)
    ymax = max(all_vals)
    padding = 0.05 * (ymax-ymin)
    ax.set_ylim(ymin-padding, ymax+padding)
axes[0].legend()
fig.suptitle('Projected Relative Thickness Change', fontsize=14)
plt.tight_layout()
plt.show()
print('Plot done!')

# updated the high resolution grids so that they can be spatially represented
folder = r'C:\Users\p09367ap\OneDrive - The University of Manchester\ERP\high res grids'

# the function below joins the projected mass balance value for each grid onto the corresponding spatialised grid cell
mb_cols = [col for col in hr_df.columns
           if col.startswith('mb_')]
for layer_name in hr_df['layer'].unique():
    print(f'Processing {layer_name}')
    gpkg_path = os.path.join(
        folder, f'{layer_name}.gpkg'
    )
    if not os.path.exists(gpkg_path):
        print(f'Could not find: {gpkg_path}')
        continue

    gdf = gpd.read_file(gpkg_path)
    layer_predictions = hr_df.loc[
        hr_df['layer'] == layer_name,
        ['id_grid'] + mb_cols
    ]
    gdf = gdf.merge(
        layer_predictions, on='id_grid', how='left'
    )

    output_path = os.path.join(
        folder, f'{layer_name}_predicted.gpkg'
    )

    gdf.to_file(
        output_path, driver='GPKG'
    )
    print(f'Saved {output_path}')