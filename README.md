# XGBoost-glacier-modelling
This repository contains the complete code, a workflow overview and documentation for a project assessing the performance of XGBoost frameworks in glacier modelling, looking specifically at the Bussemand catchment in Greenland. The project utilises several publicly available data sources to develop the XGBoost model.

# Table of Contents
- [Project Overview](#project-overview)
- [Technical Stack](#technical-stack)
- [Workflow](#workflow)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)

# Project Overview
The project developed an XGBoost model to reconstruct and predict glacier mass balance, applied here in the Bussemand catchment in Greenland, but designed to be applicable anywhere. Performance was compared with a traditional temperature-index model, and feature importance and SHAP analyses were used to assess controls on mass balance variability. The model was applied to a higher resolution dataset of the Bussemand catchment, and future climate projections to evaluate its scalability and predictive capability under future climate conditions.

# Technical Stack
## Programming Environment
- Python 3.13
### Core Libraries
- Numpy 2.4.2: for core numerical operations
- Pandas 3.0.3: data processing and manipulation
- XGBoost 3.3.0: training and applying the new glacier model
- SHAP 0.52.0: analysing feature importance
- Matplotlib 3.10.8: graphical visualisations
- Scipy 1.17.0: interpolation functions, numerical methods and optimisation routines
- Scikit-learn 1.8.0: machine learning, cross-validation metrics, and random-forest regression
- Rasterio 1.5.0: reading, writing and processing raster data
- Geopandas 1.1.4: vector data handling
- XArray 2026.4.0: used for NetCDF climate datasets
- OS 3.3.0: creating folders and reading path files
## Datasets used
- Khan et al. (2025) geodetic mass balance observations of the Greenland ice sheet (https://doi.org/10.5194/essd-17-3047-2025.)
- PROMICE daily temperature observations in °C (https://doi.org/10.22008/FK2/IW73UU)
- IceBridge BedMachine DEM V6 for altitude (m above sea level), slope and aspect (https://doi.org/10.5067/6B6B225B8V2D)
- Landsat-8/9 Imagery bands 2, 3, 5, 6 and 7 for albedo calculations (https://doi.org/10.5066/P9OGBGM6)
- ERA5 daily reanalysis data for temperature interpolation (2m surface temperature), and glacier-wide temperature (daily precipitation in mm) (https://doi.org/10.24381/cds.4991cf48)
- CMIP6 temperature and precipitation projections (same units as above) (https://doi.org/10.24381/cds.c866074c.)

# Workflow
The project has three main phases
1. **Phase 1: data derivation**
   - DDerived monthly positive degree days from PROMICE temperature observations, with ERA5-supported interpolation where necessary
   - Applied lapse-rate corrections using the IceBridge DEM
   - Derived precipitation volumes from ERA5 precipitation data, and calculated snowfall-rainfall percentages from temperature
   - Calculated monthly glacier-wide albedo from Landsat-8/9 imagery, with missing values estimated using random forest regression
   - Bias-corrected CMIP6 projections were used for future climate, and this future climate supported projection of future albedo using the trained random-forest model
2. **Phase 2: model building, training and analysis**
   - A distributed temperature-index model was developed as a baseline
   - The necessary seasonality, interaction and lagged variables were derived
   - Several XGBoost models were trained with progressively complex feature sets
   - SHAP analyses and model variance contribution calculations assessed feature importance
3. **Phase 3: application of the XGBoost model**
   - Every trained XGBoost model was applied to the high resolution dataset
   - The best performing model, based on agreement with observed data and out-of-fold R², was selected and applied to the future climate projections of the Bussemand catchment

# Repository Structure
```
├── README.md
├── requirements.txt
├── notebooks/
│   ├──

```

# Setup and Installation

# How to Run
