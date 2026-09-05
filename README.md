# XGBoost-glacier-modelling
This repository contains the complete code, a workflow overview and documentation for a project assessing the performance of XGBoost frameworks in glacier modelling, looking specifically at the Bussemand catchment in Greenland. The project utilises several publicly available data sources to develop the XGBoost model.

# Table of Contents
- [Project Overview](#project-overview)
- [Technical Stack](#technical-stack)
- [Workflow](#workflow)
- [Setup and Installation](#setup-and-installation)
- [Usage Guide](#usage-guide)

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
Not included in the scripts in this repository is the QGIS workflow. Temperature, albedo and the training mass balance data were all in raster form and analysed in QGIS. A 1 x 1 km grid of the Bussemand catchment was generated, and raster values were assigned to each grid using zonal statistics.
The project has three main phases
1. **Phase 1: data derivation**
   - Derived monthly positive degree days from PROMICE temperature observations, with ERA5-supported interpolation where necessary
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
├── scripts/
│   ├── 01-climate-cleaning.py
│   ├── 02-lapse_rate_calculations.py
│   ├── 03-precipitation_joining.py
│   ├── 04-Landsat-8-9_albedo_calculation.py
│   ├── 05-CMIP6_temperature_extraction.py
│   ├── 06-CMIP6_precipitation_extraction_and_joining.py
│   ├── 07-albedo_cleaning.py
│   └── 08-model_development.py

```

# Setup and Installation
## 1. Clone the Repository
```bash
git clone https://github.com/<username>/XGBoost-glacier-modelling.git
cd XGBoost-glacier-modelling
```
## 2. Create Python Environment (recommended)
```bash
python -m venv .venv
```
## 3. Install Required Packages
```bash
pip install -r requirements.txt
```

## 4. Download Input Data
The scripts require:
- Daily temperature observations with coordinates
- Monthly precipitation volume
- ERA5 observations from the region of observed temperature
- Landsat-8/9 imagery
- DEM for the relevant area
- CMIP6 projections
- Geodetic mass balance dataset of the desired catchment area
Any catchment can be studied using the workflow of this project, so datasets from any glacier catchment can be selected to fulfil the dataset criteria (provided they all cover the same catchment)
Links to datasets used in this specific project are provided in the **Datasets Used** section above

# Usage Guide
1. **Download necessary data**: acquire the datasets of your chosen catchment and download these into the relevant directory
2. **Dataset derivation**: apply the scripts in the following way:
   - 01-climate-cleaning.py: apply to the catchment daily climate observations with ERA5 readings to get a cleaned dataset of daily climate observations
   - 02-lapse_rate_calculations.py: apply to the daily climate observations to get monthly mean temperature and PDD rasters. Process these rasters using zonal statistics in QGIS
   - 03-precipitation_joining.py: join the monthly precipitation to the exported grid CSV with processed temperature values
   - 04-Landsat-8-9_albedo_calculation.py: apply to Landsat imagery to create rasters to be processed in QGIS
   - 05-CMIP6_temperature_extraction.py: apply to the NC files of CMIP6 temperature projections to get temperature projections until 2100. Apply 02-lapse_rate_calculations.py to these datasets to get temperature rasters for future climate
   - 06-CMIP6_precipitation_extraction_and_joining.py: apply to NC files of CMIP6 precipitation projections and join onto the grids with future climate projections
   - 07-albedo_cleaning.py: applied to all grid datasets. The observed data grid is used to train the random forest model, and then the trained random forest model is applied to future climate projections to fill future albedo projections
3. **Model Development**: use the 08-model_development.py script to build, plot the outputs and evaluates the temperature-index model and XGBoost model. The XGBoost model script is in a loop, and can be automatically applied to feature sets with various complexity
