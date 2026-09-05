# XGBoost-glacier-modelling
This repository contains the complete code, a workflow overview and documentation for a project assessing the performance of XGBoost frameworks in glacier modelling, looking specifically at the Bussemand catchment in Greenland. The project utilises several publicly available data sources to develop the XGBoost model.

# Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Workflow](#workflow)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)

# Project Overview
The aim of this project was to develop and apply an XGBoost framework to the Bussemand catchment in Greenland, to assess its performance in several aspects. To assess the reconstructive performance of the model, a traditional distributed temperature-index model was built, allowing direct comparisons between performances of different models. The model was applied to analyse the influence of different variables on mass balance variation, done through quantifying the percentage of variation caused by each variable, and SHAP analysis to reveal predictive influence of each feature. The model was then applied to a higher resolution dataset, and a dataset of projected climate changes in the Bussemand catchment, showing the model's performance for increasing the resolution of existing datasets, and projeting future glacier behaviour respectively.

# Data Description

# Workflow
The project has three main phases
1. **Phase 1: data derivation**
   - Daily PROMICE temperature observations were used, and the ERA5 daily temperature estimates were used to support interpolation of missing data. These values were used to derive monthly positive degree days
   - The IceBridge DEM is used as a foundational raster to support lapse-rate corrections of temperature throughout the entire time series
   - Monthly Landsat-8/9 images were used to calculate glacier-wide albedo for the months in the time series, and missing values were filled with a random-forest regression trained on existing data
   - Precipitation data was derived from monthly ERA5 precipitation outputs and partitioned for snowfall and rainfall based on temperature
   - Future climate data was derived from bias-corrected CMIP6 projections, and the projected climate was used in the random-forest regression to project future albedo values
2. **Phase 2: model building, training and analysis**
   - A distributed temperature-index model was developed, applied to the prepared dataset and used as a baseline for performance comparison
   - Interaction and lagged variables were derived
   - Several XGBoost models were built and trained, each trained on feature sets of varying complexity to assess the impact of different feature sets on model performance
   - Model performance was analysed by plotting prediction outputs against observed values, and cross-validation R^2 scores demonstrated generalisability of the models
   - The influence of features was analysed with calculations of the percentage of model variation driven by each feature, and SHAP analysis to show predictive influence
3. **Phase 3: application of the XGBoost model**
   - Every XGBoost model trained on the different feature sets was applied to the high resolution dataset
   - The XGBoost model that followed the observed data the closest, with the highest out-of-fold R^2 score was selected and applied to future climate projections of the Bussemand catchment

# Repository Structure

# Setup and Installation

# How to Run
