# Building Energy Optimization Analysis - AlphaBuilding Dataset

A comprehensive research-style analysis of the AlphaBuilding Synthetic Operation Dataset for building energy optimization, developed for the EnergyEta Data Science Internship Case Study.

## Overview

This project analyzes the AlphaBuilding Synthetic Operation Dataset to evaluate its suitability for developing generalizable building energy optimization solutions. The analysis includes:

- **Data Understanding and Scoping**: Dataset origin, scope, and relevance
- **Exploratory Data Analysis (EDA)**: Data quality assessment, preprocessing, and visualizations
- **Mathematical & Statistical Modeling**: Time-series decomposition and predictive modeling
- **Conclusions and Recommendations**: Final verdict on dataset suitability

## Dataset Information

**AlphaBuilding Synthetic Operation Dataset**
- **Source**: LBNL (Lawrence Berkeley National Laboratory)
- **URL**: https://lbnl-eta.github.io/AlphaBuilding-SyntheticDataset/
- **Size**: 1.2 TB compressed (HDF5 format)
- **Content**: 
  - HVAC, lighting, and miscellaneous electric loads
  - Occupant counts and environmental parameters
  - 10-minute interval data
  - 1,395 annual simulations
  - 3 climate zones: Miami, San Francisco, Chicago
  - 3 energy efficiency levels

## Project Structure

```
project_6_building_energy_optimization/
├── README.md
├── building_energy_optimization_analysis.py
├── dataset_alphabuilding.csv (sample data)
└── exports/
    └── [visualization outputs]
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

## Usage

```bash
cd project_6_building_energy_optimization
python building_energy_optimization_analysis.py
```

## Analysis Sections

### I. Data Understanding and Scoping (20%)
- **Data Explanation**: Origin, scope, and key features of AlphaBuilding dataset
- **Relevance**: Why this dataset is relevant to EnergyEta's goals
- **Problem Statement**: Forecasting next-day peak energy demand

### II. Exploratory Data Analysis (30%)
- **Data Quality Assessment**: Missing values, outliers, data granularity
- **Preprocessing Steps**: 
  - Handling missing values
  - Feature engineering (temporal features: hour, day_of_week, season)
  - Outlier detection using IQR method
  - Daily aggregation for forecasting
- **Visualizations**:
  1. Multi-Variable Time Series Analysis (Energy, Temperature, Occupancy)
  2. Multi-Variable Correlation Analysis (Heatmaps, patterns, relationships)

### III. Mathematical & Statistical Implementation (40%)

**Technique 1: Time-Series Decomposition (Descriptive Statistics)**
- Decomposes energy consumption into trend, seasonal, and residual components
- Identifies anomalies in residual component
- Visualizes all components with annotations

**Technique 2: Predictive Modeling - Next-Day Peak Energy Forecasting**
- Linear regression model for forecasting
- Features: previous day's peak, total energy, temperature, occupancy, temporal features
- Model evaluation: MAE, RMSE, R² scores
- Residuals analysis
- Limitations and improvement suggestions

### IV. Conclusion and Recommendation (10%)
- Final verdict on dataset suitability
- Effort vs. insight value assessment
- Actionable recommendations for EnergyEta

## Generated Visualizations

1. **1_multi_variable_time_series_analysis.png**
   - Energy consumption breakdown (whole building, HVAC, lighting)
   - Temperature and occupancy patterns
   - 3D relationship: Energy vs Occupancy vs Temperature
   - Annotations highlighting peaks and anomalies

2. **2_multi_variable_correlation_analysis.png**
   - Correlation heatmap matrix
   - Weekday vs Weekend energy patterns
   - Climate zone and efficiency level analysis
   - Multi-variable scatter with trend lines

3. **3_time_series_decomposition.png**
   - Original time series
   - Trend component
   - Seasonal component
   - Residual component with anomaly annotations

4. **4_predictive_modeling_forecasting.png**
   - Training and test set predictions
   - Time series of forecasts vs actuals
   - Residuals analysis with outlier annotations

5. **5_final_conclusion_recommendation.png**
   - Key findings summary
   - Model performance metrics
   - Data quality assessment
   - Final recommendation

## Key Features

- **Multi-variable visualizations**: 3-5 variables plotted together with annotations
- **Advanced annotations**: Highlights peaks, anomalies, and insights
- **Comprehensive analysis**: Covers all required sections
- **Production-ready**: High-resolution outputs (300 DPI)
- **Automatic sample data**: Creates synthetic data if dataset not available

## Notes

- The script includes sample data generation for demonstration
- Replace with actual AlphaBuilding HDF5 data for production analysis
- All visualizations are saved as high-resolution PNG files (300 DPI)
- The analysis follows the EnergyEta case study requirements exactly

