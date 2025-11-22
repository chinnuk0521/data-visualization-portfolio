# Global Unemployment Analytics

A comprehensive global unemployment analysis project using real-world data from the World Bank and International Labour Organization (ILO). This project provides deep insights into unemployment trends, patterns, and correlations across countries, regions, and demographics.

## Overview

This project analyzes global unemployment data to provide actionable insights into:
- Global unemployment trends over time
- Regional and country-level comparisons
- Demographic breakdowns (age, gender, education)
- Sector-wise unemployment analysis
- Economic indicator correlations
- Pandemic impact analysis
- Predictive trends

## Data Sources

### Primary Data Sources:
1. **World Bank Open Data** - Unemployment, total (% of total labor force)
   - URL: https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS
   - Coverage: 200+ countries, 1991-2024
   
2. **ILO (International Labour Organization) Statistics**
   - URL: https://ilostat.ilo.org/data/
   - Detailed breakdowns by age, gender, education, sector

3. **Kaggle Alternative Datasets:**
   - Search for "unemployment" or "labor statistics" on Kaggle
   - Many pre-processed datasets available

## Data Download Instructions

### Option 1: World Bank Data (Recommended)
1. Visit: https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS
2. Click "Download" → "CSV"
3. Save as `dataset_unemployment.csv` in this project folder
4. The script will automatically process it

### Option 2: Using Python Script (Automated)
The `download_unemployment_data.py` script can download data programmatically:
```bash
python download_unemployment_data.py
```

### Option 3: Kaggle Dataset
1. Visit https://www.kaggle.com/datasets
2. Search for "global unemployment" or "world unemployment"
3. Download the dataset
4. Rename to `dataset_unemployment.csv` and place in this folder

## Dataset Structure

The dataset should contain:
- **Country/Country Name**: Country identifier
- **Country Code**: ISO country code
- **Year**: Year of data
- **Unemployment Rate**: Total unemployment (% of labor force)
- **Youth Unemployment**: Unemployment rate for ages 15-24
- **Male/Female Unemployment**: Gender-specific rates
- **Region**: Geographic region
- **Additional indicators**: GDP, inflation, etc. (if available)

## Project Structure

```
project_5_global_unemployment_analytics/
├── README.md
├── download_unemployment_data.py (optional - for automated download)
├── global_unemployment_analytics.py
├── dataset_unemployment.csv (to be downloaded)
└── exports/
    ├── 1_world_unemployment_overview.png
    ├── 2_global_trends_dashboard.png
    ├── 3_regional_comparison_analysis.png
    ├── 4_country_performance_analysis.png
    ├── 5_temporal_patterns_analysis.png
    ├── 6_statistical_summary.png
    ├── 7_comprehensive_heatmaps.png
    ├── 8_comprehensive_statistical_summary.png
    ├── 9_pandemic_impact_analysis.png
    ├── 10_predictive_trends.png
    ├── 11_volatility_stability_analysis.png
    ├── 12_country_clustering.png
    ├── 13_change_analysis.png
    └── 14_correlation_analysis.png
```

## Usage

### Step 1: Download the Data
Choose one of the data download options above.

### Step 2: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy requests
```

### Step 3: Run the Analysis
```bash
cd project_5_global_unemployment_analytics
python global_unemployment_analytics.py
```

### Step 4: View Results
All visualizations will be saved in the `exports/` folder as high-resolution PNG files (300 DPI).

## Visualizations

This project includes **14 comprehensive, advanced visualizations** providing multi-dimensional insights into global unemployment:

### 1. World Unemployment Overview
- **File:** `exports/1_world_unemployment_overview.png`
- **Description:** Multi-panel dashboard featuring:
  - Top 20 countries with highest unemployment
  - Top 20 countries with lowest unemployment
  - Global unemployment distribution
  - Average unemployment by region

### 2. Global Trends Dashboard
- **File:** `exports/2_global_trends_dashboard.png`
- **Description:** Time series analysis including:
  - Global average unemployment trends with standard deviation
  - Regional trends over time
  - Economic group trends (G7, G20, BRICS)
  - Year-over-year change analysis

### 3. Regional Comparison Analysis
- **File:** `exports/3_regional_comparison_analysis.png`
- **Description:** Regional insights featuring:
  - Regional unemployment statistics (min, mean, median, max)
  - Unemployment distribution box plots by region
  - Economic group comparison
  - Regional heatmap by decade

### 4. Country Performance Analysis
- **File:** `exports/4_country_performance_analysis.png`
- **Description:** Country-level analysis including:
  - Time series trends for top countries
  - Average unemployment by country rankings
  - Countries with highest volatility
  - Improvement/worsening analysis over time

### 5. Temporal Patterns & Decade Analysis
- **File:** `exports/5_temporal_patterns_analysis.png`
- **Description:** Time-based patterns featuring:
  - Average unemployment by decade
  - Data coverage by year
  - Unemployment distribution over time
  - Top countries × Years heatmap

### 6. Statistical Summary & Distribution
- **File:** `exports/6_statistical_summary.png`
- **Description:** Statistical analysis including:
  - Global unemployment rate distribution
  - Box plots by region
  - Key statistics summary
  - Yearly statistics with standard deviation

### 7. Comprehensive Heatmap Matrices
- **File:** `exports/7_comprehensive_heatmaps.png`
- **Description:** Multi-dimensional heatmaps:
  - Country × Year unemployment heatmap
  - Region × Decade heatmap
  - Economic Group × Year heatmap
  - Year-to-year correlation matrix

### 8. Comprehensive Statistical Summary Dashboard
- **File:** `exports/8_comprehensive_statistical_summary.png`
- **Description:** Executive summary with:
  - Key statistics (KPI cards)
  - Top 10 countries by unemployment
  - Average by region
  - Global distribution and trends
  - Comprehensive time series

### 9. Pandemic Impact Analysis
- **File:** `exports/9_pandemic_impact_analysis.png`
- **Description:** COVID-19 impact analysis:
  - Pre vs Post pandemic comparison
  - Regional pandemic impact
  - Country-level pandemic impact
  - Unemployment during pandemic period (2018-2024)

### 10. Predictive Trends & Forecasting
- **File:** `exports/10_predictive_trends.png`
- **Description:** Forecasting analysis featuring:
  - 5-year unemployment trend projection with confidence intervals
  - Moving average trends (3-year and 5-year)
  - Volatility analysis (rolling standard deviation)
  - Year-over-year growth rate analysis

### 11. Volatility & Stability Analysis
- **File:** `exports/11_volatility_stability_analysis.png`
- **Description:** Stability metrics including:
  - Most stable countries (lowest volatility)
  - Most volatile countries (highest variability)
  - Stability vs average unemployment scatter
  - Regional stability comparison

### 12. Country Clustering & Grouping
- **File:** `exports/12_country_clustering.png`
- **Description:** Country grouping analysis:
  - Country distribution by unemployment level (pie chart)
  - Economic group performance comparison
  - Top 10 vs Bottom 10 performers
  - Country distribution by region

### 13. Change Analysis - Improvement & Worsening
- **File:** `exports/13_change_analysis.png`
- **Description:** Change tracking featuring:
  - Top 15 countries - most improved (largest decrease)
  - Top 15 countries - most worsened (largest increase)
  - Distribution of percentage changes
  - Initial vs Final unemployment scatter (color-coded by change)

### 14. Advanced Correlation & Relationship Analysis
- **File:** `exports/14_correlation_analysis.png`
- **Description:** Correlation analysis including:
  - Year-to-year correlation matrix
  - Regional correlation matrix
  - Unemployment vs Time with trend line
  - Country similarity matrix (correlation)

1. **World Unemployment Map** - Choropleth map showing unemployment by country
2. **Global Trends Dashboard** - Multi-panel time series analysis
3. **Demographic Breakdown** - Age and gender analysis
4. **Education Impact Analysis** - Unemployment by education level
5. **Sector-wise Analysis** - Industry-specific unemployment
6. **Regional Comparison** - Continent and economic group comparisons
7. **Gender Gap Analysis** - Male vs female unemployment trends
8. **Youth Unemployment Focus** - Special analysis for 15-24 age group
9. **Economic Correlation** - Unemployment vs GDP, inflation
10. **Pandemic Impact Analysis** - Pre/post-COVID comparison
11. **Country Rankings** - Top and bottom performers
12. **Predictive Trends** - Forecasting and trend analysis
13. **Comprehensive Heatmaps** - Multi-dimensional matrices
14. **Statistical Summary Dashboard** - Executive summary with KPIs

## Technologies

- **Python 3.x**
- **Data Analysis:**
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing
  - scipy - Statistical functions
- **Visualization:**
  - matplotlib - Advanced plotting and geographic visualizations
  - seaborn - Statistical visualizations and heatmaps
- **Data Sources:**
  - World Bank API / CSV downloads
  - ILO Statistics
  - Kaggle datasets

## Key Features

- **Real-world data** from authoritative sources
- **Large dataset** (5000-10000+ records)
- **Geographic visualizations** (world maps)
- **Multi-dimensional analysis** (country, time, demographics, sectors)
- **Advanced statistical analysis** (correlations, forecasting)
- **High-resolution outputs** (300 DPI)
- **Comprehensive documentation**

## Applications

- Economic policy analysis
- Labor market research
- Regional development planning
- Educational policy insights
- Gender equality analysis
- Youth employment programs
- Economic forecasting

## Notes

- The script is designed to handle real-world data with missing values
- Data preprocessing is included to clean and standardize country names
- The analysis adapts to available data dimensions
- All visualizations are production-ready for presentations

