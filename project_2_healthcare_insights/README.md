# Healthcare Data Insights

A comprehensive healthcare analytics project that analyzes patient data, disease patterns, admission trends, and recovery rates.

## Overview

This project provides valuable insights into healthcare operations by analyzing patient demographics, disease prevalence, hospital admission patterns, and recovery outcomes. The analysis helps healthcare providers understand patient populations and improve care delivery.

## Dataset

- **File:** `dataset_healthcare.csv`
- **Description:** Contains patient records with the following fields:
  - `patient_id`: Unique patient identifier
  - `age`: Patient age
  - `disease`: Disease diagnosis
  - `admission_date`: Date of hospital admission
  - `recovery_days`: Number of days until recovery
  - `recovered`: Recovery status (True/False)
  - `recovery_rate`: Recovery rate percentage

## Visualizations

This project includes **6 comprehensive, advanced visualizations** providing multi-dimensional insights into healthcare operations and patient outcomes.

### 1. Comprehensive Admission Trends Dashboard
- **File:** `exports/1_comprehensive_admission_trends.png`
- **Description:** Multi-panel dashboard featuring:
  - Monthly admission trends with mean reference line
  - Admissions by day of week analysis
  - Quarterly admission patterns
  - Top 5 diseases monthly admission trends

### 2. Age Distribution & Demographics Analysis
- **File:** `exports/2_age_demographics_analysis.png`
- **Description:** Comprehensive demographic analysis including:
  - Patient age distribution histogram with statistical measures
  - Age group distribution by disease (stacked bar chart)
  - Age vs Recovery Days scatter plot (color-coded by recovery rate)
  - Recovery metrics by age group comparison

### 3. Disease Analysis & Prevalence Dashboard
- **File:** `exports/3_disease_analysis_dashboard.png`
- **Description:** Deep dive into disease patterns featuring:
  - Disease prevalence - patient count by disease
  - Disease prevalence by age group heatmap
  - Recovery metrics by disease (recovery days, recovery rate, recovery percentage)
  - Recovery days distribution box plots by disease

### 4. Recovery Rate Comprehensive Analysis
- **File:** `exports/4_recovery_rate_analysis.png`
- **Description:** Advanced recovery analysis including:
  - Recovery rate distribution histogram
  - Average recovery rate by disease (horizontal bar chart)
  - Recovery days vs recovery rate scatter (separated by recovery status)
  - Recovery metrics by age group comparison

### 5. Temporal Patterns & Seasonality Analysis
- **File:** `exports/5_temporal_patterns_analysis.png`
- **Description:** Time-based pattern analysis featuring:
  - Monthly recovery trends (recovery percentage and recovery days)
  - Top 3 diseases monthly trends
  - Average recovery days by month
  - Recovery rate heatmap: Disease × Month

### 6. Comprehensive Statistical Summary Dashboard
- **File:** `exports/6_comprehensive_statistical_summary.png`
- **Description:** Executive summary dashboard with:
  - Key healthcare statistics (KPI cards)
  - Top 5 diseases by patient count
  - Recovery days, recovery rate, and age distribution histograms
  - Recovery status pie chart
  - Quarterly performance comparison (multi-metric analysis)

## Key Insights

- Patient demographics and age distribution
- Disease prevalence and patterns
- Hospital admission trends and seasonality
- Recovery rates by disease type
- Factors affecting patient recovery

## Project Structure

```
project_2_healthcare_insights/
├── README.md
├── healthcare_analytics_dashboard.py
├── dataset_healthcare.csv
└── exports/
    ├── 1_comprehensive_admission_trends.png
    ├── 2_age_demographics_analysis.png
    ├── 3_disease_analysis_dashboard.png
    ├── 4_recovery_rate_analysis.png
    ├── 5_temporal_patterns_analysis.png
    └── 6_comprehensive_statistical_summary.png
```

## Usage

### Running the Analysis

1. **Prerequisites:** Install required Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

2. **Generate Visualizations:**
   ```bash
   cd project_2_healthcare_insights
   python healthcare_analytics_dashboard.py
   ```

3. **Output:** All visualizations will be saved to the `exports/` folder as high-resolution PNG files (300 DPI)

### Customization

The `healthcare_analytics_dashboard.py` script is fully customizable:
- Modify `FIG_SIZE` and `DPI` constants for different output sizes
- Adjust color schemes and styling
- Add or remove visualization sections
- Customize age group bins and time periods

## Technologies

- **Python 3.x**
- **Data Analysis:**
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing
  - scipy - Statistical functions
- **Visualization:**
  - matplotlib - Advanced plotting and customization
  - seaborn - Statistical visualizations and heatmaps
- **Features:**
  - High-resolution output (300 DPI)
  - Professional styling and color schemes
  - Multi-panel dashboards
  - Statistical analysis integration

## Applications

- Healthcare resource planning
- Disease outbreak monitoring
- Patient care optimization
- Hospital capacity management
- Recovery outcome analysis

