# Sales Analytics Dashboard

A comprehensive sales analytics project that analyzes sales data and visualizes revenue growth trends.

## Overview

This project provides insights into sales performance through data analysis and visualization. It tracks revenue growth patterns and helps identify key trends in sales data.

## Dataset

- **File:** `dataset_sales.csv`
- **Description:** Contains sales data with the following fields:
  - `date`: Date of sale
  - `category`: Product category (Home & Garden, Sports, Food & Beverages, Clothing, Electronics)
  - `region`: Sales region (Central, East, West, South, North)
  - `revenue`: Revenue amount
  - `orders`: Number of orders

## Visualizations

This project includes **6 comprehensive, advanced visualizations** providing multi-dimensional insights into sales performance and trends.

### 1. Comprehensive Revenue Trends Dashboard
- **File:** `exports/1_comprehensive_revenue_trends.png`
- **Description:** Multi-panel dashboard featuring:
  - Monthly revenue and orders trends (dual-axis)
  - Revenue by day of week analysis
  - Quarterly revenue comparison
  - Monthly revenue growth rate (color-coded positive/negative)

### 2. Category & Region Performance Analysis
- **File:** `exports/2_category_region_analysis.png`
- **Description:** Performance analysis including:
  - Revenue by product category (horizontal bar chart)
  - Revenue by region (horizontal bar chart)
  - Category × Region revenue heatmap
  - Orders by product category

### 3. Revenue vs Orders Analysis
- **File:** `exports/3_revenue_orders_analysis.png`
- **Description:** Revenue-orders correlation analysis featuring:
  - Revenue vs Orders scatter plot by category
  - Average revenue per order by category
  - Revenue vs Orders scatter plot by region
  - Average revenue per order by region

### 4. Temporal Patterns & Seasonality Analysis
- **File:** `exports/4_temporal_patterns_analysis.png`
- **Description:** Time-based pattern analysis including:
  - Top 5 categories monthly revenue trends
  - Monthly revenue trends by region
  - Average revenue by month
  - Category × Month revenue heatmap

### 5. Performance Metrics & KPIs Dashboard
- **File:** `exports/5_performance_metrics_kpis.png`
- **Description:** Key performance indicators featuring:
  - Revenue distribution histogram
  - Orders distribution histogram
  - Revenue per order distribution
  - Top 10 revenue days analysis

### 6. Comprehensive Statistical Summary Dashboard
- **File:** `exports/6_comprehensive_statistical_summary.png`
- **Description:** Executive summary dashboard with:
  - Key sales statistics (KPI cards)
  - Top 5 categories by revenue
  - Revenue by region
  - Revenue, orders, and revenue per order distribution histograms
  - Quarterly performance comparison (multi-metric analysis)

## Project Structure

```
project_1_sales_analytics/
├── README.md
├── sales_analytics_dashboard.py
├── sales_dashboard.ipynb
├── dataset_sales.csv
└── exports/
    ├── 1_comprehensive_revenue_trends.png
    ├── 2_category_region_analysis.png
    ├── 3_revenue_orders_analysis.png
    ├── 4_temporal_patterns_analysis.png
    ├── 5_performance_metrics_kpis.png
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
   cd project_1_sales_analytics
   python sales_analytics_dashboard.py
   ```

3. **Output:** All visualizations will be saved to the `exports/` folder as high-resolution PNG files (300 DPI)

### Customization

The `sales_analytics_dashboard.py` script is fully customizable:
- Modify `FIG_SIZE` and `DPI` constants for different output sizes
- Adjust color schemes and styling
- Add or remove visualization sections
- Customize time periods and aggregation methods

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

