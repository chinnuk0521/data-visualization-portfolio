# Enterprise Data Intelligence

A comprehensive enterprise-level data analysis project that provides business intelligence insights across multiple departments and operational metrics.

## Overview

This project analyzes enterprise-wide data to provide actionable insights into revenue, costs, profitability, human resources, customer metrics, and operational efficiency. The analysis helps organizations make data-driven decisions and optimize business performance.

## Dataset

- **File:** `dataset_enterprise.csv`
- **Description:** Contains comprehensive enterprise data with the following fields:
  - `date`: Date of record
  - `department`: Department name (Marketing, R&D, Customer Service, HR, Finance, etc.)
  - `revenue`: Department revenue
  - `costs`: Operational costs
  - `profit_margin`: Profit margin percentage
  - `profit`: Net profit
  - `employees_hired`: Number of employees hired
  - `employees_attrition`: Number of employees who left
  - `employee_performance`: Employee performance score
  - `product_category`: Product category (Product A, B, C, D, E)
  - `market_segment`: Market segment (Consumer, Government, Enterprise, SMB)
  - `sales_volume`: Sales volume
  - `customer_churn_probability`: Probability of customer churn
  - `retention_rate`: Customer retention rate
  - `daily_efficiency`: Daily operational efficiency score
  - `operational_cost`: Operational cost breakdown

## Visualizations

This project includes **11 comprehensive, advanced visualizations** providing multi-dimensional insights into enterprise operations.

### 1. Revenue Trend
- **File:** `exports/1_revenue_trend.png`
- **Description:** Visualizes revenue trends over time, showing growth patterns and identifying revenue drivers

### 2. Multi-Dimensional Correlation Heatmap
- **File:** `exports/2_correlation_heatmap.png`
- **Description:** Advanced correlation matrix showing relationships between all key enterprise metrics (revenue, costs, profit, HR metrics, customer metrics, operational efficiency). Uses triangular heatmap with color-coded correlation coefficients to identify strong positive/negative relationships.

### 3. Advanced Time Series Dashboard
- **File:** `exports/3_advanced_time_series_dashboard.png`
- **Description:** Comprehensive multi-panel dashboard featuring:
  - Revenue vs Costs dual-axis time series with filled areas
  - Monthly profit analysis with color-coded positive/negative bars
  - Employee performance and retention rate trends
  - Sales volume vs customer churn probability correlation
  - Operational efficiency trend with mean reference line

### 4. Department Performance Radar Chart
- **File:** `exports/4_department_radar_chart.png`
- **Description:** Multi-dimensional radar/spider chart comparing all departments across 6 key metrics (Revenue, Profit Margin, Employee Performance, Retention Rate, Daily Efficiency, Sales Volume). Normalized to 0-100 scale for fair comparison.

### 5. Product-Market Segment Heatmap
- **File:** `exports/5_product_market_heatmap.png`
- **Description:** Dual heatmap matrix showing:
  - Revenue distribution across product categories and market segments
  - Profit analysis by product-market combinations
  - Identifies high-performing product-segment pairs

### 6. Employee Metrics Analysis Dashboard
- **File:** `exports/6_employee_metrics_analysis.png`
- **Description:** Comprehensive HR analytics featuring:
  - Hiring vs Attrition comparison by department
  - Net employee change visualization
  - Performance vs Attrition Rate scatter plot (bubble size = employees hired)
  - Monthly hiring and attrition trends

### 7. Customer Analytics Dashboard
- **File:** `exports/7_customer_analytics_dashboard.png`
- **Description:** Deep dive into customer metrics including:
  - Customer churn probability distribution with statistical measures
  - Retention rate by department (horizontal bar chart)
  - Churn vs Retention scatter plot (color-coded by revenue)
  - Monthly customer metrics trends (dual-axis)

### 8. Financial Performance Analysis
- **File:** `exports/8_financial_performance_analysis.png`
- **Description:** Advanced financial intelligence dashboard with:
  - Revenue vs Profit scatter by department (bubble size = profit margin)
  - Profit margin distribution box plots by department
  - Cost efficiency analysis (costs vs revenue with break-even line)
  - Quarterly profit trend analysis

### 9. Operational Efficiency Matrix
- **File:** `exports/9_operational_efficiency_matrix.png`
- **Description:** Multi-faceted efficiency analysis including:
  - Efficiency vs Performance scatter (bubble = sales volume, color = cost)
  - Efficiency distribution violin plots by department
  - Monthly efficiency and operational cost trends
  - Department × Quarter efficiency heatmap

### 10. Sales Volume & Product Performance
- **File:** `exports/10_sales_product_performance.png`
- **Description:** Sales and product intelligence dashboard featuring:
  - Sales volume by product category (horizontal bar chart)
  - Product performance: Revenue vs Profit scatter (bubble = sales volume)
  - Monthly sales volume and revenue correlation
  - Market segment performance comparison (normalized metrics)

### 11. Comprehensive Statistical Summary Dashboard
- **File:** `exports/11_comprehensive_statistical_dashboard.png`
- **Description:** Executive summary dashboard with:
  - Key Performance Indicators (KPI) summary cards
  - Revenue distribution by department (pie chart)
  - Top 5 products by revenue
  - Revenue, profit, and efficiency distribution histograms
  - Quarterly performance comparison (multi-metric bar and line chart)

## Analysis Areas

### Financial Metrics
- Revenue analysis and trends
- Cost management and optimization
- Profit margin analysis
- Profitability by department

### Human Resources
- Employee hiring trends
- Attrition analysis
- Performance tracking
- Workforce optimization

### Customer Analytics
- Customer churn prediction
- Retention rate analysis
- Market segment performance
- Customer lifetime value

### Operational Metrics
- Daily efficiency tracking
- Operational cost analysis
- Sales volume analysis
- Product category performance

## Project Structure

```
project_4_enterprise_data_intelligence/
├── README.md
├── dataset_enterprise.csv
├── enterprise_analytics_dashboard.py
└── exports/
    ├── 1_revenue_trend.png
    ├── 2_correlation_heatmap.png
    ├── 3_advanced_time_series_dashboard.png
    ├── 4_department_radar_chart.png
    ├── 5_product_market_heatmap.png
    ├── 6_employee_metrics_analysis.png
    ├── 7_customer_analytics_dashboard.png
    ├── 8_financial_performance_analysis.png
    ├── 9_operational_efficiency_matrix.png
    ├── 10_sales_product_performance.png
    └── 11_comprehensive_statistical_dashboard.png
```

## Usage

### Running the Analysis

1. **Prerequisites:** Install required Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

2. **Generate Visualizations:**
   ```bash
   cd project_4_enterprise_data_intelligence
   python enterprise_analytics_dashboard.py
   ```

3. **Output:** All visualizations will be saved to the `exports/` folder as high-resolution PNG files (300 DPI)

### Customization

The `enterprise_analytics_dashboard.py` script is fully customizable:
- Modify `FIG_SIZE` and `DPI` constants for different output sizes
- Adjust color schemes and styling
- Add or remove visualization sections
- Customize aggregation methods and time periods

## Key Insights

- Department-wise performance comparison
- Revenue growth patterns
- Cost optimization opportunities
- Employee retention strategies
- Customer churn prevention
- Market segment analysis
- Product performance evaluation

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

- Strategic planning and decision-making
- Budget allocation and optimization
- Performance monitoring and KPI tracking
- Risk management and forecasting
- Resource allocation
- Customer relationship management
- Operational efficiency improvement

