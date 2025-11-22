"""
Enterprise Data Intelligence - Advanced Analytics Dashboard
Comprehensive visualization suite with complex, multi-dimensional analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
FIG_SIZE = (16, 10)
DPI = 300
EXPORT_PATH = 'exports/'

# Load data
print("Loading enterprise dataset...")
df = pd.read_csv('dataset_enterprise.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year_month'] = df['date'].dt.to_period('M')

print(f"Dataset loaded: {len(df)} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# 1. MULTI-DIMENSIONAL CORRELATION HEATMAP
# ============================================================================
print("\n1. Generating Multi-dimensional Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(14, 12))

numeric_cols = ['revenue', 'costs', 'profit_margin', 'employees_hired', 
                'employees_attrition', 'employee_performance', 'sales_volume',
                'customer_churn_probability', 'retention_rate', 
                'daily_efficiency', 'operational_cost', 'profit']

correlation_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdYlBu_r', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax)

ax.set_title('Enterprise Metrics Correlation Matrix\nMulti-Dimensional Relationship Analysis', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}2_correlation_heatmap.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 2_correlation_heatmap.png")

# ============================================================================
# 2. ADVANCED TIME SERIES DASHBOARD - MULTI-METRIC
# ============================================================================
print("\n2. Generating Advanced Time Series Dashboard...")
monthly_agg = df.groupby('year_month').agg({
    'revenue': 'sum',
    'costs': 'sum',
    'profit': 'sum',
    'sales_volume': 'sum',
    'employee_performance': 'mean',
    'retention_rate': 'mean',
    'customer_churn_probability': 'mean'
}).reset_index()
monthly_agg['year_month'] = monthly_agg['year_month'].astype(str)

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Revenue & Costs Over Time
ax1 = fig.add_subplot(gs[0, :])
ax1_twin = ax1.twinx()
ax1.plot(monthly_agg['year_month'], monthly_agg['revenue'], 
         marker='o', linewidth=2.5, label='Revenue', color='#2ecc71', markersize=6)
ax1_twin.plot(monthly_agg['year_month'], monthly_agg['costs'], 
              marker='s', linewidth=2.5, label='Costs', color='#e74c3c', markersize=6)
ax1.fill_between(monthly_agg['year_month'], monthly_agg['revenue'], 
                  alpha=0.3, color='#2ecc71')
ax1_twin.fill_between(monthly_agg['year_month'], monthly_agg['costs'], 
                       alpha=0.3, color='#e74c3c')
ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold', color='#2ecc71')
ax1_twin.set_ylabel('Costs ($)', fontsize=12, fontweight='bold', color='#e74c3c')
ax1.set_title('Revenue vs Costs Trend Analysis', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.tick_params(axis='x', rotation=45)

# Profit Trend
ax2 = fig.add_subplot(gs[1, 0])
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in monthly_agg['profit']]
ax2.bar(range(len(monthly_agg)), monthly_agg['profit'], color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
ax2.set_ylabel('Profit ($)', fontsize=11, fontweight='bold')
ax2.set_title('Monthly Profit Analysis', fontsize=13, fontweight='bold')
ax2.set_xticks(range(0, len(monthly_agg), 3))
ax2.set_xticklabels([monthly_agg['year_month'].iloc[i] for i in range(0, len(monthly_agg), 3)], 
                     rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Employee Performance & Retention
ax3 = fig.add_subplot(gs[1, 1])
ax3_twin = ax3.twinx()
ax3.plot(monthly_agg['year_month'], monthly_agg['employee_performance'], 
         marker='o', linewidth=2, label='Performance', color='#3498db', markersize=5)
ax3_twin.bar(range(len(monthly_agg)), monthly_agg['retention_rate'], 
             alpha=0.4, label='Retention Rate', color='#9b59b6', width=0.6)
ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
ax3.set_ylabel('Employee Performance', fontsize=11, fontweight='bold', color='#3498db')
ax3_twin.set_ylabel('Retention Rate (%)', fontsize=11, fontweight='bold', color='#9b59b6')
ax3.set_title('HR Metrics: Performance & Retention', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.set_xticks(range(0, len(monthly_agg), 3))
ax3.set_xticklabels([monthly_agg['year_month'].iloc[i] for i in range(0, len(monthly_agg), 3)], 
                     rotation=45, ha='right')

# Sales Volume & Churn
ax4 = fig.add_subplot(gs[2, 0])
ax4_twin = ax4.twinx()
ax4.bar(range(len(monthly_agg)), monthly_agg['sales_volume'], 
        alpha=0.6, label='Sales Volume', color='#f39c12', edgecolor='black')
ax4_twin.plot(monthly_agg['year_month'], monthly_agg['customer_churn_probability']*100, 
              marker='D', linewidth=2.5, label='Churn Probability', color='#e67e22', markersize=5)
ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
ax4.set_ylabel('Sales Volume', fontsize=11, fontweight='bold', color='#f39c12')
ax4_twin.set_ylabel('Churn Probability (%)', fontsize=11, fontweight='bold', color='#e67e22')
ax4.set_title('Sales Volume vs Customer Churn', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.set_xticks(range(0, len(monthly_agg), 3))
ax4.set_xticklabels([monthly_agg['year_month'].iloc[i] for i in range(0, len(monthly_agg), 3)], 
                     rotation=45, ha='right')

# Efficiency Trend
ax5 = fig.add_subplot(gs[2, 1])
efficiency_data = df.groupby('year_month')['daily_efficiency'].mean().reset_index()
efficiency_data['year_month'] = efficiency_data['year_month'].astype(str)
ax5.plot(efficiency_data['year_month'], efficiency_data['daily_efficiency'], 
         marker='o', linewidth=2.5, markersize=6, color='#1abc9c')
ax5.fill_between(efficiency_data['year_month'], efficiency_data['daily_efficiency'], 
                 alpha=0.3, color='#1abc9c')
ax5.axhline(y=efficiency_data['daily_efficiency'].mean(), color='red', 
            linestyle='--', linewidth=2, label=f'Mean: {efficiency_data["daily_efficiency"].mean():.1f}')
ax5.set_xlabel('Month', fontsize=11, fontweight='bold')
ax5.set_ylabel('Daily Efficiency', fontsize=11, fontweight='bold')
ax5.set_title('Operational Efficiency Trend', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.tick_params(axis='x', rotation=45)

plt.suptitle('Enterprise Intelligence Dashboard - Multi-Metric Time Series Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig(f'{EXPORT_PATH}3_advanced_time_series_dashboard.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 3_advanced_time_series_dashboard.png")

# ============================================================================
# 3. DEPARTMENT PERFORMANCE COMPARISON - RADAR CHART
# ============================================================================
print("\n3. Generating Department Performance Radar Chart...")
dept_metrics = df.groupby('department').agg({
    'revenue': 'mean',
    'profit_margin': 'mean',
    'employee_performance': 'mean',
    'retention_rate': 'mean',
    'daily_efficiency': 'mean',
    'sales_volume': 'mean'
}).reset_index()

# Normalize metrics to 0-100 scale for radar chart
metrics_to_plot = ['revenue', 'profit_margin', 'employee_performance', 
                   'retention_rate', 'daily_efficiency', 'sales_volume']
dept_normalized = dept_metrics.copy()
for metric in metrics_to_plot:
    dept_normalized[metric] = ((dept_metrics[metric] - dept_metrics[metric].min()) / 
                               (dept_metrics[metric].max() - dept_metrics[metric].min())) * 100

# Create radar chart
fig, ax = plt.subplots(figsize=(16, 12), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

colors = plt.cm.Set3(np.linspace(0, 1, len(dept_metrics)))

for idx, dept in enumerate(dept_metrics['department']):
    values = dept_normalized[dept_normalized['department'] == dept][metrics_to_plot].values[0].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2.5, label=dept, color=colors[idx], markersize=8)
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Revenue', 'Profit Margin', 'Employee\nPerformance', 
                    'Retention Rate', 'Daily\nEfficiency', 'Sales Volume'], 
                   fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Department Performance Comparison\nMulti-Dimensional Analysis (Normalized)', 
             fontsize=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.savefig(f'{EXPORT_PATH}4_department_radar_chart.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 4_department_radar_chart.png")

# ============================================================================
# 4. PRODUCT CATEGORY & MARKET SEGMENT HEATMAP
# ============================================================================
print("\n4. Generating Product-Market Segment Heatmap...")
product_market = df.groupby(['product_category', 'market_segment']).agg({
    'revenue': 'sum',
    'profit': 'sum',
    'sales_volume': 'sum',
    'profit_margin': 'mean'
}).reset_index()

pivot_revenue = product_market.pivot(index='product_category', 
                                     columns='market_segment', 
                                     values='revenue')
pivot_profit = product_market.pivot(index='product_category', 
                                    columns='market_segment', 
                                    values='profit')

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(pivot_revenue, annot=True, fmt='.0f', cmap='YlOrRd', 
            linewidths=1, cbar_kws={'label': 'Revenue ($)'}, 
            ax=axes[0], square=True)
axes[0].set_title('Revenue by Product Category & Market Segment', 
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Market Segment', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Product Category', fontsize=12, fontweight='bold')

sns.heatmap(pivot_profit, annot=True, fmt='.0f', cmap='RdYlGn', 
            linewidths=1, cbar_kws={'label': 'Profit ($)'}, 
            ax=axes[1], square=True, center=0)
axes[1].set_title('Profit by Product Category & Market Segment', 
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Market Segment', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Product Category', fontsize=12, fontweight='bold')

plt.suptitle('Product-Market Segment Performance Matrix', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}5_product_market_heatmap.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 5_product_market_heatmap.png")

# ============================================================================
# 5. EMPLOYEE METRICS - HIRING VS ATTRITION ANALYSIS
# ============================================================================
print("\n5. Generating Employee Metrics Analysis...")
dept_hr = df.groupby('department').agg({
    'employees_hired': 'sum',
    'employees_attrition': 'sum',
    'employee_performance': 'mean'
}).reset_index()
dept_hr['net_employee_change'] = dept_hr['employees_hired'] - dept_hr['employees_attrition']

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Hiring vs Attrition by Department
x_pos = np.arange(len(dept_hr))
width = 0.35
axes[0, 0].bar(x_pos - width/2, dept_hr['employees_hired'], width, 
               label='Hired', color='#2ecc71', alpha=0.8, edgecolor='black')
axes[0, 0].bar(x_pos + width/2, dept_hr['employees_attrition'], width, 
               label='Attrition', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('Department', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Number of Employees', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Employee Hiring vs Attrition by Department', 
                     fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(dept_hr['department'], rotation=45, ha='right')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Net Employee Change
colors_net = ['#e74c3c' if x < 0 else '#2ecc71' for x in dept_hr['net_employee_change']]
axes[0, 1].barh(dept_hr['department'], dept_hr['net_employee_change'], 
                color=colors_net, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1.5)
axes[0, 1].set_xlabel('Net Employee Change', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Net Employee Change by Department', 
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Performance vs Attrition Rate
dept_hr['attrition_rate'] = (dept_hr['employees_attrition'] / 
                             (dept_hr['employees_hired'] + dept_hr['employees_attrition'] + 1)) * 100
scatter = axes[1, 0].scatter(dept_hr['employee_performance'], dept_hr['attrition_rate'],
                            s=dept_hr['employees_hired']*50, alpha=0.6, 
                            c=range(len(dept_hr)), cmap='viridis', edgecolors='black', linewidth=2)
for i, dept in enumerate(dept_hr['department']):
    axes[1, 0].annotate(dept, (dept_hr['employee_performance'].iloc[i], 
                               dept_hr['attrition_rate'].iloc[i]),
                       fontsize=9, ha='center', va='center', fontweight='bold')
axes[1, 0].set_xlabel('Employee Performance', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Attrition Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Performance vs Attrition Rate\n(Bubble size = Employees Hired)', 
                     fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Monthly Hiring Trend
monthly_hr = df.groupby('year_month').agg({
    'employees_hired': 'sum',
    'employees_attrition': 'sum'
}).reset_index()
monthly_hr['year_month'] = monthly_hr['year_month'].astype(str)
x_month = range(len(monthly_hr))
axes[1, 1].plot(x_month, monthly_hr['employees_hired'], 
                marker='o', linewidth=2.5, label='Hired', color='#2ecc71', markersize=6)
axes[1, 1].plot(x_month, monthly_hr['employees_attrition'], 
                marker='s', linewidth=2.5, label='Attrition', color='#e74c3c', markersize=6)
axes[1, 1].fill_between(x_month, monthly_hr['employees_hired'], alpha=0.3, color='#2ecc71')
axes[1, 1].fill_between(x_month, monthly_hr['employees_attrition'], alpha=0.3, color='#e74c3c')
axes[1, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Number of Employees', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Monthly Hiring & Attrition Trends', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(range(0, len(monthly_hr), 3))
axes[1, 1].set_xticklabels([monthly_hr['year_month'].iloc[i] for i in range(0, len(monthly_hr), 3)], 
                           rotation=45, ha='right')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Human Resources Analytics Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}6_employee_metrics_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 6_employee_metrics_analysis.png")

# ============================================================================
# 6. CUSTOMER ANALYTICS - CHURN & RETENTION DEEP DIVE
# ============================================================================
print("\n6. Generating Customer Analytics Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Churn Probability Distribution
axes[0, 0].hist(df['customer_churn_probability'], bins=30, color='#e74c3c', 
                alpha=0.7, edgecolor='black')
axes[0, 0].axvline(df['customer_churn_probability'].mean(), color='blue', 
                   linestyle='--', linewidth=2.5, label=f'Mean: {df["customer_churn_probability"].mean():.3f}')
axes[0, 0].axvline(df['customer_churn_probability'].median(), color='green', 
                   linestyle='--', linewidth=2.5, label=f'Median: {df["customer_churn_probability"].median():.3f}')
axes[0, 0].set_xlabel('Churn Probability', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Customer Churn Probability Distribution', 
                     fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Retention Rate by Department
dept_customer = df.groupby('department').agg({
    'retention_rate': 'mean',
    'customer_churn_probability': 'mean'
}).reset_index().sort_values('retention_rate', ascending=True)

y_pos = np.arange(len(dept_customer))
axes[0, 1].barh(y_pos, dept_customer['retention_rate'], 
                color=plt.cm.RdYlGn(dept_customer['retention_rate']/100), 
                alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(y_pos)
axes[0, 1].set_yticklabels(dept_customer['department'], fontsize=10)
axes[0, 1].set_xlabel('Retention Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Average Retention Rate by Department', 
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(dept_customer['retention_rate']):
    axes[0, 1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')

# Churn vs Retention Scatter
axes[1, 0].scatter(df['customer_churn_probability']*100, df['retention_rate'],
                   alpha=0.5, s=50, c=df['revenue'], cmap='viridis', 
                   edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
cbar.set_label('Revenue ($)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Churn Probability (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Retention Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Churn Probability vs Retention Rate\n(Color = Revenue)', 
                     fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Monthly Churn & Retention Trends
monthly_customer = df.groupby('year_month').agg({
    'customer_churn_probability': 'mean',
    'retention_rate': 'mean'
}).reset_index()
monthly_customer['year_month'] = monthly_customer['year_month'].astype(str)
x_month = range(len(monthly_customer))
ax_twin = axes[1, 1].twinx()
line1 = axes[1, 1].plot(x_month, monthly_customer['retention_rate'], 
                        marker='o', linewidth=2.5, label='Retention Rate', 
                        color='#2ecc71', markersize=6)
line2 = ax_twin.plot(x_month, monthly_customer['customer_churn_probability']*100, 
                     marker='s', linewidth=2.5, label='Churn Probability', 
                     color='#e74c3c', markersize=6)
axes[1, 1].fill_between(x_month, monthly_customer['retention_rate'], 
                        alpha=0.3, color='#2ecc71')
ax_twin.fill_between(x_month, monthly_customer['customer_churn_probability']*100, 
                     alpha=0.3, color='#e74c3c')
axes[1, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Retention Rate (%)', fontsize=12, fontweight='bold', color='#2ecc71')
ax_twin.set_ylabel('Churn Probability (%)', fontsize=12, fontweight='bold', color='#e74c3c')
axes[1, 1].set_title('Monthly Customer Metrics Trends', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(range(0, len(monthly_customer), 3))
axes[1, 1].set_xticklabels([monthly_customer['year_month'].iloc[i] for i in range(0, len(monthly_customer), 3)], 
                           rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
axes[1, 1].legend(lines, labels, loc='upper left', fontsize=10)

plt.suptitle('Customer Analytics & Retention Intelligence', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}7_customer_analytics_dashboard.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 7_customer_analytics_dashboard.png")

# ============================================================================
# 7. FINANCIAL PERFORMANCE - PROFITABILITY ANALYSIS
# ============================================================================
print("\n7. Generating Financial Performance Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Revenue vs Profit by Department
dept_financial = df.groupby('department').agg({
    'revenue': 'sum',
    'profit': 'sum',
    'profit_margin': 'mean',
    'costs': 'sum'
}).reset_index()

scatter = axes[0, 0].scatter(dept_financial['revenue'], dept_financial['profit'],
                            s=dept_financial['profit_margin']*20, 
                            c=dept_financial['profit_margin'], cmap='RdYlGn',
                            alpha=0.7, edgecolors='black', linewidth=2, vmin=0, vmax=50)
for i, dept in enumerate(dept_financial['department']):
    axes[0, 0].annotate(dept, (dept_financial['revenue'].iloc[i], 
                               dept_financial['profit'].iloc[i]),
                       fontsize=9, ha='center', va='center', fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[0, 0])
cbar.set_label('Profit Margin (%)', fontsize=11, fontweight='bold')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].set_xlabel('Total Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Total Profit ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Revenue vs Profit by Department\n(Bubble size = Profit Margin)', 
                     fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Profit Margin Distribution
axes[0, 1].boxplot([df[df['department'] == dept]['profit_margin'].values 
                    for dept in dept_financial['department']],
                   labels=dept_financial['department'], vert=True)
axes[0, 1].set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Profit Margin Distribution by Department', 
                     fontsize=13, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Costs vs Revenue Efficiency
axes[1, 0].scatter(dept_financial['costs'], dept_financial['revenue'],
                   s=300, alpha=0.6, c=range(len(dept_financial)), 
                   cmap='coolwarm', edgecolors='black', linewidth=2)
for i, dept in enumerate(dept_financial['department']):
    axes[1, 0].annotate(dept, (dept_financial['costs'].iloc[i], 
                               dept_financial['revenue'].iloc[i]),
                       fontsize=9, ha='center', va='center', fontweight='bold')
# Add diagonal line (1:1 ratio)
max_val = max(dept_financial['costs'].max(), dept_financial['revenue'].max())
axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, 
                label='Break-even Line', alpha=0.7)
axes[1, 0].set_xlabel('Total Costs ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Cost Efficiency Analysis', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Quarterly Profit Trend
quarterly = df.groupby(['year', 'quarter']).agg({
    'profit': 'sum',
    'revenue': 'sum'
}).reset_index()
quarterly['period'] = quarterly['year'].astype(str) + '-Q' + quarterly['quarter'].astype(str)
x_q = range(len(quarterly))
axes[1, 1].bar(x_q, quarterly['profit'], color=plt.cm.RdYlGn(0.3 + quarterly['profit']/quarterly['profit'].max()*0.4),
               alpha=0.8, edgecolor='black')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 1].set_xlabel('Quarter', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Profit ($)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Quarterly Profit Trend', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(x_q[::2])
axes[1, 1].set_xticklabels(quarterly['period'].iloc[::2], rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(quarterly['profit']):
    if abs(v) > 1000:
        axes[1, 1].text(i, v, f'${v/1000:.1f}K', ha='center', 
                       va='bottom' if v > 0 else 'top', fontsize=8, fontweight='bold')

plt.suptitle('Financial Performance & Profitability Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}8_financial_performance_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 8_financial_performance_analysis.png")

# ============================================================================
# 8. OPERATIONAL EFFICIENCY MATRIX
# ============================================================================
print("\n8. Generating Operational Efficiency Matrix...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Efficiency vs Performance by Department
dept_ops = df.groupby('department').agg({
    'daily_efficiency': 'mean',
    'employee_performance': 'mean',
    'operational_cost': 'mean',
    'sales_volume': 'mean'
}).reset_index()

axes[0, 0].scatter(dept_ops['daily_efficiency'], dept_ops['employee_performance'],
                   s=dept_ops['sales_volume']/10, alpha=0.7,
                   c=dept_ops['operational_cost'], cmap='plasma',
                   edgecolors='black', linewidth=2)
cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
cbar.set_label('Operational Cost ($)', fontsize=11, fontweight='bold')
for i, dept in enumerate(dept_ops['department']):
    axes[0, 0].annotate(dept, (dept_ops['daily_efficiency'].iloc[i], 
                               dept_ops['employee_performance'].iloc[i]),
                       fontsize=9, ha='center', va='center', fontweight='bold')
axes[0, 0].set_xlabel('Daily Efficiency', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Employee Performance', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Efficiency vs Performance\n(Bubble = Sales Volume, Color = Cost)', 
                     fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Efficiency Distribution
axes[0, 1].violinplot([df[df['department'] == dept]['daily_efficiency'].values 
                      for dept in dept_ops['department']],
                     positions=range(len(dept_ops)), showmeans=True)
axes[0, 1].set_xticks(range(len(dept_ops)))
axes[0, 1].set_xticklabels(dept_ops['department'], rotation=45, ha='right')
axes[0, 1].set_ylabel('Daily Efficiency', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Efficiency Distribution by Department', 
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Monthly Efficiency Trend
monthly_eff = df.groupby('year_month').agg({
    'daily_efficiency': 'mean',
    'operational_cost': 'mean'
}).reset_index()
monthly_eff['year_month'] = monthly_eff['year_month'].astype(str)
x_month = range(len(monthly_eff))
ax_twin = axes[1, 0].twinx()
line1 = axes[1, 0].plot(x_month, monthly_eff['daily_efficiency'], 
                        marker='o', linewidth=2.5, label='Efficiency', 
                        color='#3498db', markersize=6)
line2 = ax_twin.plot(x_month, monthly_eff['operational_cost'], 
                     marker='s', linewidth=2.5, label='Operational Cost', 
                     color='#e67e22', markersize=6)
axes[1, 0].fill_between(x_month, monthly_eff['daily_efficiency'], 
                        alpha=0.3, color='#3498db')
axes[1, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Daily Efficiency', fontsize=12, fontweight='bold', color='#3498db')
ax_twin.set_ylabel('Operational Cost ($)', fontsize=12, fontweight='bold', color='#e67e22')
axes[1, 0].set_title('Monthly Efficiency & Cost Trends', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(range(0, len(monthly_eff), 3))
axes[1, 0].set_xticklabels([monthly_eff['year_month'].iloc[i] for i in range(0, len(monthly_eff), 3)], 
                           rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
axes[1, 0].legend(lines, labels, loc='upper left', fontsize=10)

# Efficiency Heatmap by Department and Quarter
dept_quarter_eff = df.groupby(['department', 'quarter']).agg({
    'daily_efficiency': 'mean'
}).reset_index()
pivot_eff = dept_quarter_eff.pivot(index='department', columns='quarter', values='daily_efficiency')
sns.heatmap(pivot_eff, annot=True, fmt='.1f', cmap='YlGnBu', 
            linewidths=1, cbar_kws={'label': 'Efficiency'}, 
            ax=axes[1, 1], square=False)
axes[1, 1].set_xlabel('Quarter', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Department', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Efficiency Heatmap: Department × Quarter', 
                     fontsize=13, fontweight='bold')

plt.suptitle('Operational Efficiency Intelligence Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}9_operational_efficiency_matrix.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 9_operational_efficiency_matrix.png")

# ============================================================================
# 9. SALES VOLUME & PRODUCT PERFORMANCE
# ============================================================================
print("\n9. Generating Sales Volume & Product Performance Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Sales Volume by Product Category
product_sales = df.groupby('product_category').agg({
    'sales_volume': 'sum',
    'revenue': 'sum',
    'profit': 'sum'
}).reset_index().sort_values('sales_volume', ascending=False)

axes[0, 0].barh(product_sales['product_category'], product_sales['sales_volume'],
                color=plt.cm.viridis(np.linspace(0, 1, len(product_sales))),
                alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('Total Sales Volume', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Sales Volume by Product Category', 
                     fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(product_sales['sales_volume']):
    axes[0, 0].text(v + 100, i, f'{v:,}', va='center', fontweight='bold')

# Product Performance: Revenue vs Profit
axes[0, 1].scatter(product_sales['revenue'], product_sales['profit'],
                   s=product_sales['sales_volume']/5, alpha=0.7,
                   c=range(len(product_sales)), cmap='coolwarm',
                   edgecolors='black', linewidth=2)
for i, prod in enumerate(product_sales['product_category']):
    axes[0, 1].annotate(prod, (product_sales['revenue'].iloc[i], 
                              product_sales['profit'].iloc[i]),
                       fontsize=9, ha='center', va='center', fontweight='bold')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0, 1].set_xlabel('Total Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Total Profit ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Product Performance: Revenue vs Profit\n(Bubble size = Sales Volume)', 
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Monthly Sales Volume Trend
monthly_sales = df.groupby('year_month').agg({
    'sales_volume': 'sum',
    'revenue': 'sum'
}).reset_index()
monthly_sales['year_month'] = monthly_sales['year_month'].astype(str)
x_month = range(len(monthly_sales))
ax_twin = axes[1, 0].twinx()
line1 = axes[1, 0].bar(x_month, monthly_sales['sales_volume'], 
                       alpha=0.6, color='#3498db', label='Sales Volume', width=0.6)
line2 = ax_twin.plot(x_month, monthly_sales['revenue'], 
                     marker='o', linewidth=2.5, label='Revenue', 
                     color='#e74c3c', markersize=6)
axes[1, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Sales Volume', fontsize=12, fontweight='bold', color='#3498db')
ax_twin.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold', color='#e74c3c')
axes[1, 0].set_title('Monthly Sales Volume & Revenue Correlation', 
                     fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(range(0, len(monthly_sales), 3))
axes[1, 0].set_xticklabels([monthly_sales['year_month'].iloc[i] for i in range(0, len(monthly_sales), 3)], 
                           rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3, axis='y')
lines = [line1] + line2
labels = ['Sales Volume', 'Revenue']
axes[1, 0].legend(lines, labels, loc='upper left', fontsize=10)

# Market Segment Performance
market_perf = df.groupby('market_segment').agg({
    'sales_volume': 'sum',
    'revenue': 'sum',
    'profit': 'sum',
    'customer_churn_probability': 'mean'
}).reset_index()

x_pos = np.arange(len(market_perf))
width = 0.25
axes[1, 1].bar(x_pos - width, market_perf['sales_volume']/1000, width, 
               label='Sales Volume (K)', color='#3498db', alpha=0.8, edgecolor='black')
axes[1, 1].bar(x_pos, market_perf['revenue']/1000000, width, 
               label='Revenue (M$)', color='#2ecc71', alpha=0.8, edgecolor='black')
axes[1, 1].bar(x_pos + width, market_perf['profit']/1000000, width, 
               label='Profit (M$)', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Market Segment', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Normalized Values', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Market Segment Performance Comparison', 
                     fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(market_perf['market_segment'])
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Sales Volume & Product Performance Intelligence', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}10_sales_product_performance.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 10_sales_product_performance.png")

# ============================================================================
# 10. COMPREHENSIVE STATISTICAL SUMMARY DASHBOARD
# ============================================================================
print("\n10. Generating Comprehensive Statistical Summary Dashboard...")
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# Key Metrics Summary
ax1 = fig.add_subplot(gs[0, 0])
metrics_summary = {
    'Total Revenue': f"${df['revenue'].sum()/1e6:.2f}M",
    'Total Profit': f"${df['profit'].sum()/1e6:.2f}M",
    'Avg Profit Margin': f"{df['profit_margin'].mean():.1f}%",
    'Total Sales Volume': f"{df['sales_volume'].sum():,}",
    'Avg Retention': f"{df['retention_rate'].mean():.1f}%",
    'Avg Efficiency': f"{df['daily_efficiency'].mean():.1f}"
}
y_pos = np.arange(len(metrics_summary))
ax1.barh(y_pos, [1]*len(metrics_summary), color=plt.cm.viridis(np.linspace(0, 1, len(metrics_summary))))
ax1.set_yticks(y_pos)
ax1.set_yticklabels(list(metrics_summary.keys()), fontsize=11, fontweight='bold')
ax1.set_xticks([])
ax1.set_title('Key Performance Indicators', fontsize=13, fontweight='bold')
for i, (key, val) in enumerate(metrics_summary.items()):
    ax1.text(0.5, i, val, ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Department Revenue Share
ax2 = fig.add_subplot(gs[0, 1])
dept_rev = df.groupby('department')['revenue'].sum().sort_values(ascending=False)
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(dept_rev)))
wedges, texts, autotexts = ax2.pie(dept_rev, labels=dept_rev.index, autopct='%1.1f%%',
                                    startangle=90, colors=colors_pie, textprops={'fontsize': 9, 'fontweight': 'bold'})
ax2.set_title('Revenue Distribution by Department', fontsize=13, fontweight='bold')

# Top Products by Revenue
ax3 = fig.add_subplot(gs[0, 2])
top_products = df.groupby('product_category')['revenue'].sum().sort_values(ascending=True).tail(5)
ax3.barh(range(len(top_products)), top_products.values,
         color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_products))),
         alpha=0.8, edgecolor='black')
ax3.set_yticks(range(len(top_products)))
ax3.set_yticklabels(top_products.index, fontsize=10, fontweight='bold')
ax3.set_xlabel('Revenue ($)', fontsize=11, fontweight='bold')
ax3.set_title('Top 5 Products by Revenue', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_products.values):
    ax3.text(v, i, f'${v/1000:.0f}K', va='center', fontweight='bold')

# Revenue Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df['revenue'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax4.axvline(df['revenue'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: ${df["revenue"].mean():,.0f}')
ax4.axvline(df['revenue'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: ${df["revenue"].median():,.0f}')
ax4.set_xlabel('Revenue ($)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Revenue Distribution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Profit Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df['profit'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
ax5.axvline(df['profit'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: ${df["profit"].mean():,.0f}')
ax5.axvline(0, color='black', linestyle='-', linewidth=1.5)
ax5.set_xlabel('Profit ($)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Profit Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Efficiency Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(df['daily_efficiency'], bins=40, color='#9b59b6', alpha=0.7, edgecolor='black')
ax6.axvline(df['daily_efficiency'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["daily_efficiency"].mean():.1f}')
ax6.set_xlabel('Daily Efficiency', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Efficiency Distribution', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# Quarterly Performance Comparison
ax7 = fig.add_subplot(gs[2, :])
quarterly_comp = df.groupby(['year', 'quarter']).agg({
    'revenue': 'sum',
    'profit': 'sum',
    'sales_volume': 'sum'
}).reset_index()
quarterly_comp['period'] = quarterly_comp['year'].astype(str) + '-Q' + quarterly_comp['quarter'].astype(str)
x_q = range(len(quarterly_comp))
width = 0.25
ax7.bar([x - width for x in x_q], quarterly_comp['revenue']/1000, width,
        label='Revenue (K$)', color='#3498db', alpha=0.8, edgecolor='black')
ax7.bar(x_q, quarterly_comp['profit']/1000, width,
        label='Profit (K$)', color='#2ecc71', alpha=0.8, edgecolor='black')
ax7_twin = ax7.twinx()
ax7_twin.plot(x_q, quarterly_comp['sales_volume']/100, 
              marker='o', linewidth=2.5, label='Sales Volume (×100)', 
              color='#e74c3c', markersize=8)
ax7.set_xlabel('Quarter', fontsize=12, fontweight='bold')
ax7.set_ylabel('Financial Metrics (K$)', fontsize=12, fontweight='bold')
ax7_twin.set_ylabel('Sales Volume (×100)', fontsize=12, fontweight='bold', color='#e74c3c')
ax7.set_title('Quarterly Performance Comparison', fontsize=14, fontweight='bold')
ax7.set_xticks(x_q[::2])
ax7.set_xticklabels(quarterly_comp['period'].iloc[::2], rotation=45, ha='right')
ax7.legend(loc='upper left', fontsize=10)
ax7_twin.legend(loc='upper right', fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle('Enterprise Intelligence - Comprehensive Statistical Summary Dashboard', 
             fontsize=20, fontweight='bold', y=0.995)
plt.savefig(f'{EXPORT_PATH}11_comprehensive_statistical_dashboard.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 11_comprehensive_statistical_dashboard.png")

print("\n" + "="*60)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nTotal visualizations created: 11")
print(f"Export directory: {EXPORT_PATH}")
print("\nGenerated files:")
print("  1. 2_correlation_heatmap.png")
print("  2. 3_advanced_time_series_dashboard.png")
print("  3. 4_department_radar_chart.png")
print("  4. 5_product_market_heatmap.png")
print("  5. 6_employee_metrics_analysis.png")
print("  6. 7_customer_analytics_dashboard.png")
print("  7. 8_financial_performance_analysis.png")
print("  8. 9_operational_efficiency_matrix.png")
print("  9. 10_sales_product_performance.png")
print(" 10. 11_comprehensive_statistical_dashboard.png")
print("\nAll visualizations are high-resolution (300 DPI) and ready for presentation!")

