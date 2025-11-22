"""
Sales Analytics Dashboard - Advanced Analytics
Comprehensive visualization suite with advanced sales performance analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
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
print("Loading sales dataset...")
df = pd.read_csv('dataset_sales.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year_month'] = df['date'].dt.to_period('M')
df['day_of_week'] = df['date'].dt.day_name()
df['week'] = df['date'].dt.isocalendar().week
df['revenue_per_order'] = df['revenue'] / df['orders']

print(f"Dataset loaded: {len(df)} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Categories: {df['category'].nunique()}")
print(f"Regions: {df['region'].nunique()}")
print(f"Total Revenue: ${df['revenue'].sum():,.2f}")
print(f"Total Orders: {df['orders'].sum():,}")

# ============================================================================
# 1. COMPREHENSIVE REVENUE TRENDS DASHBOARD
# ============================================================================
print("\n1. Generating Comprehensive Revenue Trends Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Monthly revenue trends
monthly_revenue = df.groupby('year_month')['revenue'].sum().reset_index()
monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)
monthly_orders = df.groupby('year_month')['orders'].sum().reset_index()
monthly_orders['year_month'] = monthly_orders['year_month'].astype(str)

ax_twin = axes[0, 0].twinx()
line1 = axes[0, 0].plot(range(len(monthly_revenue)), monthly_revenue['revenue'],
                       marker='o', linewidth=2.5, label='Revenue', 
                       color='#2ecc71', markersize=8)
line2 = ax_twin.plot(range(len(monthly_orders)), monthly_orders['orders'],
                    marker='s', linewidth=2.5, label='Orders', 
                    color='#3498db', markersize=8)
axes[0, 0].fill_between(range(len(monthly_revenue)), monthly_revenue['revenue'],
                       alpha=0.3, color='#2ecc71')
axes[0, 0].axhline(y=monthly_revenue['revenue'].mean(), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Mean: ${monthly_revenue["revenue"].mean():,.0f}')
axes[0, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Revenue ($)', fontsize=12, fontweight='bold', color='#2ecc71')
ax_twin.set_ylabel('Number of Orders', fontsize=12, fontweight='bold', color='#3498db')
axes[0, 0].set_title('Monthly Revenue & Orders Trends', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(range(0, len(monthly_revenue), 2))
axes[0, 0].set_xticklabels([monthly_revenue['year_month'].iloc[i] 
                            for i in range(0, len(monthly_revenue), 2)], 
                           rotation=45, ha='right')
axes[0, 0].grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
axes[0, 0].legend(lines, labels, loc='upper left', fontsize=10)

# Revenue by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_revenue = df.groupby('day_of_week')['revenue'].sum().reindex(day_order, fill_value=0)
colors_day = plt.cm.Set3(np.linspace(0, 1, len(day_revenue)))
axes[0, 1].bar(range(len(day_revenue)), day_revenue.values,
              color=colors_day, alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Day of Week', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Revenue by Day of Week', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(range(len(day_revenue)))
axes[0, 1].set_xticklabels(day_revenue.index, rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(day_revenue.values):
    axes[0, 1].text(i, v, f'${v/1000:.1f}K', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9)

# Quarterly revenue comparison
quarterly_revenue = df.groupby('quarter')['revenue'].sum()
axes[1, 0].bar(quarterly_revenue.index, quarterly_revenue.values,
              color=plt.cm.viridis(np.linspace(0, 1, len(quarterly_revenue))),
              alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Quarter', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Revenue by Quarter', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(quarterly_revenue.index)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(quarterly_revenue.values):
    axes[1, 0].text(quarterly_revenue.index[i], v, f'${v/1000:.0f}K', 
                   ha='center', va='bottom', fontweight='bold')

# Revenue growth rate
monthly_revenue['growth_rate'] = monthly_revenue['revenue'].pct_change() * 100
axes[1, 1].bar(range(len(monthly_revenue)), monthly_revenue['growth_rate'],
              color=['#2ecc71' if x > 0 else '#e74c3c' for x in monthly_revenue['growth_rate']],
              alpha=0.8, edgecolor='black')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
axes[1, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Monthly Revenue Growth Rate', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(range(0, len(monthly_revenue), 2))
axes[1, 1].set_xticklabels([monthly_revenue['year_month'].iloc[i] 
                            for i in range(0, len(monthly_revenue), 2)], 
                           rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(monthly_revenue['growth_rate']):
    if not pd.isna(v) and abs(v) > 1:
        axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', 
                       va='bottom' if v > 0 else 'top', fontweight='bold', fontsize=8)

plt.suptitle('Sales Revenue Trends Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}1_comprehensive_revenue_trends.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 1_comprehensive_revenue_trends.png")

# ============================================================================
# 2. CATEGORY & REGION PERFORMANCE ANALYSIS
# ============================================================================
print("\n2. Generating Category & Region Performance Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Revenue by category
category_revenue = df.groupby('category')['revenue'].sum().sort_values(ascending=True)
axes[0, 0].barh(range(len(category_revenue)), category_revenue.values,
              color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(category_revenue))),
              alpha=0.8, edgecolor='black')
axes[0, 0].set_yticks(range(len(category_revenue)))
axes[0, 0].set_yticklabels(category_revenue.index, fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Total Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Revenue by Product Category', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(category_revenue.values):
    axes[0, 0].text(v, i, f'${v/1000:.0f}K', va='center', fontweight='bold')

# Revenue by region
region_revenue = df.groupby('region')['revenue'].sum().sort_values(ascending=True)
axes[0, 1].barh(range(len(region_revenue)), region_revenue.values,
               color=plt.cm.coolwarm(np.linspace(0, 1, len(region_revenue))),
               alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(region_revenue)))
axes[0, 1].set_yticklabels(region_revenue.index, fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Total Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Revenue by Region', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(region_revenue.values):
    axes[0, 1].text(v, i, f'${v/1000:.0f}K', va='center', fontweight='bold')

# Category × Region heatmap
category_region = df.groupby(['category', 'region'])['revenue'].sum().unstack(fill_value=0)
sns.heatmap(category_region, annot=True, fmt='.0f', cmap='YlOrRd', 
           linewidths=1, cbar_kws={'label': 'Revenue ($)'}, 
           ax=axes[1, 0], square=False)
axes[1, 0].set_xlabel('Region', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Category', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Revenue Heatmap: Category × Region', fontsize=14, fontweight='bold')

# Orders by category
category_orders = df.groupby('category')['orders'].sum().sort_values(ascending=True)
x_pos = np.arange(len(category_orders))
width = 0.6
axes[1, 1].barh(x_pos, category_orders.values, width,
               color=plt.cm.viridis(np.linspace(0, 1, len(category_orders))),
               alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(x_pos)
axes[1, 1].set_yticklabels(category_orders.index, fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Total Orders', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Orders by Product Category', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(category_orders.values):
    axes[1, 1].text(v, i, f'{int(v)}', va='center', fontweight='bold')

plt.suptitle('Category & Region Performance Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}2_category_region_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 2_category_region_analysis.png")

# ============================================================================
# 3. REVENUE VS ORDERS ANALYSIS
# ============================================================================
print("\n3. Generating Revenue vs Orders Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Revenue vs Orders scatter by category
for category in df['category'].unique():
    cat_data = df[df['category'] == category]
    axes[0, 0].scatter(cat_data['orders'], cat_data['revenue'],
                      s=100, alpha=0.6, label=category, edgecolors='black', linewidth=1)
axes[0, 0].set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Revenue vs Orders by Category', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='best', fontsize=9, ncol=2)
axes[0, 0].grid(True, alpha=0.3)

# Revenue per order by category
category_avg_revenue = df.groupby('category')['revenue_per_order'].mean().sort_values(ascending=True)
axes[0, 1].barh(range(len(category_avg_revenue)), category_avg_revenue.values,
               color=plt.cm.plasma(np.linspace(0, 1, len(category_avg_revenue))),
               alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(category_avg_revenue)))
axes[0, 1].set_yticklabels(category_avg_revenue.index, fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Average Revenue per Order ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Average Revenue per Order by Category', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(category_avg_revenue.values):
    axes[0, 1].text(v, i, f'${v:.0f}', va='center', fontweight='bold')

# Revenue vs Orders scatter by region
for region in df['region'].unique():
    region_data = df[df['region'] == region]
    axes[1, 0].scatter(region_data['orders'], region_data['revenue'],
                      s=100, alpha=0.6, label=region, edgecolors='black', linewidth=1)
axes[1, 0].set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Revenue vs Orders by Region', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc='best', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Revenue per order by region
region_avg_revenue = df.groupby('region')['revenue_per_order'].mean().sort_values(ascending=True)
axes[1, 1].barh(range(len(region_avg_revenue)), region_avg_revenue.values,
              color=plt.cm.coolwarm(np.linspace(0, 1, len(region_avg_revenue))),
              alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(region_avg_revenue)))
axes[1, 1].set_yticklabels(region_avg_revenue.index, fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Average Revenue per Order ($)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Average Revenue per Order by Region', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(region_avg_revenue.values):
    axes[1, 1].text(v, i, f'${v:.0f}', va='center', fontweight='bold')

plt.suptitle('Revenue vs Orders Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}3_revenue_orders_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 3_revenue_orders_analysis.png")

# ============================================================================
# 4. TEMPORAL PATTERNS & SEASONALITY
# ============================================================================
print("\n4. Generating Temporal Patterns & Seasonality Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Monthly revenue by category
top_categories = df.groupby('category')['revenue'].sum().sort_values(ascending=False).head(5).index
for category in top_categories:
    cat_monthly = df[df['category'] == category].groupby('year_month')['revenue'].sum()
    cat_monthly.index = cat_monthly.index.astype(str)
    axes[0, 0].plot(range(len(cat_monthly)), cat_monthly.values,
                   marker='o', linewidth=2, label=category, markersize=6)
axes[0, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Top 5 Categories - Monthly Revenue Trends', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='best', fontsize=9, ncol=2)
axes[0, 0].grid(True, alpha=0.3)

# Monthly revenue by region
for region in df['region'].unique():
    region_monthly = df[df['region'] == region].groupby('year_month')['revenue'].sum()
    region_monthly.index = region_monthly.index.astype(str)
    axes[0, 1].plot(range(len(region_monthly)), region_monthly.values,
                   marker='s', linewidth=2, label=region, markersize=6)
axes[0, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Monthly Revenue Trends by Region', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc='best', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Revenue by month
monthly_avg = df.groupby('month')['revenue'].mean()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[1, 0].bar(monthly_avg.index, monthly_avg.values,
              color=plt.cm.coolwarm(np.linspace(0, 1, len(monthly_avg))),
              alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Average Revenue ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Average Revenue by Month', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(monthly_avg.index)
axes[1, 0].set_xticklabels([month_names[i-1] for i in monthly_avg.index])
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(monthly_avg.values):
    axes[1, 0].text(monthly_avg.index[i], v, f'${v:.0f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)

# Category × Month heatmap
category_month = df.groupby(['category', 'month'])['revenue'].mean().unstack(fill_value=0)
sns.heatmap(category_month, annot=True, fmt='.0f', cmap='RdYlGn', 
           linewidths=1, cbar_kws={'label': 'Avg Revenue ($)'}, 
           ax=axes[1, 1], square=False)
axes[1, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Category', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Revenue Heatmap: Category × Month', fontsize=14, fontweight='bold')
axes[1, 1].set_xticklabels(month_names, rotation=45, ha='right')

plt.suptitle('Temporal Patterns & Seasonality Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}4_temporal_patterns_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 4_temporal_patterns_analysis.png")

# ============================================================================
# 5. PERFORMANCE METRICS & KPIs
# ============================================================================
print("\n5. Generating Performance Metrics & KPIs Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Revenue distribution
axes[0, 0].hist(df['revenue'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(df['revenue'].mean(), color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: ${df["revenue"].mean():,.0f}')
axes[0, 0].axvline(df['revenue'].median(), color='blue', linestyle='--', linewidth=2.5,
                  label=f'Median: ${df["revenue"].median():,.0f}')
axes[0, 0].set_xlabel('Revenue ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Revenue Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Orders distribution
axes[0, 1].hist(df['orders'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(df['orders'].mean(), color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {df["orders"].mean():.1f}')
axes[0, 1].axvline(df['orders'].median(), color='blue', linestyle='--', linewidth=2.5,
                  label=f'Median: {df["orders"].median():.0f}')
axes[0, 1].set_xlabel('Number of Orders', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Orders Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Revenue per order distribution
axes[1, 0].hist(df['revenue_per_order'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(df['revenue_per_order'].mean(), color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: ${df["revenue_per_order"].mean():.2f}')
axes[1, 0].set_xlabel('Revenue per Order ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Revenue per Order Distribution', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Top performing days
top_days = df.nlargest(10, 'revenue')[['date', 'revenue', 'category', 'region']]
top_days['date_str'] = top_days['date'].dt.strftime('%Y-%m-%d')
axes[1, 1].barh(range(len(top_days)), top_days['revenue'],
              color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_days))),
              alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(top_days)))
axes[1, 1].set_yticklabels([f"{row['date_str']}\n{row['category']}" 
                            for _, row in top_days.iterrows()], 
                           fontsize=9, fontweight='bold')
axes[1, 1].set_xlabel('Revenue ($)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Top 10 Revenue Days', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_days['revenue']):
    axes[1, 1].text(v, i, f'${v:.0f}', va='center', fontweight='bold', fontsize=9)

plt.suptitle('Performance Metrics & KPIs Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}5_performance_metrics_kpis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 5_performance_metrics_kpis.png")

# ============================================================================
# 6. COMPREHENSIVE STATISTICAL SUMMARY DASHBOARD
# ============================================================================
print("\n6. Generating Comprehensive Statistical Summary Dashboard...")
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# Key Statistics Summary
ax1 = fig.add_subplot(gs[0, 0])
stats_summary = {
    'Total Revenue': f"${df['revenue'].sum()/1000:.0f}K",
    'Total Orders': f"{df['orders'].sum():,}",
    'Avg Revenue': f"${df['revenue'].mean():.0f}",
    'Avg Orders': f"{df['orders'].mean():.1f}",
    'Avg Rev/Order': f"${df['revenue_per_order'].mean():.2f}",
    'Categories': f"{df['category'].nunique()}"
}
y_pos = np.arange(len(stats_summary))
ax1.barh(y_pos, [1]*len(stats_summary), 
        color=plt.cm.viridis(np.linspace(0, 1, len(stats_summary))))
ax1.set_yticks(y_pos)
ax1.set_yticklabels(list(stats_summary.keys()), fontsize=11, fontweight='bold')
ax1.set_xticks([])
ax1.set_title('Key Sales Statistics', fontsize=13, fontweight='bold')
for i, (key, val) in enumerate(stats_summary.items()):
    ax1.text(0.5, i, val, ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Top Categories by Revenue
ax2 = fig.add_subplot(gs[0, 1])
top_cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=True).tail(5)
ax2.barh(range(len(top_cat_rev)), top_cat_rev.values,
        color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_cat_rev))),
        alpha=0.8, edgecolor='black')
ax2.set_yticks(range(len(top_cat_rev)))
ax2.set_yticklabels(top_cat_rev.index, fontsize=11, fontweight='bold')
ax2.set_xlabel('Revenue ($)', fontsize=11, fontweight='bold')
ax2.set_title('Top 5 Categories by Revenue', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_cat_rev.values):
    ax2.text(v, i, f'${v/1000:.0f}K', va='center', fontweight='bold', fontsize=10)

# Top Regions by Revenue
ax3 = fig.add_subplot(gs[0, 2])
top_reg_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=True)
ax3.barh(range(len(top_reg_rev)), top_reg_rev.values,
        color=plt.cm.coolwarm(np.linspace(0, 1, len(top_reg_rev))),
        alpha=0.8, edgecolor='black')
ax3.set_yticks(range(len(top_reg_rev)))
ax3.set_yticklabels(top_reg_rev.index, fontsize=11, fontweight='bold')
ax3.set_xlabel('Revenue ($)', fontsize=11, fontweight='bold')
ax3.set_title('Revenue by Region', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_reg_rev.values):
    ax3.text(v, i, f'${v/1000:.0f}K', va='center', fontweight='bold', fontsize=10)

# Revenue Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df['revenue'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax4.axvline(df['revenue'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: ${df["revenue"].mean():,.0f}')
ax4.set_xlabel('Revenue ($)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Revenue Distribution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Orders Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df['orders'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax5.axvline(df['orders'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["orders"].mean():.1f}')
ax5.set_xlabel('Orders', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Orders Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Revenue per Order Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(df['revenue_per_order'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
ax6.axvline(df['revenue_per_order'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: ${df["revenue_per_order"].mean():.2f}')
ax6.set_xlabel('Revenue per Order ($)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Revenue per Order Distribution', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# Quarterly Performance Comparison
ax7 = fig.add_subplot(gs[2, :])
quarterly_comp = df.groupby('quarter').agg({
    'revenue': 'sum',
    'orders': 'sum',
    'revenue_per_order': 'mean'
}).reset_index()
x_q = range(len(quarterly_comp))
width = 0.25
ax7.bar([x - width for x in x_q], quarterly_comp['revenue']/1000, width,
        label='Revenue (K$)', color='#2ecc71', alpha=0.8, edgecolor='black')
ax7.bar(x_q, quarterly_comp['orders']/10, width,
        label='Orders (×10)', color='#3498db', alpha=0.8, edgecolor='black')
ax7_twin = ax7.twinx()
ax7_twin.bar([x + width for x in x_q], quarterly_comp['revenue_per_order'], width,
            label='Avg Rev/Order ($)', color='#9b59b6', alpha=0.8, edgecolor='black')
ax7.set_xlabel('Quarter', fontsize=12, fontweight='bold')
ax7.set_ylabel('Revenue & Orders (Normalized)', fontsize=12, fontweight='bold')
ax7_twin.set_ylabel('Revenue per Order ($)', fontsize=12, fontweight='bold', color='#9b59b6')
ax7.set_title('Quarterly Performance Comparison', fontsize=14, fontweight='bold')
ax7.set_xticks(x_q)
ax7.set_xticklabels([f'Q{i}' for i in quarterly_comp['quarter']])
ax7.legend(loc='upper left', fontsize=10)
ax7_twin.legend(loc='upper right', fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle('Sales Analytics - Comprehensive Statistical Summary Dashboard', 
             fontsize=20, fontweight='bold', y=0.995)
plt.savefig(f'{EXPORT_PATH}6_comprehensive_statistical_summary.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 6_comprehensive_statistical_summary.png")

print("\n" + "="*60)
print("ALL SALES VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nTotal visualizations created: 6")
print(f"Export directory: {EXPORT_PATH}")
print("\nGenerated files:")
print("  1. 1_comprehensive_revenue_trends.png")
print("  2. 2_category_region_analysis.png")
print("  3. 3_revenue_orders_analysis.png")
print("  4. 4_temporal_patterns_analysis.png")
print("  5. 5_performance_metrics_kpis.png")
print("  6. 6_comprehensive_statistical_summary.png")
print("\nAll visualizations are high-resolution (300 DPI) and ready for presentation!")

