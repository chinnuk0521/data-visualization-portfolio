"""
Global Unemployment Analytics Dashboard - Advanced Analytics
Comprehensive visualization suite with advanced global unemployment analyses
Uses real-world data from World Bank, ILO, or Kaggle
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

# Load and preprocess data
print("="*70)
print("GLOBAL UNEMPLOYMENT ANALYTICS DASHBOARD")
print("="*70)
print("\nLoading unemployment dataset...")

try:
    # Try to load the dataset
    df = pd.read_csv('dataset_unemployment.csv')
    print(f"[OK] Dataset loaded: {len(df)} records")
    print(f"[OK] Columns: {list(df.columns)}")
    
    # Display first few rows to understand structure
    print("\nFirst few rows:")
    print(df.head())
    
except FileNotFoundError:
    print("\n[ERROR] 'dataset_unemployment.csv' not found!")
    print("\nPlease download the data first:")
    print("1. Visit: https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS")
    print("2. Click 'Download' → 'CSV'")
    print("3. Save as 'dataset_unemployment.csv' in this folder")
    print("\nOR run: python download_unemployment_data.py")
    exit(1)
except Exception as e:
    print(f"\n[ERROR] Error loading dataset: {str(e)}")
    print("Please check the CSV file format and try again.")
    exit(1)

# Data preprocessing and standardization
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Standardize column names (handle different formats)
column_mapping = {
    'Country Name': 'country',
    'Country': 'country',
    'country_name': 'country',
    'Country Code': 'country_code',
    'CountryCode': 'country_code',
    'country_code': 'country_code',
    'Year': 'year',
    'year': 'year',
    'Time': 'year',
    'Unemployment, total (% of total labor force)': 'unemployment_rate',
    'Unemployment Rate': 'unemployment_rate',
    'unemployment_rate': 'unemployment_rate',
    'Unemployment': 'unemployment_rate',
    'Value': 'unemployment_rate',
    'value': 'unemployment_rate'
}

# Rename columns
df.columns = [column_mapping.get(col, col) for col in df.columns]

# If data is in wide format (years as columns), convert to long format
if 'year' not in df.columns or df['year'].dtype == 'object':
    # Check if we have year columns
    year_cols = [col for col in df.columns if str(col).isdigit() and len(str(col)) == 4]
    if year_cols:
        print(f"Converting wide format to long format ({len(year_cols)} years)...")
        id_vars = [col for col in df.columns if col not in year_cols]
        df = pd.melt(df, id_vars=id_vars, value_vars=year_cols, 
                    var_name='year', value_name='unemployment_rate')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Ensure year is numeric
if 'year' in df.columns:
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

# Ensure unemployment_rate is numeric
if 'unemployment_rate' in df.columns:
    df['unemployment_rate'] = pd.to_numeric(df['unemployment_rate'], errors='coerce')
    df = df.dropna(subset=['unemployment_rate'])

# Clean country names
if 'country' in df.columns:
    df['country'] = df['country'].astype(str).str.strip()
    # Remove common prefixes/suffixes
    df['country'] = df['country'].str.replace(r'^.*,', '', regex=True)  # Remove prefixes
    df['country'] = df['country'].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces

# Add region mapping (continents)
region_mapping = {
    # North America
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    # Europe
    'Germany': 'Europe', 'France': 'Europe', 'United Kingdom': 'Europe', 'Italy': 'Europe',
    'Spain': 'Europe', 'Poland': 'Europe', 'Netherlands': 'Europe', 'Belgium': 'Europe',
    'Greece': 'Europe', 'Portugal': 'Europe', 'Sweden': 'Europe', 'Norway': 'Europe',
    'Denmark': 'Europe', 'Finland': 'Europe', 'Switzerland': 'Europe', 'Austria': 'Europe',
    # Asia
    'China': 'Asia', 'India': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia',
    'Indonesia': 'Asia', 'Thailand': 'Asia', 'Malaysia': 'Asia', 'Philippines': 'Asia',
    'Vietnam': 'Asia', 'Singapore': 'Asia', 'Bangladesh': 'Asia', 'Pakistan': 'Asia',
    'Saudi Arabia': 'Asia', 'United Arab Emirates': 'Asia', 'Turkey': 'Asia', 'Iran': 'Asia',
    # South America
    'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
    'Colombia': 'South America', 'Peru': 'South America', 'Venezuela': 'South America',
    # Africa
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Nigeria': 'Africa', 'Kenya': 'Africa',
    # Oceania
    'Australia': 'Oceania', 'New Zealand': 'Oceania'
}

df['region'] = df['country'].map(region_mapping).fillna('Other')

# Add economic groups
g7_countries = ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 'Italy', 'Japan']
g20_countries = g7_countries + ['China', 'India', 'Brazil', 'Russia', 'South Korea', 
                                'Australia', 'Mexico', 'Argentina', 'Saudi Arabia', 
                                'South Africa', 'Turkey', 'Indonesia']
brics_countries = ['Brazil', 'Russia', 'India', 'China', 'South Africa']

df['economic_group'] = 'Other'
df.loc[df['country'].isin(g7_countries), 'economic_group'] = 'G7'
df.loc[df['country'].isin(g20_countries), 'economic_group'] = 'G20'
df.loc[df['country'].isin(brics_countries), 'economic_group'] = 'BRICS'

# Calculate decade
df['decade'] = (df['year'] // 10) * 10

# Filter to reasonable year range if needed
if 'year' in df.columns:
    df = df[(df['year'] >= 2000) & (df['year'] <= 2024)]

print(f"[OK] Data cleaned and processed")
print(f"[OK] Total records: {len(df):,}")
print(f"[OK] Countries: {df['country'].nunique()}")
print(f"[OK] Year range: {df['year'].min()} - {df['year'].max()}")
print(f"[OK] Regions: {df['region'].nunique()}")

# Create exports directory
import os
os.makedirs(EXPORT_PATH, exist_ok=True)

# ============================================================================
# 1. WORLD UNEMPLOYMENT OVERVIEW - COUNTRY RANKINGS
# ============================================================================
print("\n" + "="*70)
print("1. Generating World Unemployment Overview...")
print("="*70)

# Get latest year data for each country
latest_year = df['year'].max()
latest_data = df[df['year'] == latest_year].copy()

if len(latest_data) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Top 20 countries with highest unemployment
    top_unemployed = latest_data.nlargest(20, 'unemployment_rate')
    axes[0, 0].barh(range(len(top_unemployed)), top_unemployed['unemployment_rate'],
                   color=plt.cm.Reds(np.linspace(0.5, 0.9, len(top_unemployed))),
                   alpha=0.8, edgecolor='black')
    axes[0, 0].set_yticks(range(len(top_unemployed)))
    axes[0, 0].set_yticklabels(top_unemployed['country'], fontsize=10, fontweight='bold')
    axes[0, 0].set_xlabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'Top 20 Countries - Highest Unemployment ({latest_year})', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(top_unemployed['unemployment_rate']):
        axes[0, 0].text(v, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # Top 20 countries with lowest unemployment
    bottom_unemployed = latest_data.nsmallest(20, 'unemployment_rate')
    axes[0, 1].barh(range(len(bottom_unemployed)), bottom_unemployed['unemployment_rate'],
                    color=plt.cm.Greens(np.linspace(0.5, 0.9, len(bottom_unemployed))),
                    alpha=0.8, edgecolor='black')
    axes[0, 1].set_yticks(range(len(bottom_unemployed)))
    axes[0, 1].set_yticklabels(bottom_unemployed['country'], fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Top 20 Countries - Lowest Unemployment ({latest_year})', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(bottom_unemployed['unemployment_rate']):
        axes[0, 1].text(v, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # Unemployment distribution
    axes[1, 0].hist(latest_data['unemployment_rate'], bins=30, 
                   color='#3498db', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(latest_data['unemployment_rate'].mean(), color='red', 
                      linestyle='--', linewidth=2.5,
                      label=f'Mean: {latest_data["unemployment_rate"].mean():.1f}%')
    axes[1, 0].axvline(latest_data['unemployment_rate'].median(), color='green', 
                      linestyle='--', linewidth=2.5,
                      label=f'Median: {latest_data["unemployment_rate"].median():.1f}%')
    axes[1, 0].set_xlabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Countries', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Global Unemployment Distribution ({latest_year})', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Regional comparison
    regional_avg = latest_data.groupby('region')['unemployment_rate'].mean().sort_values(ascending=True)
    axes[1, 1].barh(range(len(regional_avg)), regional_avg.values,
                   color=plt.cm.viridis(np.linspace(0, 1, len(regional_avg))),
                   alpha=0.8, edgecolor='black')
    axes[1, 1].set_yticks(range(len(regional_avg)))
    axes[1, 1].set_yticklabels(regional_avg.index, fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'Average Unemployment by Region ({latest_year})', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(regional_avg.values):
        axes[1, 1].text(v, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    plt.suptitle('World Unemployment Overview Dashboard', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{EXPORT_PATH}1_world_unemployment_overview.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 1_world_unemployment_overview.png")
else:
    print("   [SKIP] Insufficient data for latest year analysis")

# ============================================================================
# 2. GLOBAL TRENDS DASHBOARD - TIME SERIES ANALYSIS
# ============================================================================
print("\n2. Generating Global Trends Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Global average unemployment over time
global_trends = df.groupby('year')['unemployment_rate'].agg(['mean', 'median', 'std']).reset_index()

axes[0, 0].plot(global_trends['year'], global_trends['mean'], 
               marker='o', linewidth=2.5, label='Mean', color='#2ecc71', markersize=8)
axes[0, 0].plot(global_trends['year'], global_trends['median'], 
               marker='s', linewidth=2.5, label='Median', color='#3498db', markersize=8)
axes[0, 0].fill_between(global_trends['year'], 
                       global_trends['mean'] - global_trends['std'],
                       global_trends['mean'] + global_trends['std'],
                       alpha=0.3, color='#2ecc71', label='±1 Std Dev')
axes[0, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Global Average Unemployment Trends', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Regional trends over time
top_regions = df['region'].value_counts().head(5).index
for region in top_regions:
    region_data = df[df['region'] == region].groupby('year')['unemployment_rate'].mean().reset_index()
    axes[0, 1].plot(region_data['year'], region_data['unemployment_rate'],
                   marker='o', linewidth=2, label=region, markersize=6)
axes[0, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Unemployment Trends by Region', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc='best', fontsize=9, ncol=2)
axes[0, 1].grid(True, alpha=0.3)

# Economic group trends
for group in ['G7', 'G20', 'BRICS']:
    group_data = df[df['economic_group'] == group].groupby('year')['unemployment_rate'].mean().reset_index()
    if len(group_data) > 0:
        axes[1, 0].plot(group_data['year'], group_data['unemployment_rate'],
                       marker='s', linewidth=2.5, label=group, markersize=7)
axes[1, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Unemployment Trends by Economic Group', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Year-over-year change
yearly_change = global_trends['mean'].pct_change() * 100
axes[1, 1].bar(global_trends['year'][1:], yearly_change[1:],
              color=['#2ecc71' if x > 0 else '#e74c3c' for x in yearly_change[1:]],
              alpha=0.8, edgecolor='black')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
axes[1, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Year-over-Year Change (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Global Unemployment Year-over-Year Change', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Global Unemployment Trends Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}2_global_trends_dashboard.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 2_global_trends_dashboard.png")

# ============================================================================
# 3. REGIONAL COMPARISON ANALYSIS
# ============================================================================
print("\n3. Generating Regional Comparison Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Regional statistics
regional_stats = df.groupby('region').agg({
    'unemployment_rate': ['mean', 'median', 'std', 'min', 'max']
}).reset_index()
regional_stats.columns = ['region', 'mean', 'median', 'std', 'min', 'max']
regional_stats = regional_stats.sort_values('mean', ascending=True)

x_pos = np.arange(len(regional_stats))
width = 0.15
axes[0, 0].bar(x_pos - 2*width, regional_stats['min'], width,
              label='Min', color='#2ecc71', alpha=0.8, edgecolor='black')
axes[0, 0].bar(x_pos - width, regional_stats['mean'], width,
              label='Mean', color='#3498db', alpha=0.8, edgecolor='black')
axes[0, 0].bar(x_pos, regional_stats['median'], width,
              label='Median', color='#9b59b6', alpha=0.8, edgecolor='black')
axes[0, 0].bar(x_pos + width, regional_stats['max'], width,
              label='Max', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('Region', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Regional Unemployment Statistics', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(regional_stats['region'], rotation=45, ha='right')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Regional box plots
regional_data = [df[df['region'] == region]['unemployment_rate'].values 
                for region in regional_stats['region']]
bp = axes[0, 1].boxplot(regional_data, labels=regional_stats['region'], 
                        patch_artist=True, vert=True)
colors_box = plt.cm.Set3(np.linspace(0, 1, len(regional_data)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0, 1].set_xlabel('Region', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Unemployment Distribution by Region', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Economic group comparison
economic_stats = df.groupby('economic_group')['unemployment_rate'].agg(['mean', 'std']).reset_index()
economic_stats = economic_stats[economic_stats['economic_group'] != 'Other'].sort_values('mean', ascending=True)

axes[1, 0].barh(range(len(economic_stats)), economic_stats['mean'],
               color=plt.cm.coolwarm(np.linspace(0, 1, len(economic_stats))),
               alpha=0.8, edgecolor='black', xerr=economic_stats['std'],
               capsize=5)
axes[1, 0].set_yticks(range(len(economic_stats)))
axes[1, 0].set_yticklabels(economic_stats['economic_group'], fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Unemployment by Economic Group', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')
for i, (mean, std) in enumerate(zip(economic_stats['mean'], economic_stats['std'])):
    axes[1, 0].text(mean, i, f'{mean:.1f}%', va='center', fontweight='bold')

# Regional heatmap by decade
if 'decade' in df.columns:
    regional_decade = df.groupby(['region', 'decade'])['unemployment_rate'].mean().unstack(fill_value=0)
    sns.heatmap(regional_decade, annot=True, fmt='.1f', cmap='RdYlGn_r', 
               linewidths=1, cbar_kws={'label': 'Unemployment Rate (%)'}, 
               ax=axes[1, 1], square=False)
    axes[1, 1].set_xlabel('Decade', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Region', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Regional Unemployment Heatmap by Decade', fontsize=14, fontweight='bold')

plt.suptitle('Regional Comparison Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}3_regional_comparison_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 3_regional_comparison_analysis.png")

# ============================================================================
# 4. COUNTRY PERFORMANCE ANALYSIS
# ============================================================================
print("\n4. Generating Country Performance Analysis...")

# Select countries with most data points
country_counts = df['country'].value_counts()
top_countries = country_counts.head(15).index

if len(top_countries) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Time series for top countries
    for country in top_countries:
        country_data = df[df['country'] == country].groupby('year')['unemployment_rate'].mean().reset_index()
        axes[0, 0].plot(country_data['year'], country_data['unemployment_rate'],
                       marker='o', linewidth=2, label=country, markersize=5)
    axes[0, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Unemployment Trends - Top 15 Countries by Data Points', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=8, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average unemployment by country
    country_avg = df.groupby('country')['unemployment_rate'].agg(['mean', 'std']).reset_index()
    country_avg = country_avg.sort_values('mean', ascending=False).head(20)
    
    axes[0, 1].barh(range(len(country_avg)), country_avg['mean'],
                    color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(country_avg))),
                    alpha=0.8, edgecolor='black', xerr=country_avg['std'], capsize=3)
    axes[0, 1].set_yticks(range(len(country_avg)))
    axes[0, 1].set_yticklabels(country_avg['country'], fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Top 20 Countries - Average Unemployment', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Volatility analysis (std dev)
    country_volatility = df.groupby('country')['unemployment_rate'].std().sort_values(ascending=False).head(15)
    axes[1, 0].barh(range(len(country_volatility)), country_volatility.values,
                    color=plt.cm.plasma(np.linspace(0, 1, len(country_volatility))),
                    alpha=0.8, edgecolor='black')
    axes[1, 0].set_yticks(range(len(country_volatility)))
    axes[1, 0].set_yticklabels(country_volatility.index, fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel('Standard Deviation (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Countries with Highest Unemployment Volatility', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Improvement/Worsening analysis (comparing first vs last available year)
    country_trends = []
    for country in top_countries:
        country_data = df[df['country'] == country].sort_values('year')
        if len(country_data) >= 2:
            first_rate = country_data.iloc[0]['unemployment_rate']
            last_rate = country_data.iloc[-1]['unemployment_rate']
            change = last_rate - first_rate
            country_trends.append({'country': country, 'change': change, 
                                 'first': first_rate, 'last': last_rate})
    
    if country_trends:
        trend_df = pd.DataFrame(country_trends).sort_values('change')
        colors_trend = ['#2ecc71' if x < 0 else '#e74c3c' for x in trend_df['change']]
        axes[1, 1].barh(range(len(trend_df)), trend_df['change'],
                       color=colors_trend, alpha=0.8, edgecolor='black')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        axes[1, 1].set_yticks(range(len(trend_df)))
        axes[1, 1].set_yticklabels(trend_df['country'], fontsize=10, fontweight='bold')
        axes[1, 1].set_xlabel('Change in Unemployment Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Unemployment Change Over Time Period', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(trend_df['change']):
            axes[1, 1].text(v, i, f'{v:+.1f}%', va='center', 
                           fontweight='bold', fontsize=9)
    
    plt.suptitle('Country Performance Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{EXPORT_PATH}4_country_performance_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 4_country_performance_analysis.png")

# ============================================================================
# 5. TEMPORAL PATTERNS & DECADE ANALYSIS
# ============================================================================
print("\n5. Generating Temporal Patterns & Decade Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Decade comparison
if 'decade' in df.columns:
    decade_stats = df.groupby('decade')['unemployment_rate'].agg(['mean', 'std']).reset_index()
    axes[0, 0].bar(decade_stats['decade'], decade_stats['mean'],
                  color=plt.cm.viridis(np.linspace(0, 1, len(decade_stats))),
                  alpha=0.8, edgecolor='black', yerr=decade_stats['std'], capsize=5)
    axes[0, 0].set_xlabel('Decade', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Average Unemployment by Decade', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, (dec, mean) in enumerate(zip(decade_stats['decade'], decade_stats['mean'])):
        axes[0, 0].text(dec, mean, f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

# Year distribution
yearly_counts = df['year'].value_counts().sort_index()
axes[0, 1].bar(yearly_counts.index, yearly_counts.values,
              color=plt.cm.coolwarm(np.linspace(0, 1, len(yearly_counts))),
              alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Data Points', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Data Coverage by Year', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Unemployment rate distribution by year (sample years)
sample_years = sorted(df['year'].unique())[::max(1, len(df['year'].unique())//5)]
if len(sample_years) > 0:
    for year in sample_years[:5]:
        year_data = df[df['year'] == year]['unemployment_rate']
        axes[1, 0].hist(year_data, bins=20, alpha=0.5, label=str(year), edgecolor='black')
    axes[1, 0].set_xlabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Unemployment Distribution Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

# Heatmap: Top countries × Years
top_countries_heatmap = country_counts.head(10).index
if len(top_countries_heatmap) > 0:
    country_year = df[df['country'].isin(top_countries_heatmap)].groupby(['country', 'year'])['unemployment_rate'].mean().unstack(fill_value=0)
    if len(country_year) > 0:
        sns.heatmap(country_year, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   linewidths=0.5, cbar_kws={'label': 'Unemployment Rate (%)'}, 
                   ax=axes[1, 1], square=False)
        axes[1, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Country', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Unemployment Heatmap: Top Countries × Years', 
                            fontsize=14, fontweight='bold')

plt.suptitle('Temporal Patterns & Decade Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}5_temporal_patterns_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 5_temporal_patterns_analysis.png")

# ============================================================================
# 6. STATISTICAL SUMMARY & DISTRIBUTION ANALYSIS
# ============================================================================
print("\n6. Generating Statistical Summary & Distribution Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Overall distribution
axes[0, 0].hist(df['unemployment_rate'], bins=40, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(df['unemployment_rate'].mean(), color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {df["unemployment_rate"].mean():.2f}%')
axes[0, 0].axvline(df['unemployment_rate'].median(), color='green', linestyle='--', linewidth=2.5,
                  label=f'Median: {df["unemployment_rate"].median():.2f}%')
axes[0, 0].set_xlabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Global Unemployment Rate Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Box plot by region
regional_data_box = [df[df['region'] == region]['unemployment_rate'].values 
                     for region in regional_stats['region']]
bp = axes[0, 1].boxplot(regional_data_box, labels=regional_stats['region'], 
                        patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0, 1].set_xlabel('Region', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Unemployment Distribution by Region', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Statistical summary table
summary_stats = {
    'Total Records': f"{len(df):,}",
    'Countries': f"{df['country'].nunique()}",
    'Mean Rate': f"{df['unemployment_rate'].mean():.2f}%",
    'Median Rate': f"{df['unemployment_rate'].median():.2f}%",
    'Std Deviation': f"{df['unemployment_rate'].std():.2f}%",
    'Min Rate': f"{df['unemployment_rate'].min():.2f}%",
    'Max Rate': f"{df['unemployment_rate'].max():.2f}%"
}

y_pos = np.arange(len(summary_stats))
axes[1, 0].barh(y_pos, [1]*len(summary_stats), 
               color=plt.cm.viridis(np.linspace(0, 1, len(summary_stats))))
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(list(summary_stats.keys()), fontsize=11, fontweight='bold')
axes[1, 0].set_xticks([])
axes[1, 0].set_title('Key Statistics Summary', fontsize=14, fontweight='bold')
for i, (key, val) in enumerate(summary_stats.items()):
    axes[1, 0].text(0.5, i, val, ha='center', va='center', fontsize=12, 
                   fontweight='bold', color='white')
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)
axes[1, 0].spines['bottom'].set_visible(False)
axes[1, 0].spines['left'].set_visible(False)

# Yearly statistics trend
yearly_stats = df.groupby('year')['unemployment_rate'].agg(['mean', 'std']).reset_index()
axes[1, 1].plot(yearly_stats['year'], yearly_stats['mean'],
               marker='o', linewidth=2.5, label='Mean', color='#2ecc71', markersize=8)
axes[1, 1].fill_between(yearly_stats['year'],
                        yearly_stats['mean'] - yearly_stats['std'],
                        yearly_stats['mean'] + yearly_stats['std'],
                        alpha=0.3, color='#2ecc71', label='±1 Std Dev')
axes[1, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Yearly Statistics with Standard Deviation', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Statistical Summary & Distribution Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}6_statistical_summary.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 6_statistical_summary.png")

# ============================================================================
# 7. COMPREHENSIVE HEATMAP MATRICES
# ============================================================================
print("\n7. Generating Comprehensive Heatmap Matrices...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Country × Year heatmap (top countries)
top_countries_hm = country_counts.head(15).index
if len(top_countries_hm) > 0:
    country_year_hm = df[df['country'].isin(top_countries_hm)].pivot_table(
        values='unemployment_rate', index='country', columns='year', aggfunc='mean')
    if len(country_year_hm) > 0:
        sns.heatmap(country_year_hm, annot=False, cmap='RdYlGn_r', 
                   linewidths=0.5, cbar_kws={'label': 'Unemployment Rate (%)'}, 
                   ax=axes[0, 0], square=False)
        axes[0, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Country', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Unemployment Heatmap: Country × Year', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='y', labelsize=8)

# Region × Decade heatmap
if 'decade' in df.columns:
    region_decade_hm = df.pivot_table(values='unemployment_rate', index='region', 
                                     columns='decade', aggfunc='mean')
    if len(region_decade_hm) > 0:
        sns.heatmap(region_decade_hm, annot=True, fmt='.1f', cmap='YlOrRd', 
                   linewidths=1, cbar_kws={'label': 'Unemployment Rate (%)'}, 
                   ax=axes[0, 1], square=False)
        axes[0, 1].set_xlabel('Decade', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Region', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Unemployment Heatmap: Region × Decade', 
                            fontsize=14, fontweight='bold')

# Economic Group × Year
economic_year = df[df['economic_group'] != 'Other'].pivot_table(
    values='unemployment_rate', index='economic_group', columns='year', aggfunc='mean')
if len(economic_year) > 0:
    sns.heatmap(economic_year, annot=True, fmt='.1f', cmap='coolwarm', 
               linewidths=1, cbar_kws={'label': 'Unemployment Rate (%)'}, 
               ax=axes[1, 0], square=False, center=economic_year.values.mean())
    axes[1, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Economic Group', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Unemployment Heatmap: Economic Group × Year', 
                         fontsize=14, fontweight='bold')

# Correlation matrix (if we have multiple years)
if len(df['year'].unique()) > 5:
    # Create correlation between years
    year_corr = df.pivot_table(values='unemployment_rate', index='country', 
                               columns='year', aggfunc='mean').corr()
    sns.heatmap(year_corr, annot=True, fmt='.2f', cmap='coolwarm', 
               linewidths=0.5, cbar_kws={'label': 'Correlation'}, 
               ax=axes[1, 1], square=True, center=0, vmin=-1, vmax=1)
    axes[1, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Year', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Year-to-Year Correlation Matrix', fontsize=14, fontweight='bold')

plt.suptitle('Comprehensive Heatmap Matrices', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}7_comprehensive_heatmaps.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 7_comprehensive_heatmaps.png")

# ============================================================================
# 8. COMPREHENSIVE STATISTICAL SUMMARY DASHBOARD
# ============================================================================
print("\n8. Generating Comprehensive Statistical Summary Dashboard...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# Key Statistics
ax1 = fig.add_subplot(gs[0, 0])
stats_summary = {
    'Total Records': f"{len(df):,}",
    'Countries': f"{df['country'].nunique()}",
    'Years Covered': f"{df['year'].max() - df['year'].min() + 1}",
    'Avg Rate': f"{df['unemployment_rate'].mean():.2f}%",
    'Median Rate': f"{df['unemployment_rate'].median():.2f}%",
    'Regions': f"{df['region'].nunique()}"
}
y_pos = np.arange(len(stats_summary))
ax1.barh(y_pos, [1]*len(stats_summary), 
        color=plt.cm.viridis(np.linspace(0, 1, len(stats_summary))))
ax1.set_yticks(y_pos)
ax1.set_yticklabels(list(stats_summary.keys()), fontsize=11, fontweight='bold')
ax1.set_xticks([])
ax1.set_title('Key Statistics', fontsize=13, fontweight='bold')
for i, (key, val) in enumerate(stats_summary.items()):
    ax1.text(0.5, i, val, ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Top 10 Countries
ax2 = fig.add_subplot(gs[0, 1])
top_10 = df.groupby('country')['unemployment_rate'].mean().sort_values(ascending=True).tail(10)
ax2.barh(range(len(top_10)), top_10.values,
        color=plt.cm.Reds(np.linspace(0.5, 0.9, len(top_10))),
        alpha=0.8, edgecolor='black')
ax2.set_yticks(range(len(top_10)))
ax2.set_yticklabels(top_10.index, fontsize=10, fontweight='bold')
ax2.set_xlabel('Avg Unemployment (%)', fontsize=11, fontweight='bold')
ax2.set_title('Top 10 Countries - Highest Unemployment', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Regional Average
ax3 = fig.add_subplot(gs[0, 2])
regional_avg_summary = df.groupby('region')['unemployment_rate'].mean().sort_values(ascending=True)
ax3.barh(range(len(regional_avg_summary)), regional_avg_summary.values,
        color=plt.cm.viridis(np.linspace(0, 1, len(regional_avg_summary))),
        alpha=0.8, edgecolor='black')
ax3.set_yticks(range(len(regional_avg_summary)))
ax3.set_yticklabels(regional_avg_summary.index, fontsize=10, fontweight='bold')
ax3.set_xlabel('Avg Unemployment (%)', fontsize=11, fontweight='bold')
ax3.set_title('Average by Region', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df['unemployment_rate'], bins=40, color='#3498db', alpha=0.7, edgecolor='black')
ax4.axvline(df['unemployment_rate'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["unemployment_rate"].mean():.2f}%')
ax4.set_xlabel('Unemployment Rate (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Global Distribution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Yearly Trend
ax5 = fig.add_subplot(gs[1, 1])
yearly_avg = df.groupby('year')['unemployment_rate'].mean()
ax5.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2.5, 
        color='#2ecc71', markersize=6)
ax5.fill_between(yearly_avg.index, yearly_avg.values, alpha=0.3, color='#2ecc71')
ax5.set_xlabel('Year', fontsize=11, fontweight='bold')
ax5.set_ylabel('Avg Unemployment (%)', fontsize=11, fontweight='bold')
ax5.set_title('Global Average Trend', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Economic Groups
ax6 = fig.add_subplot(gs[1, 2])
economic_avg = df[df['economic_group'] != 'Other'].groupby('economic_group')['unemployment_rate'].mean()
if len(economic_avg) > 0:
    ax6.bar(range(len(economic_avg)), economic_avg.values,
           color=plt.cm.Set3(np.linspace(0, 1, len(economic_avg))),
           alpha=0.8, edgecolor='black')
    ax6.set_xticks(range(len(economic_avg)))
    ax6.set_xticklabels(economic_avg.index, fontsize=10, fontweight='bold')
    ax6.set_ylabel('Avg Unemployment (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Economic Groups', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

# Comprehensive Time Series
ax7 = fig.add_subplot(gs[2, :])
# Global average
global_avg = df.groupby('year')['unemployment_rate'].mean()
ax7.plot(global_avg.index, global_avg.values, marker='o', linewidth=3, 
        label='Global Average', color='#2ecc71', markersize=8)
# Regional averages
for region in top_regions[:4]:
    region_avg = df[df['region'] == region].groupby('year')['unemployment_rate'].mean()
    ax7.plot(region_avg.index, region_avg.values, marker='s', linewidth=2, 
            label=region, markersize=5, alpha=0.7)
ax7.set_xlabel('Year', fontsize=12, fontweight='bold')
ax7.set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
ax7.set_title('Comprehensive Time Series: Global & Regional Trends', 
             fontsize=14, fontweight='bold')
ax7.legend(loc='best', fontsize=10, ncol=3)
ax7.grid(True, alpha=0.3)

plt.suptitle('Global Unemployment Analytics - Comprehensive Statistical Summary Dashboard', 
             fontsize=20, fontweight='bold', y=0.995)
plt.savefig(f'{EXPORT_PATH}8_comprehensive_statistical_summary.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 8_comprehensive_statistical_summary.png")

# ============================================================================
# 9. PANDEMIC IMPACT ANALYSIS (Pre/Post COVID-19)
# ============================================================================
print("\n9. Generating Pandemic Impact Analysis...")

# Define pandemic period (2020-2021 as peak impact)
pandemic_years = [2020, 2021]
pre_pandemic = df[df['year'] < 2020]
post_pandemic = df[df['year'] >= 2020]

if len(pre_pandemic) > 0 and len(post_pandemic) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Pre vs Post pandemic comparison
    pre_avg = pre_pandemic.groupby('year')['unemployment_rate'].mean()
    post_avg = post_pandemic.groupby('year')['unemployment_rate'].mean()
    
    axes[0, 0].plot(pre_avg.index, pre_avg.values, marker='o', linewidth=2.5,
                   label='Pre-Pandemic (Before 2020)', color='#2ecc71', markersize=8)
    axes[0, 0].plot(post_avg.index, post_avg.values, marker='s', linewidth=2.5,
                   label='Post-Pandemic (2020+)', color='#e74c3c', markersize=8)
    axes[0, 0].axvline(x=2020, color='black', linestyle='--', linewidth=2, 
                      label='Pandemic Start (2020)')
    axes[0, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Pandemic Impact: Pre vs Post COVID-19', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Regional pandemic impact
    regional_impact = []
    for region in df['region'].unique():
        region_pre = pre_pandemic[pre_pandemic['region'] == region]['unemployment_rate'].mean()
        region_post = post_pandemic[post_pandemic['region'] == region]['unemployment_rate'].mean()
        if not pd.isna(region_pre) and not pd.isna(region_post):
            regional_impact.append({
                'region': region,
                'pre': region_pre,
                'post': region_post,
                'change': region_post - region_pre
            })
    
    if regional_impact:
        impact_df = pd.DataFrame(regional_impact).sort_values('change', ascending=True)
        x_pos = np.arange(len(impact_df))
        width = 0.35
        axes[0, 1].bar(x_pos - width/2, impact_df['pre'], width,
                      label='Pre-Pandemic', color='#2ecc71', alpha=0.8, edgecolor='black')
        axes[0, 1].bar(x_pos + width/2, impact_df['post'], width,
                      label='Post-Pandemic', color='#e74c3c', alpha=0.8, edgecolor='black')
        axes[0, 1].set_xlabel('Region', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Regional Pandemic Impact Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(impact_df['region'], rotation=45, ha='right')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Change in unemployment (pre to post)
    country_impact = []
    for country in df['country'].unique():
        country_pre = pre_pandemic[pre_pandemic['country'] == country]['unemployment_rate'].mean()
        country_post = post_pandemic[post_pandemic['country'] == country]['unemployment_rate'].mean()
        if not pd.isna(country_pre) and not pd.isna(country_post):
            country_impact.append({
                'country': country,
                'change': country_post - country_pre
            })
    
    if country_impact:
        country_impact_df = pd.DataFrame(country_impact).sort_values('change', ascending=False).head(20)
        colors_impact = ['#e74c3c' if x > 0 else '#2ecc71' for x in country_impact_df['change']]
        axes[1, 0].barh(range(len(country_impact_df)), country_impact_df['change'],
                        color=colors_impact, alpha=0.8, edgecolor='black')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        axes[1, 0].set_yticks(range(len(country_impact_df)))
        axes[1, 0].set_yticklabels(country_impact_df['country'], fontsize=9, fontweight='bold')
        axes[1, 0].set_xlabel('Change in Unemployment Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Top 20 Countries - Pandemic Impact (Post - Pre)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(country_impact_df['change']):
            axes[1, 0].text(v, i, f'{v:+.1f}%', va='center', fontweight='bold', fontsize=8)
    
    # Year-by-year during pandemic
    pandemic_yearly = df[df['year'] >= 2018].groupby('year')['unemployment_rate'].agg(['mean', 'std']).reset_index()
    axes[1, 1].plot(pandemic_yearly['year'], pandemic_yearly['mean'],
                   marker='o', linewidth=3, color='#e74c3c', markersize=10)
    axes[1, 1].fill_between(pandemic_yearly['year'],
                            pandemic_yearly['mean'] - pandemic_yearly['std'],
                            pandemic_yearly['mean'] + pandemic_yearly['std'],
                            alpha=0.3, color='#e74c3c')
    axes[1, 1].axvspan(2020, 2021, alpha=0.2, color='red', label='Peak Pandemic Period')
    axes[1, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Unemployment During Pandemic Period (2018-2024)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Pandemic Impact Analysis (COVID-19)', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{EXPORT_PATH}9_pandemic_impact_analysis.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: 9_pandemic_impact_analysis.png")
else:
    print("   [SKIP] Insufficient data for pandemic analysis")

# ============================================================================
# 10. PREDICTIVE TRENDS & FORECASTING
# ============================================================================
print("\n10. Generating Predictive Trends & Forecasting...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Simple linear trend projection
global_trend = df.groupby('year')['unemployment_rate'].mean().reset_index()
if len(global_trend) >= 5:
    # Fit linear regression
    x = global_trend['year'].values
    y = global_trend['unemployment_rate'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Extend to future years
    future_years = np.arange(df['year'].max() + 1, df['year'].max() + 6)
    future_values = slope * future_years + intercept
    
    axes[0, 0].plot(global_trend['year'], global_trend['unemployment_rate'],
                   marker='o', linewidth=2.5, label='Historical Data', 
                   color='#3498db', markersize=8)
    axes[0, 0].plot(future_years, future_values, '--', linewidth=2.5,
                   label=f'Projected Trend (R²={r_value**2:.3f})', color='#e74c3c', markersize=8)
    axes[0, 0].axvline(x=df['year'].max(), color='black', linestyle=':', linewidth=2,
                      label='Current Year')
    axes[0, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Unemployment Trend Projection (5-Year Forecast)', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add confidence interval
    std_err_future = std_err * np.sqrt(1 + 1/len(x) + (future_years - x.mean())**2 / np.sum((x - x.mean())**2))
    axes[0, 0].fill_between(future_years, 
                            future_values - 1.96*std_err_future,
                            future_values + 1.96*std_err_future,
                            alpha=0.2, color='#e74c3c', label='95% Confidence Interval')

# Moving average trends
if len(global_trend) >= 3:
    global_trend['ma_3'] = global_trend['unemployment_rate'].rolling(window=3, center=True).mean()
    global_trend['ma_5'] = global_trend['unemployment_rate'].rolling(window=5, center=True).mean()
    
    axes[0, 1].plot(global_trend['year'], global_trend['unemployment_rate'],
                   marker='o', linewidth=1.5, label='Actual', color='#95a5a6', 
                   markersize=5, alpha=0.6)
    axes[0, 1].plot(global_trend['year'], global_trend['ma_3'],
                   linewidth=2.5, label='3-Year Moving Average', color='#3498db')
    axes[0, 1].plot(global_trend['year'], global_trend['ma_5'],
                   linewidth=2.5, label='5-Year Moving Average', color='#2ecc71')
    axes[0, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Moving Average Trends', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

# Volatility analysis (rolling standard deviation)
if len(global_trend) >= 5:
    global_trend['volatility'] = global_trend['unemployment_rate'].rolling(window=5).std()
    axes[1, 0].plot(global_trend['year'], global_trend['volatility'],
                   marker='s', linewidth=2.5, color='#9b59b6', markersize=7)
    axes[1, 0].fill_between(global_trend['year'], global_trend['volatility'],
                            alpha=0.3, color='#9b59b6')
    axes[1, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Volatility (Std Dev)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Unemployment Volatility Over Time (5-Year Rolling)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

# Growth rate analysis
if len(global_trend) > 1:
    global_trend['growth_rate'] = global_trend['unemployment_rate'].pct_change() * 100
    colors_growth = ['#2ecc71' if x < 0 else '#e74c3c' for x in global_trend['growth_rate'][1:]]
    axes[1, 1].bar(global_trend['year'][1:], global_trend['growth_rate'][1:],
                  color=colors_growth, alpha=0.8, edgecolor='black')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    axes[1, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Year-over-Year Growth Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Unemployment Growth Rate Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Predictive Trends & Forecasting Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}10_predictive_trends.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 10_predictive_trends.png")

# ============================================================================
# 11. VOLATILITY & STABILITY ANALYSIS
# ============================================================================
print("\n11. Generating Volatility & Stability Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Country stability (coefficient of variation)
country_stability = df.groupby('country')['unemployment_rate'].agg(['mean', 'std']).reset_index()
country_stability['cv'] = (country_stability['std'] / country_stability['mean']) * 100
country_stability = country_stability[country_stability['mean'] > 0].sort_values('cv', ascending=True).head(20)

axes[0, 0].barh(range(len(country_stability)), country_stability['cv'],
               color=plt.cm.Greens(np.linspace(0.5, 0.9, len(country_stability))),
               alpha=0.8, edgecolor='black')
axes[0, 0].set_yticks(range(len(country_stability)))
axes[0, 0].set_yticklabels(country_stability['country'], fontsize=10, fontweight='bold')
axes[0, 0].set_xlabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Most Stable Countries (Lowest Volatility)', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Most volatile countries
most_volatile = country_stability.sort_values('cv', ascending=False).head(20)
axes[0, 1].barh(range(len(most_volatile)), most_volatile['cv'],
               color=plt.cm.Reds(np.linspace(0.5, 0.9, len(most_volatile))),
               alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(most_volatile)))
axes[0, 1].set_yticklabels(most_volatile['country'], fontsize=10, fontweight='bold')
axes[0, 1].set_xlabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Most Volatile Countries (Highest Variability)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Stability vs Average unemployment
if len(country_stability) > 10:
    scatter = axes[1, 0].scatter(country_stability['mean'], country_stability['cv'],
                                s=100, alpha=0.6, c=range(len(country_stability)),
                                cmap='viridis', edgecolors='black', linewidth=1)
    axes[1, 0].set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Stability vs Average Unemployment', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

# Regional stability comparison
regional_stability = df.groupby('region')['unemployment_rate'].agg(['mean', 'std']).reset_index()
regional_stability['cv'] = (regional_stability['std'] / regional_stability['mean']) * 100
regional_stability = regional_stability.sort_values('cv', ascending=True)

axes[1, 1].barh(range(len(regional_stability)), regional_stability['cv'],
               color=plt.cm.plasma(np.linspace(0, 1, len(regional_stability))),
               alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(regional_stability)))
axes[1, 1].set_yticklabels(regional_stability['region'], fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Regional Stability Comparison', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.suptitle('Volatility & Stability Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}11_volatility_stability_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 11_volatility_stability_analysis.png")

# ============================================================================
# 12. COUNTRY CLUSTERING & GROUPING ANALYSIS
# ============================================================================
print("\n12. Generating Country Clustering & Grouping Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Group countries by unemployment level
country_avg_cluster = df.groupby('country')['unemployment_rate'].mean().reset_index()
country_avg_cluster['category'] = pd.cut(country_avg_cluster['unemployment_rate'], 
                                        bins=[0, 5, 10, 15, 25, 100],
                                        labels=['Very Low (<5%)', 'Low (5-10%)', 
                                               'Moderate (10-15%)', 'High (15-25%)', 
                                               'Very High (>25%)'])

category_counts = country_avg_cluster['category'].value_counts()
axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
              startangle=90, colors=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(category_counts))),
              textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[0, 0].set_title('Country Distribution by Unemployment Level', fontsize=14, fontweight='bold')

# Economic group performance
economic_perf = df[df['economic_group'] != 'Other'].groupby('economic_group')['unemployment_rate'].agg(['mean', 'std']).reset_index()
economic_perf = economic_perf.sort_values('mean', ascending=True)

x_pos = np.arange(len(economic_perf))
axes[0, 1].bar(x_pos, economic_perf['mean'],
              color=plt.cm.Set3(np.linspace(0, 1, len(economic_perf))),
              alpha=0.8, edgecolor='black', yerr=economic_perf['std'], capsize=5)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(economic_perf['economic_group'], fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Economic Group Performance Comparison', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Top vs Bottom performers
top_bottom = pd.concat([
    country_avg_cluster.nsmallest(10, 'unemployment_rate'),
    country_avg_cluster.nlargest(10, 'unemployment_rate')
]).reset_index(drop=True)
top_bottom['group'] = ['Top 10' if i < 10 else 'Bottom 10' for i in range(len(top_bottom))]

axes[1, 0].barh(range(len(top_bottom)), top_bottom['unemployment_rate'].values,
               color=['#2ecc71' if g == 'Top 10' else '#e74c3c' for g in top_bottom['group']],
               alpha=0.8, edgecolor='black')
axes[1, 0].set_yticks(range(len(top_bottom)))
axes[1, 0].set_yticklabels(top_bottom['country'].values, fontsize=9, fontweight='bold')
axes[1, 0].set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Top 10 vs Bottom 10 Performers', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Regional distribution
regional_dist = df.groupby('region')['country'].nunique().sort_values(ascending=True)
axes[1, 1].barh(range(len(regional_dist)), regional_dist.values,
               color=plt.cm.viridis(np.linspace(0, 1, len(regional_dist))),
               alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(regional_dist)))
axes[1, 1].set_yticklabels(regional_dist.index, fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Number of Countries', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Country Distribution by Region', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.suptitle('Country Clustering & Grouping Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}12_country_clustering.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 12_country_clustering.png")

# ============================================================================
# 13. CHANGE ANALYSIS - IMPROVEMENT & WORSENING
# ============================================================================
print("\n13. Generating Change Analysis - Improvement & Worsening...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Countries with most improvement
country_changes = []
for country in df['country'].unique():
    country_data = df[df['country'] == country].sort_values('year')
    if len(country_data) >= 2:
        first_rate = country_data.iloc[0]['unemployment_rate']
        last_rate = country_data.iloc[-1]['unemployment_rate']
        change_pct = ((last_rate - first_rate) / first_rate) * 100 if first_rate > 0 else 0
        country_changes.append({
            'country': country,
            'first': first_rate,
            'last': last_rate,
            'change': last_rate - first_rate,
            'change_pct': change_pct
        })

if country_changes:
    changes_df = pd.DataFrame(country_changes)
    
    # Most improved (largest decrease)
    most_improved = changes_df.nsmallest(15, 'change')
    axes[0, 0].barh(range(len(most_improved)), most_improved['change'],
                   color=plt.cm.Greens(np.linspace(0.5, 0.9, len(most_improved))),
                   alpha=0.8, edgecolor='black')
    axes[0, 0].set_yticks(range(len(most_improved)))
    axes[0, 0].set_yticklabels(most_improved['country'], fontsize=10, fontweight='bold')
    axes[0, 0].set_xlabel('Change in Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Top 15 Countries - Most Improved (Largest Decrease)', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(most_improved['change']):
        axes[0, 0].text(v, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # Most worsened (largest increase)
    most_worsened = changes_df.nlargest(15, 'change')
    axes[0, 1].barh(range(len(most_worsened)), most_worsened['change'],
                   color=plt.cm.Reds(np.linspace(0.5, 0.9, len(most_worsened))),
                   alpha=0.8, edgecolor='black')
    axes[0, 1].set_yticks(range(len(most_worsened)))
    axes[0, 1].set_yticklabels(most_worsened['country'], fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel('Change in Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Top 15 Countries - Most Worsened (Largest Increase)', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(most_worsened['change']):
        axes[0, 1].text(v, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # Percentage change distribution
    axes[1, 0].hist(changes_df['change_pct'], bins=30, color='#3498db', 
                   alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(changes_df['change_pct'].mean(), color='red', 
                       linestyle='--', linewidth=2.5,
                       label=f'Mean: {changes_df["change_pct"].mean():.1f}%')
    axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1.5)
    axes[1, 0].set_xlabel('Percentage Change (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Countries', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Distribution of Percentage Changes', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Scatter: Initial vs Final
    axes[1, 1].scatter(changes_df['first'], changes_df['last'],
                      s=100, alpha=0.6, c=changes_df['change'],
                      cmap='RdYlGn', edgecolors='black', linewidth=1)
    # Add diagonal line
    max_val = max(changes_df['first'].max(), changes_df['last'].max())
    axes[1, 1].plot([0, max_val], [0, max_val], 'k--', linewidth=2, 
                   label='No Change Line', alpha=0.7)
    axes[1, 1].set_xlabel('Initial Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Final Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Initial vs Final Unemployment (Color = Change)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Change (%)', fontsize=11, fontweight='bold')

plt.suptitle('Change Analysis - Improvement & Worsening', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}13_change_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 13_change_analysis.png")

# ============================================================================
# 14. ADVANCED CORRELATION & RELATIONSHIP ANALYSIS
# ============================================================================
print("\n14. Generating Advanced Correlation & Relationship Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Year correlation matrix (if enough years)
if len(df['year'].unique()) >= 5:
    year_pivot = df.pivot_table(values='unemployment_rate', index='country', 
                               columns='year', aggfunc='mean')
    year_corr = year_pivot.corr()
    sns.heatmap(year_corr, annot=True, fmt='.2f', cmap='coolwarm', 
               linewidths=0.5, cbar_kws={'label': 'Correlation'}, 
               ax=axes[0, 0], square=True, center=0, vmin=-1, vmax=1)
    axes[0, 0].set_title('Year-to-Year Correlation Matrix', fontsize=14, fontweight='bold')

# Regional correlation
regional_pivot = df.pivot_table(values='unemployment_rate', index='year', 
                               columns='region', aggfunc='mean')
if len(regional_pivot.columns) > 1:
    regional_corr = regional_pivot.corr()
    sns.heatmap(regional_corr, annot=True, fmt='.2f', cmap='RdYlGn', 
               linewidths=1, cbar_kws={'label': 'Correlation'}, 
               ax=axes[0, 1], square=True)
    axes[0, 1].set_title('Regional Correlation Matrix', fontsize=14, fontweight='bold')

# Unemployment vs Time (with trend line)
yearly_avg_corr = df.groupby('year')['unemployment_rate'].mean().reset_index()
if len(yearly_avg_corr) >= 3:
    x = yearly_avg_corr['year'].values
    y = yearly_avg_corr['unemployment_rate'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    axes[1, 0].scatter(x, y, s=100, alpha=0.6, color='#3498db', edgecolors='black')
    axes[1, 0].plot(x, p(x), "r--", linewidth=2.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
    axes[1, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Unemployment vs Time with Trend Line', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

# Country similarity (if enough data)
if len(country_counts) >= 10:
    top_countries_sim = country_counts.head(10).index
    country_year_sim = df[df['country'].isin(top_countries_sim)].pivot_table(
        values='unemployment_rate', index='country', columns='year', aggfunc='mean')
    if len(country_year_sim) > 1:
        country_corr = country_year_sim.T.corr()
        sns.heatmap(country_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   linewidths=0.5, cbar_kws={'label': 'Correlation'}, 
                   ax=axes[1, 1], square=True, center=0, vmin=-1, vmax=1)
        axes[1, 1].set_title('Country Similarity Matrix (Correlation)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='both', labelsize=8)

plt.suptitle('Advanced Correlation & Relationship Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}14_correlation_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 14_correlation_analysis.png")

print("\n" + "="*70)
print("ALL GLOBAL UNEMPLOYMENT VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nTotal visualizations created: 14")
print(f"Export directory: {EXPORT_PATH}")
print("\nGenerated files:")
print("  1. 1_world_unemployment_overview.png")
print("  2. 2_global_trends_dashboard.png")
print("  3. 3_regional_comparison_analysis.png")
print("  4. 4_country_performance_analysis.png")
print("  5. 5_temporal_patterns_analysis.png")
print("  6. 6_statistical_summary.png")
print("  7. 7_comprehensive_heatmaps.png")
print("  8. 8_comprehensive_statistical_summary.png")
print("  9. 9_pandemic_impact_analysis.png")
print(" 10. 10_predictive_trends.png")
print(" 11. 11_volatility_stability_analysis.png")
print(" 12. 12_country_clustering.png")
print(" 13. 13_change_analysis.png")
print(" 14. 14_correlation_analysis.png")
print("\nAll visualizations are high-resolution (300 DPI) and ready for presentation!")
print("\nNote: This script adapts to your dataset structure.")
print("If you have additional data (gender, age, education, sectors),")
print("the script can be extended to include those visualizations.")

