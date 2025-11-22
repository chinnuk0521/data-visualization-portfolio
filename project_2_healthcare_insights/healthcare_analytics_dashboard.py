"""
Healthcare Data Insights - Advanced Analytics Dashboard
Comprehensive visualization suite with advanced healthcare analytics
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
print("Loading healthcare dataset...")
df = pd.read_csv('dataset_healthcare.csv')
df['admission_date'] = pd.to_datetime(df['admission_date'])
df['year'] = df['admission_date'].dt.year
df['month'] = df['admission_date'].dt.month
df['quarter'] = df['admission_date'].dt.quarter
df['year_month'] = df['admission_date'].dt.to_period('M')
df['day_of_week'] = df['admission_date'].dt.day_name()
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 100], 
                        labels=['0-18', '19-30', '31-45', '46-60', '60+'])

print(f"Dataset loaded: {len(df)} records")
print(f"Date range: {df['admission_date'].min()} to {df['admission_date'].max()}")
print(f"Total diseases: {df['disease'].nunique()}")
print(f"Recovery rate: {df['recovered'].sum()}/{len(df)} = {df['recovered'].sum()/len(df)*100:.1f}%")

# ============================================================================
# 1. COMPREHENSIVE ADMISSION TRENDS DASHBOARD
# ============================================================================
print("\n1. Generating Comprehensive Admission Trends Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Monthly admission trends
monthly_admissions = df.groupby('year_month').size().reset_index(name='count')
monthly_admissions['year_month'] = monthly_admissions['year_month'].astype(str)

axes[0, 0].plot(range(len(monthly_admissions)), monthly_admissions['count'],
               marker='o', linewidth=2.5, markersize=8, color='#3498db')
axes[0, 0].fill_between(range(len(monthly_admissions)), monthly_admissions['count'],
                       alpha=0.3, color='#3498db')
axes[0, 0].axhline(y=monthly_admissions['count'].mean(), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Mean: {monthly_admissions["count"].mean():.1f}')
axes[0, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Number of Admissions', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Monthly Admission Trends', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(range(0, len(monthly_admissions), 2))
axes[0, 0].set_xticklabels([monthly_admissions['year_month'].iloc[i] 
                            for i in range(0, len(monthly_admissions), 2)], 
                           rotation=45, ha='right')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Admission by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_admissions = df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
colors_day = plt.cm.Set3(np.linspace(0, 1, len(day_admissions)))
axes[0, 1].bar(range(len(day_admissions)), day_admissions.values,
              color=colors_day, alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Day of Week', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Admissions', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Admissions by Day of Week', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(range(len(day_admissions)))
axes[0, 1].set_xticklabels(day_admissions.index, rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(day_admissions.values):
    axes[0, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Quarterly admission trends
quarterly_admissions = df.groupby('quarter').size()
axes[1, 0].bar(quarterly_admissions.index, quarterly_admissions.values,
              color=plt.cm.viridis(np.linspace(0, 1, len(quarterly_admissions))),
              alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Quarter', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Number of Admissions', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Admissions by Quarter', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(quarterly_admissions.index)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(quarterly_admissions.values):
    axes[1, 0].text(quarterly_admissions.index[i], v, str(v), 
                   ha='center', va='bottom', fontweight='bold')

# Admission trends by disease
disease_monthly = df.groupby(['year_month', 'disease']).size().reset_index(name='count')
disease_monthly['year_month'] = disease_monthly['year_month'].astype(str)
top_diseases = df['disease'].value_counts().head(5).index

for disease in top_diseases:
    disease_data = disease_monthly[disease_monthly['disease'] == disease]
    axes[1, 1].plot(range(len(disease_data)), disease_data['count'],
                   marker='o', linewidth=2, label=disease, markersize=6)
axes[1, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Number of Admissions', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Top 5 Diseases - Monthly Admission Trends', 
                    fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=9, ncol=2)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Healthcare Admission Trends Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}1_comprehensive_admission_trends.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 1_comprehensive_admission_trends.png")

# ============================================================================
# 2. AGE DISTRIBUTION & DEMOGRAPHICS ANALYSIS
# ============================================================================
print("\n2. Generating Age Distribution & Demographics Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Age distribution histogram
axes[0, 0].hist(df['age'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {df["age"].mean():.1f} years')
axes[0, 0].axvline(df['age'].median(), color='green', linestyle='--', linewidth=2.5,
                  label=f'Median: {df["age"].median():.1f} years')
axes[0, 0].set_xlabel('Age', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Patient Age Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Age distribution by disease
age_by_disease = df.groupby(['disease', 'age_group']).size().unstack(fill_value=0)
age_by_disease_pct = age_by_disease.div(age_by_disease.sum(axis=1), axis=0) * 100
age_by_disease_pct.plot(kind='barh', stacked=True, ax=axes[0, 1], 
                        colormap='Set3', alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Percentage of Patients', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Disease', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Age Group Distribution by Disease', fontsize=14, fontweight='bold')
axes[0, 1].legend(title='Age Group', fontsize=9, loc='lower right')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Age vs Recovery Days
scatter = axes[1, 0].scatter(df['age'], df['recovery_days'],
                             c=df['recovery_rate'], cmap='RdYlGn',
                             s=100, alpha=0.6, edgecolors='black', linewidth=1)
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('Recovery Rate (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Age', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Recovery Days', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Age vs Recovery Days\n(Color = Recovery Rate)', 
                     fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Age group statistics
age_group_stats = df.groupby('age_group').agg({
    'recovery_days': 'mean',
    'recovery_rate': 'mean',
    'recovered': lambda x: (x.sum() / len(x)) * 100
}).reset_index()
age_group_stats.columns = ['age_group', 'avg_recovery_days', 'avg_recovery_rate', 'recovery_percentage']

x_pos = np.arange(len(age_group_stats))
width = 0.25
axes[1, 1].bar(x_pos - width, age_group_stats['avg_recovery_days'], width,
              label='Avg Recovery Days', color='#e74c3c', alpha=0.8, edgecolor='black')
ax_twin = axes[1, 1].twinx()
ax_twin.bar(x_pos, age_group_stats['avg_recovery_rate'], width,
           label='Avg Recovery Rate (%)', color='#2ecc71', alpha=0.8, edgecolor='black')
ax_twin.bar(x_pos + width, age_group_stats['recovery_percentage'], width,
           label='Recovery %', color='#3498db', alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Age Group', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Average Recovery Days', fontsize=12, fontweight='bold', color='#e74c3c')
ax_twin.set_ylabel('Recovery Metrics (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Recovery Metrics by Age Group', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(age_group_stats['age_group'])
axes[1, 1].legend(loc='upper left', fontsize=9)
ax_twin.legend(loc='upper right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Age Distribution & Demographics Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}2_age_demographics_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 2_age_demographics_analysis.png")

# ============================================================================
# 3. DISEASE ANALYSIS & PREVALENCE
# ============================================================================
print("\n3. Generating Disease Analysis & Prevalence Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Disease patient count
disease_counts = df['disease'].value_counts().sort_values(ascending=True)
axes[0, 0].barh(range(len(disease_counts)), disease_counts.values,
               color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(disease_counts))),
               alpha=0.8, edgecolor='black')
axes[0, 0].set_yticks(range(len(disease_counts)))
axes[0, 0].set_yticklabels(disease_counts.index, fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Number of Patients', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Disease Prevalence - Patient Count', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(disease_counts.values):
    axes[0, 0].text(v, i, str(v), va='center', fontweight='bold')

# Disease by age group heatmap
disease_age = df.groupby(['disease', 'age_group']).size().unstack(fill_value=0)
sns.heatmap(disease_age, annot=True, fmt='d', cmap='YlOrRd', 
           linewidths=1, cbar_kws={'label': 'Patient Count'}, 
           ax=axes[0, 1], square=False)
axes[0, 1].set_xlabel('Age Group', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Disease', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Disease Prevalence by Age Group', fontsize=14, fontweight='bold')

# Average recovery days by disease
disease_recovery = df.groupby('disease').agg({
    'recovery_days': 'mean',
    'recovery_rate': 'mean',
    'recovered': lambda x: (x.sum() / len(x)) * 100
}).sort_values('recovery_days', ascending=True)
disease_recovery.columns = ['avg_recovery_days', 'avg_recovery_rate', 'recovery_percentage']

x_pos = np.arange(len(disease_recovery))
width = 0.25
axes[1, 0].bar(x_pos - width, disease_recovery['avg_recovery_days'], width,
              label='Avg Recovery Days', color='#e74c3c', alpha=0.8, edgecolor='black')
ax_twin = axes[1, 0].twinx()
ax_twin.bar(x_pos, disease_recovery['avg_recovery_rate'], width,
           label='Avg Recovery Rate (%)', color='#2ecc71', alpha=0.8, edgecolor='black')
ax_twin.bar(x_pos + width, disease_recovery['recovery_percentage'], width,
           label='Recovery %', color='#3498db', alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Disease', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Average Recovery Days', fontsize=12, fontweight='bold', color='#e74c3c')
ax_twin.set_ylabel('Recovery Metrics (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Recovery Metrics by Disease', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(disease_recovery.index, rotation=45, ha='right')
axes[1, 0].legend(loc='upper left', fontsize=9)
ax_twin.legend(loc='upper right', fontsize=9)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Recovery days distribution by disease
disease_list = df['disease'].unique()
recovery_data = [df[df['disease'] == disease]['recovery_days'].values 
                for disease in disease_list]
bp = axes[1, 1].boxplot(recovery_data, labels=disease_list, patch_artist=True, vert=True)
colors_box = plt.cm.Set3(np.linspace(0, 1, len(disease_list)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_xlabel('Disease', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Recovery Days', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Recovery Days Distribution by Disease', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Disease Analysis & Prevalence Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}3_disease_analysis_dashboard.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 3_disease_analysis_dashboard.png")

# ============================================================================
# 4. RECOVERY RATE COMPREHENSIVE ANALYSIS
# ============================================================================
print("\n4. Generating Recovery Rate Comprehensive Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Recovery rate distribution
axes[0, 0].hist(df['recovery_rate'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(df['recovery_rate'].mean(), color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {df["recovery_rate"].mean():.1f}%')
axes[0, 0].axvline(df['recovery_rate'].median(), color='blue', linestyle='--', linewidth=2.5,
                  label=f'Median: {df["recovery_rate"].median():.1f}%')
axes[0, 0].set_xlabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Recovery Rate Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Recovery rate by disease
recovery_by_disease = df.groupby('disease').agg({
    'recovery_rate': 'mean',
    'recovered': lambda x: (x.sum() / len(x)) * 100
}).sort_values('recovery_rate', ascending=True)
recovery_by_disease.columns = ['avg_recovery_rate', 'recovery_percentage']

x_pos = np.arange(len(recovery_by_disease))
axes[0, 1].barh(x_pos, recovery_by_disease['avg_recovery_rate'],
               color=plt.cm.RdYlGn(recovery_by_disease['avg_recovery_rate']/100),
               alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(x_pos)
axes[0, 1].set_yticklabels(recovery_by_disease.index, fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Average Recovery Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Average Recovery Rate by Disease', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(recovery_by_disease['avg_recovery_rate']):
    axes[0, 1].text(v, i, f'{v:.1f}%', va='center', fontweight='bold')

# Recovery rate vs Recovery days
recovered_data = df[df['recovered'] == True]
not_recovered_data = df[df['recovered'] == False]

axes[1, 0].scatter(recovered_data['recovery_days'], recovered_data['recovery_rate'],
                  alpha=0.6, s=100, color='#2ecc71', label='Recovered', 
                  edgecolors='black', linewidth=1)
axes[1, 0].scatter(not_recovered_data['recovery_days'], not_recovered_data['recovery_rate'],
                  alpha=0.6, s=100, color='#e74c3c', label='Not Recovered',
                  edgecolors='black', linewidth=1)
axes[1, 0].set_xlabel('Recovery Days', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Recovery Days vs Recovery Rate', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Recovery rate by age group
age_recovery = df.groupby('age_group').agg({
    'recovery_rate': 'mean',
    'recovered': lambda x: (x.sum() / len(x)) * 100,
    'recovery_days': 'mean'
}).reset_index()
age_recovery.columns = ['age_group', 'avg_recovery_rate', 'recovery_percentage', 'avg_recovery_days']

x_pos = np.arange(len(age_recovery))
width = 0.25
axes[1, 1].bar(x_pos - width, age_recovery['avg_recovery_rate'], width,
              label='Avg Recovery Rate (%)', color='#2ecc71', alpha=0.8, edgecolor='black')
axes[1, 1].bar(x_pos, age_recovery['recovery_percentage'], width,
              label='Recovery Percentage', color='#3498db', alpha=0.8, edgecolor='black')
ax_twin = axes[1, 1].twinx()
ax_twin.bar(x_pos + width, age_recovery['avg_recovery_days'], width,
           label='Avg Recovery Days', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Age Group', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
ax_twin.set_ylabel('Recovery Days', fontsize=12, fontweight='bold', color='#e74c3c')
axes[1, 1].set_title('Recovery Metrics by Age Group', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(age_recovery['age_group'])
axes[1, 1].legend(loc='upper left', fontsize=9)
ax_twin.legend(loc='upper right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Recovery Rate Comprehensive Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}4_recovery_rate_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 4_recovery_rate_analysis.png")

# ============================================================================
# 5. TEMPORAL PATTERNS & SEASONALITY
# ============================================================================
print("\n5. Generating Temporal Patterns & Seasonality Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Monthly recovery trends
monthly_recovery = df.groupby('year_month').agg({
    'recovered': lambda x: (x.sum() / len(x)) * 100,
    'recovery_rate': 'mean',
    'recovery_days': 'mean'
}).reset_index()
monthly_recovery['year_month'] = monthly_recovery['year_month'].astype(str)

x_month = range(len(monthly_recovery))
ax_twin = axes[0, 0].twinx()
line1 = axes[0, 0].plot(x_month, monthly_recovery['recovered'],
                       marker='o', linewidth=2.5, label='Recovery %', 
                       color='#2ecc71', markersize=8)
line2 = ax_twin.plot(x_month, monthly_recovery['recovery_days'],
                    marker='s', linewidth=2.5, label='Avg Recovery Days', 
                    color='#e74c3c', markersize=8)
axes[0, 0].fill_between(x_month, monthly_recovery['recovered'], 
                       alpha=0.3, color='#2ecc71')
axes[0, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Recovery Percentage (%)', fontsize=12, fontweight='bold', color='#2ecc71')
ax_twin.set_ylabel('Average Recovery Days', fontsize=12, fontweight='bold', color='#e74c3c')
axes[0, 0].set_title('Monthly Recovery Trends', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(range(0, len(monthly_recovery), 2))
axes[0, 0].set_xticklabels([monthly_recovery['year_month'].iloc[i] 
                            for i in range(0, len(monthly_recovery), 2)], 
                           rotation=45, ha='right')
axes[0, 0].grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
axes[0, 0].legend(lines, labels, loc='best', fontsize=10)

# Disease trends over time
top_3_diseases = df['disease'].value_counts().head(3).index
for disease in top_3_diseases:
    disease_monthly = df[df['disease'] == disease].groupby('year_month').size()
    disease_monthly.index = disease_monthly.index.astype(str)
    axes[0, 1].plot(range(len(disease_monthly)), disease_monthly.values,
                   marker='o', linewidth=2, label=disease, markersize=6)
axes[0, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Top 3 Diseases - Monthly Trends', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Recovery days by month
monthly_recovery_days = df.groupby('month')['recovery_days'].mean()
axes[1, 0].bar(monthly_recovery_days.index, monthly_recovery_days.values,
              color=plt.cm.coolwarm(np.linspace(0, 1, len(monthly_recovery_days))),
              alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Average Recovery Days', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Average Recovery Days by Month', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(monthly_recovery_days.index)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[1, 0].set_xticklabels([month_names[i-1] for i in monthly_recovery_days.index])
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(monthly_recovery_days.values):
    axes[1, 0].text(monthly_recovery_days.index[i], v, f'{v:.1f}', 
                   ha='center', va='bottom', fontweight='bold')

# Recovery rate heatmap by disease and month
disease_month_recovery = df.groupby(['disease', 'month'])['recovery_rate'].mean().unstack(fill_value=0)
sns.heatmap(disease_month_recovery, annot=True, fmt='.1f', cmap='RdYlGn', 
           linewidths=1, cbar_kws={'label': 'Recovery Rate (%)'}, 
           ax=axes[1, 1], square=False)
axes[1, 1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Disease', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Recovery Rate Heatmap: Disease × Month', fontsize=14, fontweight='bold')
axes[1, 1].set_xticklabels(month_names, rotation=45, ha='right')

plt.suptitle('Temporal Patterns & Seasonality Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}5_temporal_patterns_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 5_temporal_patterns_analysis.png")

# ============================================================================
# 6. COMPREHENSIVE STATISTICAL SUMMARY DASHBOARD
# ============================================================================
print("\n6. Generating Comprehensive Statistical Summary Dashboard...")
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# Key Statistics Summary
ax1 = fig.add_subplot(gs[0, 0])
stats_summary = {
    'Total Patients': f"{len(df):,}",
    'Recovery Rate': f"{df['recovered'].sum()/len(df)*100:.1f}%",
    'Avg Recovery Days': f"{df['recovery_days'].mean():.1f}",
    'Avg Age': f"{df['age'].mean():.1f} years",
    'Total Diseases': f"{df['disease'].nunique()}",
    'Avg Recovery Rate': f"{df['recovery_rate'].mean():.1f}%"
}
y_pos = np.arange(len(stats_summary))
ax1.barh(y_pos, [1]*len(stats_summary), 
        color=plt.cm.viridis(np.linspace(0, 1, len(stats_summary))))
ax1.set_yticks(y_pos)
ax1.set_yticklabels(list(stats_summary.keys()), fontsize=11, fontweight='bold')
ax1.set_xticks([])
ax1.set_title('Key Healthcare Statistics', fontsize=13, fontweight='bold')
for i, (key, val) in enumerate(stats_summary.items()):
    ax1.text(0.5, i, val, ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Top Diseases
ax2 = fig.add_subplot(gs[0, 1])
top_diseases = df['disease'].value_counts().head(5).sort_values(ascending=True)
ax2.barh(range(len(top_diseases)), top_diseases.values,
        color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_diseases))),
        alpha=0.8, edgecolor='black')
ax2.set_yticks(range(len(top_diseases)))
ax2.set_yticklabels(top_diseases.index, fontsize=11, fontweight='bold')
ax2.set_xlabel('Patient Count', fontsize=11, fontweight='bold')
ax2.set_title('Top 5 Diseases by Patient Count', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_diseases.values):
    ax2.text(v, i, str(v), va='center', fontweight='bold', fontsize=10)

# Recovery Days Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df['recovery_days'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
ax3.axvline(df['recovery_days'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["recovery_days"].mean():.1f} days')
ax3.set_xlabel('Recovery Days', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Recovery Days Distribution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Recovery Rate Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df['recovery_rate'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax4.axvline(df['recovery_rate'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["recovery_rate"].mean():.1f}%')
ax4.set_xlabel('Recovery Rate (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Recovery Rate Distribution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Age Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df['age'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
ax5.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["age"].mean():.1f} years')
ax5.set_xlabel('Age', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Age Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Recovery Status Pie Chart
ax6 = fig.add_subplot(gs[1, 2])
recovery_status = df['recovered'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax6.pie(recovery_status.values, 
                                   labels=['Recovered', 'Not Recovered'], 
                                   autopct='%1.1f%%',
                                   startangle=90, colors=colors_pie, 
                                   textprops={'fontsize': 11, 'fontweight': 'bold'})
ax6.set_title('Recovery Status Distribution', fontsize=13, fontweight='bold')

# Quarterly Performance Comparison
ax7 = fig.add_subplot(gs[2, :])
quarterly_comp = df.groupby('quarter').agg({
    'recovered': lambda x: (x.sum() / len(x)) * 100,
    'recovery_rate': 'mean',
    'recovery_days': 'mean'
}).reset_index()
x_q = range(len(quarterly_comp))
width = 0.25
ax7.bar([x - width for x in x_q], quarterly_comp['recovered'], width,
        label='Recovery %', color='#2ecc71', alpha=0.8, edgecolor='black')
ax7.bar(x_q, quarterly_comp['recovery_rate'], width,
        label='Avg Recovery Rate (%)', color='#3498db', alpha=0.8, edgecolor='black')
ax7_twin = ax7.twinx()
ax7_twin.bar([x + width for x in x_q], quarterly_comp['recovery_days'], width,
            label='Avg Recovery Days', color='#e74c3c', alpha=0.8, edgecolor='black')
ax7.set_xlabel('Quarter', fontsize=12, fontweight='bold')
ax7.set_ylabel('Recovery Metrics (%)', fontsize=12, fontweight='bold')
ax7_twin.set_ylabel('Recovery Days', fontsize=12, fontweight='bold', color='#e74c3c')
ax7.set_title('Quarterly Performance Comparison', fontsize=14, fontweight='bold')
ax7.set_xticks(x_q)
ax7.set_xticklabels([f'Q{i}' for i in quarterly_comp['quarter']])
ax7.legend(loc='upper left', fontsize=10)
ax7_twin.legend(loc='upper right', fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle('Healthcare Insights - Comprehensive Statistical Summary Dashboard', 
             fontsize=20, fontweight='bold', y=0.995)
plt.savefig(f'{EXPORT_PATH}6_comprehensive_statistical_summary.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 6_comprehensive_statistical_summary.png")

print("\n" + "="*60)
print("ALL HEALTHCARE VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nTotal visualizations created: 6")
print(f"Export directory: {EXPORT_PATH}")
print("\nGenerated files:")
print("  1. 1_comprehensive_admission_trends.png")
print("  2. 2_age_demographics_analysis.png")
print("  3. 3_disease_analysis_dashboard.png")
print("  4. 4_recovery_rate_analysis.png")
print("  5. 5_temporal_patterns_analysis.png")
print("  6. 6_comprehensive_statistical_summary.png")
print("\nAll visualizations are high-resolution (300 DPI) and ready for presentation!")

