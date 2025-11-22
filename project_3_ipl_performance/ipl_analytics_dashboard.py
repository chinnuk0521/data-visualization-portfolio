"""
IPL Player Performance Analytics Dashboard
Comprehensive visualization suite with advanced cricket performance analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy import stats
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
print("Loading IPL dataset...")
df = pd.read_csv('dataset_ipl.csv')

print(f"Dataset loaded: {len(df)} records")
print(f"Players: {df['player'].nunique()}")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"Date range: {df['season'].min()} - {df['season'].max()}")

# Calculate additional metrics
df['runs_per_match'] = df['runs'] / df['matches']
df['boundary_percentage'] = (df['boundaries'] / df['runs'] * 100).fillna(0)
df['sixes_per_match'] = df['sixes'] / df['matches']
df['fours_per_match'] = df['fours'] / df['matches']

# ============================================================================
# 1. PLAYER PERFORMANCE TREND - MULTI-PLAYER COMPARISON
# ============================================================================
print("\n1. Generating Player Performance Trends...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Runs over seasons - Top players
top_players = df.groupby('player')['runs'].sum().sort_values(ascending=False).head(6).index
for player in top_players:
    player_data = df[df['player'] == player].sort_values('season')
    axes[0, 0].plot(player_data['season'], player_data['runs'], 
                    marker='o', linewidth=2.5, label=player, markersize=8)
axes[0, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Total Runs', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Top Players - Runs Trend Across Seasons', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='best', fontsize=9, ncol=2)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(sorted(df['season'].unique()))

# Strike Rate vs Batting Average
scatter = axes[0, 1].scatter(df['strike_rate'], df['batting_avg'],
                             s=df['runs']/5, alpha=0.6,
                             c=df['season'], cmap='viridis',
                             edgecolors='black', linewidth=1.5)
cbar = plt.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('Season', fontsize=11, fontweight='bold')
for player in df['player'].unique():
    player_data = df[df['player'] == player]
    axes[0, 1].annotate(player.split()[-1], 
                        (player_data['strike_rate'].mean(), 
                         player_data['batting_avg'].mean()),
                       fontsize=8, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
axes[0, 1].set_xlabel('Strike Rate', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Batting Average', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Strike Rate vs Batting Average\n(Bubble size = Runs, Color = Season)', 
                     fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Boundaries Analysis
boundary_data = df.groupby('player').agg({
    'fours': 'sum',
    'sixes': 'sum',
    'boundaries': 'sum'
}).sort_values('boundaries', ascending=False).head(10)
x_pos = np.arange(len(boundary_data))
width = 0.25
axes[1, 0].bar(x_pos - width, boundary_data['fours'], width,
               label='Fours', color='#3498db', alpha=0.8, edgecolor='black')
axes[1, 0].bar(x_pos, boundary_data['sixes'], width,
               label='Sixes', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1, 0].bar(x_pos + width, boundary_data['boundaries'], width,
               label='Total Boundaries', color='#2ecc71', alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Player', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Top 10 Players - Boundary Hitting Analysis', 
                     fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(boundary_data.index, rotation=45, ha='right')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Milestones (Fifties + Centuries)
milestone_data = df.groupby('player').agg({
    'fifties': 'sum',
    'centuries': 'sum'
}).sort_values(['centuries', 'fifties'], ascending=False).head(10)
milestone_data['total_milestones'] = milestone_data['fifties'] + milestone_data['centuries']
axes[1, 1].barh(range(len(milestone_data)), milestone_data['fifties'],
                label='Fifties', color='#f39c12', alpha=0.8, edgecolor='black')
axes[1, 1].barh(range(len(milestone_data)), milestone_data['centuries'],
                left=milestone_data['fifties'], label='Centuries', 
                color='#9b59b6', alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(milestone_data)))
axes[1, 1].set_yticklabels(milestone_data.index, fontsize=10, fontweight='bold')
axes[1, 1].set_xlabel('Count', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Top 10 Players - Milestone Achievements', 
                     fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.suptitle('IPL Player Performance Analysis Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}1_player_performance_dashboard.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 1_player_performance_dashboard.png")

# ============================================================================
# 2. SEASON-WISE PERFORMANCE COMPARISON
# ============================================================================
print("\n2. Generating Season-wise Performance Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Average runs per season
season_stats = df.groupby('season').agg({
    'runs': 'mean',
    'strike_rate': 'mean',
    'batting_avg': 'mean',
    'boundaries': 'mean'
}).reset_index()

ax_twin = axes[0, 0].twinx()
line1 = axes[0, 0].plot(season_stats['season'], season_stats['runs'],
                       marker='o', linewidth=3, label='Avg Runs', 
                       color='#2ecc71', markersize=10)
line2 = ax_twin.plot(season_stats['season'], season_stats['strike_rate'],
                    marker='s', linewidth=3, label='Avg Strike Rate', 
                    color='#e74c3c', markersize=10)
axes[0, 0].fill_between(season_stats['season'], season_stats['runs'],
                       alpha=0.3, color='#2ecc71')
axes[0, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Average Runs', fontsize=12, fontweight='bold', color='#2ecc71')
ax_twin.set_ylabel('Average Strike Rate', fontsize=12, fontweight='bold', color='#e74c3c')
axes[0, 0].set_title('Season-wise Average Performance Metrics', 
                     fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(season_stats['season'])
axes[0, 0].grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
axes[0, 0].legend(lines, labels, loc='upper left', fontsize=10)

# Top run scorer by season
top_scorers = df.loc[df.groupby('season')['runs'].idxmax()]
axes[0, 1].barh(range(len(top_scorers)), top_scorers['runs'],
               color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_scorers))),
               alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(top_scorers)))
axes[0, 1].set_yticklabels([f"{row['player']} ({int(row['season'])})" 
                            for _, row in top_scorers.iterrows()], 
                           fontsize=10, fontweight='bold')
axes[0, 1].set_xlabel('Total Runs', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Top Run Scorer by Season', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_scorers['runs']):
    axes[0, 1].text(v, i, f'{int(v)}', va='center', fontweight='bold', fontsize=10)

# Strike Rate distribution by season
season_list = sorted(df['season'].unique())
strike_rate_data = [df[df['season'] == season]['strike_rate'].values 
                   for season in season_list]
bp = axes[1, 0].boxplot(strike_rate_data, labels=season_list, patch_artist=True)
colors_box = plt.cm.Set3(np.linspace(0, 1, len(season_list)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Strike Rate', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Strike Rate Distribution by Season', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Batting Average by season
batting_avg_data = [df[df['season'] == season]['batting_avg'].values 
                    for season in season_list]
bp = axes[1, 1].boxplot(batting_avg_data, labels=season_list, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Batting Average', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Batting Average Distribution by Season', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('IPL Season-wise Performance Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}2_season_wise_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 2_season_wise_analysis.png")

# ============================================================================
# 3. PLAYER COMPARISON RADAR CHART
# ============================================================================
print("\n3. Generating Player Comparison Radar Chart...")
# Select top 5 players by total runs
top_5_players = df.groupby('player')['runs'].sum().sort_values(ascending=False).head(5).index

player_avg = df.groupby('player').agg({
    'runs': 'mean',
    'strike_rate': 'mean',
    'batting_avg': 'mean',
    'boundaries': 'mean',
    'fifties': 'mean',
    'centuries': 'mean'
}).loc[top_5_players]

# Normalize to 0-100 scale
metrics = ['runs', 'strike_rate', 'batting_avg', 'boundaries', 'fifties', 'centuries']
player_normalized = player_avg.copy()
for metric in metrics:
    player_normalized[metric] = ((player_avg[metric] - player_avg[metric].min()) / 
                                 (player_avg[metric].max() - player_avg[metric].min())) * 100

fig, ax = plt.subplots(figsize=(16, 12), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

colors = plt.cm.Set3(np.linspace(0, 1, len(top_5_players)))

for idx, player in enumerate(top_5_players):
    values = player_normalized.loc[player, metrics].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2.5, label=player, 
           color=colors[idx], markersize=8)
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Runs', 'Strike Rate', 'Batting Avg', 
                   'Boundaries', 'Fifties', 'Centuries'], 
                  fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Top 5 Players - Multi-Dimensional Performance Comparison\n(Normalized Metrics)', 
             fontsize=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.savefig(f'{EXPORT_PATH}3_player_radar_chart.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 3_player_radar_chart.png")

# ============================================================================
# 4. BOUNDARY HITTING ANALYSIS
# ============================================================================
print("\n4. Generating Boundary Hitting Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Fours vs Sixes scatter
scatter = axes[0, 0].scatter(df['fours'], df['sixes'],
                            s=df['runs']/5, alpha=0.6,
                            c=df['strike_rate'], cmap='plasma',
                            edgecolors='black', linewidth=1.5)
cbar = plt.colorbar(scatter, ax=axes[0, 0])
cbar.set_label('Strike Rate', fontsize=11, fontweight='bold')
for player in df['player'].unique():
    player_data = df[df['player'] == player]
    axes[0, 0].annotate(player.split()[-1], 
                       (player_data['fours'].mean(), player_data['sixes'].mean()),
                       fontsize=8, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
axes[0, 0].set_xlabel('Fours', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Sixes', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Fours vs Sixes Analysis\n(Bubble = Runs, Color = Strike Rate)', 
                     fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Boundary percentage by player
boundary_pct = df.groupby('player').agg({
    'boundary_percentage': 'mean',
    'boundaries': 'sum'
}).sort_values('boundary_percentage', ascending=False).head(10)
axes[0, 1].barh(range(len(boundary_pct)), boundary_pct['boundary_percentage'],
               color=plt.cm.viridis(np.linspace(0, 1, len(boundary_pct))),
               alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(boundary_pct)))
axes[0, 1].set_yticklabels(boundary_pct.index, fontsize=10, fontweight='bold')
axes[0, 1].set_xlabel('Boundary Percentage (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Top 10 Players - Boundary Percentage', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(boundary_pct['boundary_percentage']):
    axes[0, 1].text(v, i, f'{v:.1f}%', va='center', fontweight='bold')

# Sixes per match trend
top_6_players = df.groupby('player')['sixes'].sum().sort_values(ascending=False).head(6).index
for player in top_6_players:
    player_data = df[df['player'] == player].sort_values('season')
    axes[1, 0].plot(player_data['season'], player_data['sixes_per_match'],
                    marker='o', linewidth=2.5, label=player, markersize=7)
axes[1, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Sixes per Match', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Top Players - Sixes per Match Trend', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc='best', fontsize=9, ncol=2)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(sorted(df['season'].unique()))

# Fours per match trend
for player in top_6_players:
    player_data = df[df['player'] == player].sort_values('season')
    axes[1, 1].plot(player_data['season'], player_data['fours_per_match'],
                    marker='s', linewidth=2.5, label=player, markersize=7)
axes[1, 1].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Fours per Match', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Top Players - Fours per Match Trend', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=9, ncol=2)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(sorted(df['season'].unique()))

plt.suptitle('IPL Boundary Hitting Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}4_boundary_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 4_boundary_analysis.png")

# ============================================================================
# 5. CONSISTENCY & RELIABILITY ANALYSIS
# ============================================================================
print("\n5. Generating Consistency & Reliability Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Runs per match consistency
player_consistency = df.groupby('player').agg({
    'runs_per_match': ['mean', 'std'],
    'runs': 'sum'
}).reset_index()
player_consistency.columns = ['player', 'avg_runs_per_match', 'std_runs_per_match', 'total_runs']
player_consistency['consistency'] = 1 / (1 + player_consistency['std_runs_per_match'])
player_consistency = player_consistency.sort_values('total_runs', ascending=False).head(10)

scatter = axes[0, 0].scatter(player_consistency['avg_runs_per_match'], 
                            player_consistency['consistency'],
                           s=player_consistency['total_runs']/3, alpha=0.7,
                           c=range(len(player_consistency)), cmap='coolwarm',
                           edgecolors='black', linewidth=2)
for i, row in player_consistency.iterrows():
    axes[0, 0].annotate(row['player'].split()[-1], 
                       (row['avg_runs_per_match'], row['consistency']),
                       fontsize=9, ha='center', va='center', fontweight='bold')
axes[0, 0].set_xlabel('Average Runs per Match', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Consistency Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Player Consistency Analysis\n(Bubble size = Total Runs)', 
                     fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Strike Rate consistency
strike_rate_consistency = df.groupby('player').agg({
    'strike_rate': ['mean', 'std'],
    'runs': 'sum'
}).reset_index()
strike_rate_consistency.columns = ['player', 'avg_sr', 'std_sr', 'total_runs']
strike_rate_consistency['sr_consistency'] = 1 / (1 + strike_rate_consistency['std_sr'])
strike_rate_consistency = strike_rate_consistency.sort_values('total_runs', ascending=False).head(10)

axes[0, 1].barh(range(len(strike_rate_consistency)), 
               strike_rate_consistency['sr_consistency'],
               color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(strike_rate_consistency))),
               alpha=0.8, edgecolor='black')
axes[0, 1].set_yticks(range(len(strike_rate_consistency)))
axes[0, 1].set_yticklabels(strike_rate_consistency['player'], 
                           fontsize=10, fontweight='bold')
axes[0, 1].set_xlabel('Strike Rate Consistency Score', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Top 10 Players - Strike Rate Consistency', 
                     fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Performance by matches played
matches_performance = df.groupby('player').agg({
    'matches': 'sum',
    'runs': 'sum',
    'runs_per_match': 'mean'
}).sort_values('matches', ascending=False).head(10)

ax_twin = axes[1, 0].twinx()
x_pos = np.arange(len(matches_performance))
width = 0.35
bars1 = axes[1, 0].bar(x_pos - width/2, matches_performance['matches'], width,
                       label='Total Matches', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax_twin.bar(x_pos + width/2, matches_performance['runs_per_match'], width,
                    label='Runs per Match', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Player', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Total Matches', fontsize=12, fontweight='bold', color='#3498db')
ax_twin.set_ylabel('Runs per Match', fontsize=12, fontweight='bold', color='#e74c3c')
axes[1, 0].set_title('Matches Played vs Performance', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(matches_performance.index, rotation=45, ha='right')
axes[1, 0].legend(loc='upper left', fontsize=10)
ax_twin.legend(loc='upper right', fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Milestone frequency
milestone_freq = df.groupby('player').agg({
    'fifties': 'sum',
    'centuries': 'sum',
    'matches': 'sum'
}).reset_index()
milestone_freq['fifty_freq'] = milestone_freq['fifties'] / milestone_freq['matches']
milestone_freq['century_freq'] = milestone_freq['centuries'] / milestone_freq['matches']
milestone_freq = milestone_freq.sort_values('fifty_freq', ascending=False).head(10)

x_pos = np.arange(len(milestone_freq))
width = 0.35
axes[1, 1].bar(x_pos - width/2, milestone_freq['fifty_freq'], width,
               label='Fifties per Match', color='#f39c12', alpha=0.8, edgecolor='black')
axes[1, 1].bar(x_pos + width/2, milestone_freq['century_freq'], width,
               label='Centuries per Match', color='#9b59b6', alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Player', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Milestone Achievement Frequency', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(milestone_freq['player'], rotation=45, ha='right')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('IPL Player Consistency & Reliability Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{EXPORT_PATH}5_consistency_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 5_consistency_analysis.png")

# ============================================================================
# 6. COMPREHENSIVE STATISTICAL SUMMARY
# ============================================================================
print("\n6. Generating Comprehensive Statistical Summary...")
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# Key Statistics Summary
ax1 = fig.add_subplot(gs[0, 0])
stats_summary = {
    'Total Runs': f"{df['runs'].sum():,}",
    'Total Matches': f"{df['matches'].sum():,}",
    'Avg Strike Rate': f"{df['strike_rate'].mean():.1f}",
    'Avg Batting Avg': f"{df['batting_avg'].mean():.1f}",
    'Total Boundaries': f"{df['boundaries'].sum():,}",
    'Total Centuries': f"{int(df['centuries'].sum())}"
}
y_pos = np.arange(len(stats_summary))
ax1.barh(y_pos, [1]*len(stats_summary), 
        color=plt.cm.viridis(np.linspace(0, 1, len(stats_summary))))
ax1.set_yticks(y_pos)
ax1.set_yticklabels(list(stats_summary.keys()), fontsize=11, fontweight='bold')
ax1.set_xticks([])
ax1.set_title('Key Performance Statistics', fontsize=13, fontweight='bold')
for i, (key, val) in enumerate(stats_summary.items()):
    ax1.text(0.5, i, val, ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Top 5 Players by Total Runs
ax2 = fig.add_subplot(gs[0, 1])
top_runs = df.groupby('player')['runs'].sum().sort_values(ascending=True).tail(5)
ax2.barh(range(len(top_runs)), top_runs.values,
        color=plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_runs))),
        alpha=0.8, edgecolor='black')
ax2.set_yticks(range(len(top_runs)))
ax2.set_yticklabels(top_runs.index, fontsize=11, fontweight='bold')
ax2.set_xlabel('Total Runs', fontsize=11, fontweight='bold')
ax2.set_title('Top 5 Run Scorers (All Seasons)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_runs.values):
    ax2.text(v, i, f'{int(v)}', va='center', fontweight='bold', fontsize=10)

# Runs Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df['runs'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax3.axvline(df['runs'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["runs"].mean():.0f}')
ax3.axvline(df['runs'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {df["runs"].median():.0f}')
ax3.set_xlabel('Runs', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Runs Distribution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Strike Rate Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df['strike_rate'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
ax4.axvline(df['strike_rate'].mean(), color='blue', linestyle='--', linewidth=2,
           label=f'Mean: {df["strike_rate"].mean():.1f}')
ax4.set_xlabel('Strike Rate', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Strike Rate Distribution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Batting Average Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df['batting_avg'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax5.axvline(df['batting_avg'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["batting_avg"].mean():.1f}')
ax5.set_xlabel('Batting Average', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Batting Average Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Boundaries Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(df['boundaries'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
ax6.axvline(df['boundaries'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["boundaries"].mean():.0f}')
ax6.set_xlabel('Boundaries', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Boundaries Distribution', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# Season Performance Comparison
ax7 = fig.add_subplot(gs[2, :])
season_comp = df.groupby('season').agg({
    'runs': 'sum',
    'boundaries': 'sum',
    'centuries': 'sum'
}).reset_index()
x_season = range(len(season_comp))
width = 0.25
ax7.bar([x - width for x in x_season], season_comp['runs']/1000, width,
        label='Total Runs (K)', color='#3498db', alpha=0.8, edgecolor='black')
ax7.bar(x_season, season_comp['boundaries']/100, width,
        label='Total Boundaries (×100)', color='#2ecc71', alpha=0.8, edgecolor='black')
ax7_twin = ax7.twinx()
ax7_twin.plot(x_season, season_comp['centuries'], 
              marker='o', linewidth=3, label='Total Centuries', 
              color='#e74c3c', markersize=10)
ax7.set_xlabel('Season', fontsize=12, fontweight='bold')
ax7.set_ylabel('Runs & Boundaries (Normalized)', fontsize=12, fontweight='bold')
ax7_twin.set_ylabel('Centuries', fontsize=12, fontweight='bold', color='#e74c3c')
ax7.set_title('Season Performance Comparison', fontsize=14, fontweight='bold')
ax7.set_xticks(x_season)
ax7.set_xticklabels(season_comp['season'])
ax7.legend(loc='upper left', fontsize=10)
ax7_twin.legend(loc='upper right', fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle('IPL Performance - Comprehensive Statistical Summary Dashboard', 
             fontsize=20, fontweight='bold', y=0.995)
plt.savefig(f'{EXPORT_PATH}6_comprehensive_statistical_summary.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("   [OK] Saved: 6_comprehensive_statistical_summary.png")

print("\n" + "="*60)
print("ALL IPL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nTotal visualizations created: 6")
print(f"Export directory: {EXPORT_PATH}")
print("\nGenerated files:")
print("  1. 1_player_performance_dashboard.png")
print("  2. 2_season_wise_analysis.png")
print("  3. 3_player_radar_chart.png")
print("  4. 4_boundary_analysis.png")
print("  5. 5_consistency_analysis.png")
print("  6. 6_comprehensive_statistical_summary.png")
print("\nAll visualizations are high-resolution (300 DPI) and ready for presentation!")

