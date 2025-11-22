# IPL Player Performance Analysis

A comprehensive analysis of Indian Premier League (IPL) player performance metrics across multiple seasons.

## Overview

This project analyzes IPL player statistics to provide insights into batting performance, consistency, and trends across different seasons. The analysis helps identify top performers, track player progress, and understand performance patterns.

## Dataset

- **File:** `dataset_ipl.csv`
- **Description:** Contains IPL player performance data with the following metrics:
  - `player`: Player name
  - `season`: IPL season year
  - `runs`: Total runs scored
  - `matches`: Number of matches played
  - `strike_rate`: Batting strike rate
  - `batting_avg`: Batting average
  - `fours`: Number of fours hit
  - `sixes`: Number of sixes hit
  - `fifties`: Number of half-centuries
  - `centuries`: Number of centuries
  - `boundaries`: Total boundaries (fours + sixes)

## Analysis Areas

- **Performance Trends:** Track player performance across seasons
- **Batting Statistics:** Analyze runs, strike rate, and batting average
- **Boundary Analysis:** Study fours, sixes, and boundary hitting patterns
- **Milestone Tracking:** Monitor fifties and centuries
- **Season Comparisons:** Compare performance across different IPL seasons

## Visualizations

This project includes **6 comprehensive, advanced visualizations** providing multi-dimensional insights into IPL player performance.

### 1. Player Performance Dashboard
- **File:** `exports/1_player_performance_dashboard.png`
- **Description:** Multi-panel dashboard featuring:
  - Top players runs trend across seasons (2019-2023)
  - Strike rate vs batting average scatter plot (bubble size = runs, color = season)
  - Top 10 players boundary hitting analysis (fours, sixes, total boundaries)
  - Top 10 players milestone achievements (fifties and centuries stacked bar chart)

### 2. Season-wise Performance Analysis
- **File:** `exports/2_season_wise_analysis.png`
- **Description:** Comprehensive season comparison including:
  - Average runs and strike rate trends across seasons
  - Top run scorer by season (horizontal bar chart)
  - Strike rate distribution box plots by season
  - Batting average distribution box plots by season

### 3. Player Comparison Radar Chart
- **File:** `exports/3_player_radar_chart.png`
- **Description:** Multi-dimensional radar/spider chart comparing top 5 players across 6 key metrics:
  - Runs, Strike Rate, Batting Average, Boundaries, Fifties, Centuries
  - Normalized to 0-100 scale for fair comparison
  - Identifies all-round performance strengths

### 4. Boundary Hitting Analysis
- **File:** `exports/4_boundary_analysis.png`
- **Description:** Deep dive into boundary statistics:
  - Fours vs Sixes scatter plot (bubble = runs, color = strike rate)
  - Top 10 players boundary percentage analysis
  - Sixes per match trend for top players
  - Fours per match trend for top players

### 5. Consistency & Reliability Analysis
- **File:** `exports/5_consistency_analysis.png`
- **Description:** Player reliability metrics including:
  - Runs per match consistency analysis (scatter plot)
  - Strike rate consistency scores (horizontal bar chart)
  - Matches played vs performance comparison
  - Milestone achievement frequency (fifties and centuries per match)

### 6. Comprehensive Statistical Summary
- **File:** `exports/6_comprehensive_statistical_summary.png`
- **Description:** Executive summary dashboard with:
  - Key performance statistics (KPI cards)
  - Top 5 run scorers across all seasons
  - Runs, strike rate, batting average, and boundaries distribution histograms
  - Season performance comparison (runs, boundaries, centuries)

## Project Structure

```
project_3_ipl_performance/
├── README.md
├── ipl_analytics_dashboard.py
├── dataset_ipl.csv
└── exports/
    ├── 1_player_performance_dashboard.png
    ├── 2_season_wise_analysis.png
    ├── 3_player_radar_chart.png
    ├── 4_boundary_analysis.png
    ├── 5_consistency_analysis.png
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
   cd project_3_ipl_performance
   python ipl_analytics_dashboard.py
   ```

3. **Output:** All visualizations will be saved to the `exports/` folder as high-resolution PNG files (300 DPI)

### Customization

The `ipl_analytics_dashboard.py` script is fully customizable:
- Modify `FIG_SIZE` and `DPI` constants for different output sizes
- Adjust color schemes and styling
- Add or remove visualization sections
- Customize player selection and metrics

## Technologies

- **Python 3.x**
- **Data Analysis:**
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing
  - scipy - Statistical functions
- **Visualization:**
  - matplotlib - Advanced plotting and customization
  - seaborn - Statistical visualizations
- **Features:**
  - High-resolution output (300 DPI)
  - Professional styling and color schemes
  - Multi-panel dashboards
  - Statistical analysis integration

## Applications

- Player performance evaluation
- Team selection insights
- Fantasy cricket analysis
- Performance trend identification
- Statistical comparisons

