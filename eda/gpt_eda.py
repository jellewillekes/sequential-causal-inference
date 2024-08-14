import os
import pandas as pd
from utils.load import project_root, load_csv
from plot import *

# Load data
country = 'Germany'
cup = 'DFB_Pokal'
data = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_processed.csv'))

# Directory to save plots
save_dir = os.path.join(project_root(), 'eda', 'plots', country)
os.makedirs(save_dir, exist_ok=True)

# Variables of interest
outcome_vars = ['team_rank', 'team_rank_diff', 'next_team_win', 'next_team_points']
instrumental_vars = ['rank_diff', 'team_better']
treatment_vars = ['team_win']
control_vars = ['distance', 'team_size', 'mean_age', 'total_value', 'extra_time', 'team_home', 'next_fixture_days']

# Descriptive Statistics for each group of variables
print("Descriptive Statistics for Outcome Variables:")
print(data[outcome_vars].describe())

print("\nDescriptive Statistics for Instrumental Variables:")
print(data[instrumental_vars].describe())

print("\nDescriptive Statistics for Treatment Variables:")
print(data[treatment_vars].describe())

print("\nDescriptive Statistics for Control Variables:")
print(data[control_vars].describe())

# Call the plotting function for variable distributions
plot_variable(data, outcome_vars, instrumental_vars, treatment_vars, control_vars, save_dir)

plot_average_points_next_match_per_round_line(data, save_dir)
plot_average_rank_change_line(data, save_dir)

plot_next_game_performance_by_rank_diff(data, save_dir)
plot_league_performance_by_rank_diff(data, save_dir)

# Call the function to plot the correlation heatmap
plot_correlation_heatmap(data, outcome_vars, instrumental_vars, treatment_vars, control_vars, save_dir)

# Call the function to plot next match performance
plot_average_points_next_match_per_round(data, save_dir)
plot_win_percentage_next_match_per_round(data, save_dir)

# Call the function to plot the impact of winning on league performance
plot_league_performance(data, save_dir)
