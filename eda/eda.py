import os
import pandas as pd
from utils.load import project_root, load_csv
from plot import *

# Load data
country = 'Germany'
cup = 'DFB_Pokal'
data = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_processed.csv'))

# Directory to save plots
save_dir = os.path.join(project_root(), 'eda', 'plots/new', country)
os.makedirs(save_dir, exist_ok=True)

# Variables of interest
outcome_vars = ['team_rank', 'team_rank_diff', 'next_team_win', 'next_team_points']
instrumental_vars = ['rank_diff', 'team_better']
treatment_vars = ['team_win']
control_vars = ['distance', 'team_size', 'mean_age', 'total_value', 'extra_time', 'team_home', 'next_fixture_days']

with open(f'{save_dir}/descriptive_statistics.txt', 'w') as f:
    # Redirect print statements to the file

    f.write("Descriptive Statistics for Outcome Variables:\n")
    f.write(data[outcome_vars].describe().to_string())
    f.write("\n\n")  # Add spacing between sections

    f.write("Descriptive Statistics for Instrumental Variables:\n")
    f.write(data[instrumental_vars].describe().to_string())
    f.write("\n\n")

    f.write("Descriptive Statistics for Treatment Variables:\n")
    f.write(data[treatment_vars].describe().to_string())
    f.write("\n\n")

    f.write("Descriptive Statistics for Control Variables:\n")
    f.write(data[control_vars].describe().to_string())
    f.write("\n")

# Call the plotting function for variable distributions
plot_variable_distributions(data, outcome_vars, instrumental_vars, treatment_vars, control_vars, save_dir)
plot_financial_control_variables(data, save_dir)

plot_avg_points_by_cup_round_line(data, save_dir)
plot_rank_change_by_cup_round_line(data, save_dir)

plot_next_fixture_performance_by_rank_diff(data, save_dir)
plot_league_rank_change_by_opponent_strenght(data, save_dir)

# Call the function to plot the correlation heatmap
plot_correlation_heatmap(data, outcome_vars, instrumental_vars, treatment_vars, control_vars, save_dir)


plot_avg_points_by_cup_round(data, save_dir)
plot_win_percentage_by_cup_round(data, save_dir)


plot_league_rank_change_by_cup_round(data, save_dir)

plot_effect_travel_distance(data, save_dir)

plot_effect_fixture_days(data, save_dir)

plot_effect_fixture_days_regression(data, save_dir)

plot_extra_time_effect_on_performance(data, save_dir)
