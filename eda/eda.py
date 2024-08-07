import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load import project_root, load_csv

# Load custom style
style_path = os.path.join(project_root(), 'utils/styles', 'dark.mplstyle')
plt.style.use(style_path)

# Load data
country = 'Germany'
cup = 'DFB_Pokal'
data = load_csv(os.path.join(project_root(), 'data', 'process_data', country, f'{cup}_processed.csv'))

# Directory to save plots
save_dir = os.path.join(project_root(), 'eda', 'plots', country)
os.makedirs(save_dir, exist_ok=True)

# Variables of interest
variables = ['team_rank', 'team_rank_diff', 'next_team_win', 'next_team_points', 'rank_diff', 'team_better', 'team_win',
             'stage']

# Descriptive Statistics
print("Descriptive Statistics:")
print(data[variables].describe())

# Unique Values
print("\nUnique Values:")
for var in variables:
    print(f"{var}: {data[var].unique()}\n")


# Function to save plots
def save_plot(plotname):
    plt.savefig(os.path.join(save_dir, f'{plotname}.png'))
    plt.close()


# Function for loading data
def load_csv_data(country, file_name):
    file_path = os.path.join(project_root, 'data', 'process_data', country, file_name)
    return pd.read_csv(file_path)


# 1. Plot next match performance
def plot_next_match_performance(data):
    data['next_team_points'].fillna(0, inplace=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Average Points in Next League Match
    mean_points = data.groupby(['stage', 'team_win'])['next_team_points'].mean().unstack()
    mean_points.columns = ['No Win', 'Win']
    mean_points.plot(kind='bar', stacked=False, ax=axes[0])
    axes[0].set_title('Next Fixture Average Points vs. Win/Loss in Cup Fixture')
    axes[0].set_ylabel('Average Points')
    axes[0].set_xlabel('Round')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
    axes[0].legend(title='Cup Fixture Result')

    # Subplot 2: Win Percentage in Next League Match
    win_percentage = data.groupby(['stage', 'team_win'])['next_team_win'].mean().unstack() * 100
    win_percentage.columns = ['No Win', 'Win']
    win_percentage.plot(kind='bar', stacked=False, ax=axes[1])
    axes[1].set_title('Next Fixture Win Percentage vs. Win/Loss in Cup Fixture')
    axes[1].set_ylabel('Win Percentage (%)')
    axes[1].set_xlabel('Round')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
    axes[1].legend(title='Cup Fixture Result')

    plt.tight_layout()
    plt.show()
    save_plot('next_match_performance')


plot_next_match_performance(data)


# 3. Impact of Winning in a Round on Domestic League Performance
def plot_league_performance(data):
    mean_rank_diff = data.groupby(['stage', 'team_win'])['team_rank_diff'].mean().unstack()
    mean_rank_diff.columns = ['No Win', 'Win']
    mean_rank_diff.plot(kind='bar', stacked=False)
    plt.title('Change in League Rank vs. Win/Loss in Cup Fixture')
    plt.ylabel('Average Change in Rank')
    plt.xlabel('Round')
    plt.xticks(rotation=90)
    plt.legend(title='Cup Fixture Result')
    plt.show()
    save_plot('win_round_vs_league_performance')


plot_league_performance(data)

# Summary of missing values
sns.heatmap(data.isnull(), cbar=False)
plt.title('Missing Values Heatmap')
save_plot('missing_values_heatmap')

# Summary
print("\nSummary of missing values:")
print(data.isnull().sum())
