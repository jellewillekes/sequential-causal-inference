# plot.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.load import project_root

# Load custom style
style_path = os.path.join(project_root(), 'utils/styles', 'dark.mplstyle')
plt.style.use(style_path)


# Function to determine if a variable is categorical/discrete with only a few distinct values
def is_categorical(variable_data):
    return len(variable_data.dropna().unique()) <= 3


# Function to save plots
def save_plot(plotname, save_dir):
    plt.savefig(os.path.join(save_dir, f'{plotname}.png'))
    plt.close()


def plot_variable(data, outcome_vars, instrumental_vars, treatment_vars, control_vars, save_dir):
    for var in outcome_vars + instrumental_vars + treatment_vars + control_vars:
        plt.figure(figsize=(10, 6))  # Ensure each figure is associated with plotting commands.

        if is_categorical(data[var]):
            sns.countplot(x=var, data=data, alpha=0.8, hue='year')  # Group by 'year' to ensure legend is created
            plt.title(f'Count of {var} per Year')
            plt.ylabel('Count')
            plt.xlabel(var)
        else:
            # Plot each year separately with transparency for continuous variables
            for year, year_data in data.groupby('year'):
                clean_data = year_data[var].dropna()  # Remove NaN values for processing
                sns.histplot(clean_data, kde=True, element='step', alpha=0.4,
                             stat='density', label=str(year))
            plt.title(f'Distribution of {var} per Year')
            plt.ylabel('Density')

        if var == 'next_fixture_days':
            plt.xlim(0, 21)

        plt.xlabel(var)
        plt.legend(title='Year', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=True)
        plt.tight_layout()
        save_plot(f'{var}_yearly_distribution', save_dir)
        plt.show()


# Function to plot correlation heatmap with group headers and borders
def plot_correlation_heatmap(data, outcome_vars, instrumental_vars, treatment_vars, control_vars, save_dir):
    all_vars = outcome_vars + instrumental_vars + treatment_vars + control_vars
    corr = data[all_vars].corr()

    # Plot the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.1f',
                cbar_kws={'label': 'Correlation Coefficient'},
                xticklabels=all_vars, yticklabels=all_vars)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add borders around each group
    label_positions = [len(outcome_vars), len(outcome_vars) + len(instrumental_vars),
                       len(outcome_vars) + len(instrumental_vars) + len(treatment_vars)]
    for pos in label_positions:
        plt.gca().add_patch(plt.Rectangle((pos, pos), len(all_vars) - pos, len(all_vars) - pos,
                                          fill=False, edgecolor='black', lw=2))
        plt.gca().add_patch(plt.Rectangle((0, pos), pos, len(all_vars) - pos,
                                          fill=False, edgecolor='black', lw=2))
        plt.gca().add_patch(plt.Rectangle((pos, 0), len(all_vars) - pos, pos,
                                          fill=False, edgecolor='black', lw=2))

    # Add group labels once on top and left side with spacing
    label_offset = -1.2  # Increase this value for more space
    plt.text(len(outcome_vars) / 2, -1.5 - label_offset, 'Y', ha='center', va='center', fontsize=14, weight='bold')
    plt.text(len(outcome_vars) + len(instrumental_vars) / 2, -1.5 - label_offset, 'Z', ha='center', va='center',
             fontsize=14, weight='bold')
    plt.text(len(outcome_vars) + len(instrumental_vars) + len(treatment_vars) / 2, -1.5 - label_offset, 'T',
             ha='center',
             va='center', fontsize=14, weight='bold')
    plt.text(len(outcome_vars) + len(instrumental_vars) + len(treatment_vars) + len(control_vars) / 2,
             -1.5 - label_offset, 'C',
             ha='center', va='center', fontsize=14, weight='bold')

    label_offset = -15.8
    plt.text(-1.5 - label_offset, len(outcome_vars) / 2, 'Y', ha='center', va='center', fontsize=14, weight='bold')
    plt.text(-1.5 - label_offset, len(outcome_vars) + len(instrumental_vars) / 2, 'Z', ha='center', va='center',
             fontsize=14, weight='bold')
    plt.text(-1.5 - label_offset, len(outcome_vars) + len(instrumental_vars) + len(treatment_vars) / 2, 'T',
             ha='center',
             va='center', fontsize=14, weight='bold')
    plt.text(-1.5 - label_offset,
             len(outcome_vars) + len(instrumental_vars) + len(treatment_vars) + len(control_vars) / 2, 'C',
             ha='center', va='center', fontsize=14, weight='bold')

    # Adjust the title position above the labels
    plt.title('Correlation Heatmap', pad=40, fontsize=14)  # Increase pad to move title higher

    plt.tight_layout()
    save_plot('correlation_heatmap', save_dir)


def plot_average_points_next_match_per_round(data, save_dir):
    data['next_team_points'].fillna(0, inplace=True)

    plt.figure(figsize=(12, 6))

    # Average Points in Next League Match
    mean_points = data.groupby(['stage', 'team_win'])['next_team_points'].mean().unstack()
    mean_points.columns = ['No Win', 'Win']
    mean_points.plot(kind='bar', stacked=False)

    plt.title('Next Fixture Average Points vs. Win/Loss in Cup Fixture')
    plt.ylabel('Average Points')
    plt.xlabel('Round')
    plt.xticks(rotation=90)
    plt.legend(title='Cup Fixture Result')

    plt.tight_layout()
    save_plot('average_points_next_match_per_round', save_dir)


def plot_average_points_next_match_per_round_line(data, save_dir):
    plt.figure(figsize=(12, 6))

    # Group by cup stage and whether the team won, then calculate the mean of next league game points
    mean_points = data.groupby(['stage', 'team_win'])['next_team_points'].mean().unstack()
    mean_points.columns = ['Lose', 'Win']

    # Plotting the line chart
    mean_points.plot(kind='line', marker='o')

    # Customizing the plot
    plt.title('Next League Performance by Cup Round Result')
    plt.ylabel('Average Points')
    plt.xlabel('Cup Round')
    plt.legend(title='Cup Fixture Result', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save and show the plot
    plt.savefig(os.path.join(save_dir, 'average_next_league_game_points_by_cup_stage_line.png'))
    plt.show()


def plot_win_percentage_next_match_per_round(data, save_dir):
    plt.figure(figsize=(12, 6))

    # Win Percentage in Next League Match
    win_percentage = data.groupby(['stage', 'team_win'])['next_team_win'].mean().unstack() * 100
    win_percentage.columns = ['No Win', 'Win']
    win_percentage.plot(kind='bar', stacked=False)

    plt.title('Next Fixture Win Percentage vs. Win/Loss in Cup Fixture')
    plt.ylabel('Win Percentage (%)')
    plt.xlabel('Round')
    plt.xticks(rotation=90)
    plt.legend(title='Cup Fixture Result')

    plt.tight_layout()
    save_plot('win_percentage_next_match_per_round', save_dir)


def plot_league_performance(data, save_dir):
    mean_rank_diff = data.groupby(['stage', 'team_win'])['team_rank_diff'].mean().unstack()
    mean_rank_diff.columns = ['No Win', 'Win']
    mean_rank_diff.plot(kind='bar', stacked=False)
    plt.title('Change in League Rank vs. Win/Loss in Cup Fixture')
    plt.ylabel('Average Change in Rank')
    plt.xlabel('Round')
    plt.xticks(rotation=90)
    plt.legend(title='Cup Fixture Result')
    save_plot('win_round_vs_league_performance', save_dir)


def plot_next_game_performance_by_rank_diff(data, save_dir):
    # Filter data for next_fixture_days <= 5
    filtered_data = data[data['next_fixture_days'] <= 5]

    plt.figure(figsize=(12, 8))

    # Define bins and corresponding labels
    bins = [-float('inf'), -20, -5, -1, 1, 5, 20, float('inf')]
    labels = ['Higher League', 'Better', 'Little Better', 'Neutral', 'Little Worse', 'Worse', 'Lower League']

    # Bin rank_diff with the specified labels
    filtered_data['rank_diff_binned'] = pd.cut(filtered_data['rank_diff'], bins=bins, labels=labels)

    # Group by binned rank_diff and team_win, then calculate the mean of next_team_points
    grouped_data = filtered_data.groupby(['rank_diff_binned', 'team_win'])['next_team_points'].mean().unstack()

    # Update the legend labels
    grouped_data.columns = ['Loss', 'Win']

    # Plotting without additional color palettes (using style defined in the script)
    ax = grouped_data.plot(kind='bar', stacked=False)

    # Customizing the y-axis to show only integer values and setting gridlines at 0.5 intervals
    plt.ylim(0, 2.5)  # Set y-limit to 2.5
    plt.yticks([0.5, 1, 1.5, 2, 2.5])  # Set y-axis ticks at 0.5 intervals

    # Ensure grid lines are visible at each tick
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)

    plt.title('Next Fixture Performance per Strength Opponent by W/L in Cup')
    plt.ylabel('Average Points')
    plt.xlabel('Team Better than Opponent')
    plt.legend(title='Cup Fixture Result')

    plt.tight_layout()

    # Save the plot before showing it
    plt.savefig(os.path.join(save_dir, 'next_game_performance_by_rank_diff.png'))


def plot_average_rank_change_line(data, save_dir):
    plt.figure(figsize=(12, 6))

    # Group by cup stage and whether the team won, then calculate the mean rank change
    mean_rank_change = data.groupby(['stage', 'team_win'])['team_rank_diff'].mean().unstack()
    mean_rank_change.columns = ['Lose', 'Win']

    # Plotting the line chart
    mean_rank_change.plot(kind='line', marker='o')

    # Customizing the plot
    plt.title('Rank Change by Cup Round Result')
    plt.ylabel('Average Rank Change')
    plt.xlabel('Cup Stage')
    plt.legend(title='Cup Fixture Result', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save and show the plot
    plt.savefig(os.path.join(save_dir, 'average_rank_change_by_cup_stage_line.png'))
    plt.show()


def plot_league_performance_by_rank_diff(data, save_dir):
    plt.figure(figsize=(12, 8))

    # Define bins and corresponding labels
    bins = [-float('inf'), -20, -5, -1, 1, 5, 20, float('inf')]
    labels = ['Higher League', 'Better', 'Little Better', 'Neutral', 'Little Worse', 'Worse', 'Lower League']

    # Bin rank_diff with the specified labels
    data['rank_diff_binned'] = pd.cut(data['rank_diff'], bins=bins, labels=labels)

    # Group by binned rank_diff and team_win, then calculate the mean of team_rank_diff
    grouped_data = data.groupby(['rank_diff_binned', 'team_win'])['team_rank_diff'].mean().unstack()

    # Update the legend labels
    grouped_data.columns = ['Loss', 'Win']

    # Plotting without additional color palettes (using style defined in the script)
    ax = grouped_data.plot(kind='bar', stacked=False)

    # Ensure grid lines are visible at each tick
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)

    plt.title('Season-End Rank Change per Strength Opponent by W/L in Cup')
    plt.ylabel('Average Rank Change')
    plt.xlabel('Team Better than Opponent')
    plt.legend(title='Cup Fixture Result')

    plt.tight_layout()

    # Save the plot before showing it
    plt.savefig(os.path.join(save_dir, 'league_performance_by_rank_diff.png'))
