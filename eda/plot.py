# plot.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils.load import project_root

from sklearn.linear_model import LinearRegression

# Load custom style
style_path = os.path.join(project_root(), 'utils/styles', 'light.mplstyle')
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
    labels = ['Much Better', 'Better', 'Little Better', 'Neutral', 'Little Worse', 'Worse', 'Much Worse']

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
    labels = ['Much Better', 'Better', 'Little Better', 'Neutral', 'Little Worse', 'Worse', 'Much Worse']

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


def plot_financial_control_variables(data, save_dir):
    num_bins = 10

    # Bin the team_rank variable into 6 bins
    data['team_rank_bin'] = pd.qcut(data['team_rank'], num_bins, labels=range(1, num_bins + 1))

    # Group by year and team rank bin, then calculate mean values for control variables
    mean_values_per_bin_year = data.groupby(['year', 'team_rank_bin']).agg({
        'team_size': 'mean',
        'mean_age': 'mean',
        'mean_value': 'mean',
        'total_value': 'mean'
    }).reset_index()

    # Create a 2x2 grid of plots with some extra space at the bottom
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Line properties
    line_props = {'marker': 'o', 'alpha': 0.5}

    # Plot for Team Size by year
    for year in mean_values_per_bin_year['year'].unique():
        year_data = mean_values_per_bin_year[mean_values_per_bin_year['year'] == year]
        axes[0, 0].plot(year_data['team_rank_bin'], year_data['team_size'], label=f'{year}', **line_props)

    axes[0, 0].set_title('Mean Team Size')
    axes[0, 0].set_xlabel('Team Rank Bin')
    axes[0, 0].set_ylabel('Mean Team Size')
    axes[0, 0].grid(True)

    # Plot for Mean Age by year
    for year in mean_values_per_bin_year['year'].unique():
        year_data = mean_values_per_bin_year[mean_values_per_bin_year['year'] == year]
        axes[0, 1].plot(year_data['team_rank_bin'], year_data['mean_age'], label=f'{year}', **line_props)

    axes[0, 1].set_title('Mean Age')
    axes[0, 1].set_xlabel('Team Rank Bin')
    axes[0, 1].set_ylabel('Mean Age')
    axes[0, 1].grid(True)

    # Plot for Mean Value by year
    for year in mean_values_per_bin_year['year'].unique():
        year_data = mean_values_per_bin_year[mean_values_per_bin_year['year'] == year]
        axes[1, 0].plot(year_data['team_rank_bin'], year_data['mean_value'], label=f'{year}', **line_props)

    axes[1, 0].set_title('Mean Market Value')
    axes[1, 0].set_xlabel('Team Rank Bin')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].grid(True)

    # Plot for Total Value by year
    for year in mean_values_per_bin_year['year'].unique():
        year_data = mean_values_per_bin_year[mean_values_per_bin_year['year'] == year]
        axes[1, 1].plot(year_data['team_rank_bin'], year_data['total_value'], label=f'{year}', **line_props)

    axes[1, 1].set_title('Total Market Value')
    axes[1, 1].set_xlabel('Team Rank Bin')
    axes[1, 1].set_ylabel('Mean Total Value')
    axes[1, 1].grid(True)

    # Adjust the layout to create space for the legend below the plots
    plt.subplots_adjust(bottom=0.15)

    # Combine all handles and labels for the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Place the legend below the plots
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle('Financial Control Variables per Team Rank (Binned)', fontsize=16)

    # Save and show the plot
    save_plot('financial_control_variables_per_year', save_dir)


def plot_effect_of_travel_distance(data, save_dir):
    # Filter data for next_fixture_days <= 5
    filtered_data = data[data['next_fixture_days'] <= 5].copy()

    # Define distance bins and corresponding labels
    bins = [-1, 0, 100, 250, 500, float('inf')]
    labels = ['Home (0 km)', '0-100 km', '100-250 km', '250-500 km', '500+ km']

    # Bin the distances with explicit .loc assignment
    filtered_data['distance_bin'] = pd.cut(filtered_data['distance'], bins=bins, labels=labels)

    # Group by distance bin and cup round (stage) to calculate mean points and count of records
    grouped = filtered_data.groupby(['distance_bin', 'stage']).agg(
        mean_points=('next_team_points', 'mean'),
        count=('next_team_points', 'size')
    ).reset_index()

    # Repeat for aggregated data across all rounds
    aggregated = filtered_data.groupby('distance_bin').agg(
        mean_points=('next_team_points', 'mean'),
        count=('next_team_points', 'size')
    ).reset_index()
    aggregated['stage'] = 'All Rounds'

    # Combine the round-specific and aggregated data
    combined_data = pd.concat([grouped, aggregated], ignore_index=True)

    # Adjust the layout: create six subplots (5 rounds + 1 aggregated)
    g = sns.catplot(
        x='distance_bin',
        y='mean_points',
        col='stage',
        col_order=[1, 2, 3, 4, 5, 'All Rounds'],  # Order the plots to include the aggregated data
        col_wrap=2,  # Use 2 columns to create a 3x2 grid of plots
        data=combined_data,
        kind='bar',
        height=5,  # Increase the height of each plot
        aspect=1.5,
        hue='distance_bin',  # Set hue to distance_bin to resolve FutureWarning
        legend=False,  # Disable the default legend
        palette='viridis',
        alpha=0.8
    )

    g.set_axis_labels("Travel Distance for Cup Game", "Average Points in Next League Match")
    g.set_titles("Round {col_name}")
    g.set(ylim=(0, 3))  # Assuming points range between 0 and 3
    g.fig.suptitle('Effect of Travel Distance on Next League Match Performance\n(Only Matches with ≤ 5 Days Between)',
                   y=0.975, fontsize=16)

    # Annotate the bars with the count of records
    for ax, stage in zip(g.axes.flat, g.col_names):
        stage_data = combined_data[combined_data['stage'] == stage]  # Filter data for the current stage
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()  # Get the height (mean points) of the bar
                if not np.isnan(height):  # Only annotate if the height is a number (not NaN)
                    x_val = bar.get_x() + bar.get_width() / 2  # Get the x position of the bar
                    distance_bin = bar.get_x()  # Use bar's position to find the matching bin

                    # Get the correct count for this bin and stage
                    count = stage_data.loc[
                        (stage_data['distance_bin'] == labels[int(bar.get_x() // bar.get_width())]), 'count'].values[0]

                    # Annotate only if the count is greater than 0
                    if count > 0:
                        ax.text(x_val, height, f'(n={count})', ha='center', va='bottom', fontsize=10, color='white')

    # Adjust grid settings for clarity
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.6)

    # Save and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make space for the title
    plt.show()
    save_plot('effect_distance_on_next_league_performance.png', save_dir)


def plot_effect_of_fixture_days(data, save_dir):
    # Filter data for next_fixture_days <= 8 and exclude next_fixture_days = 1
    filtered_data = data[(data['next_fixture_days'] <= 8) & (data['next_fixture_days'] != 1)].copy()

    # Define manual bins for team ranks
    bins = [0, 6, 18, 36, float('inf')]  # Boundaries for the rank groups
    labels = ['Top 6', 'League 1', 'League 2', 'League 3']  # Labels for each group

    # Bin the teams into rank categories using manual splits
    filtered_data['rank_category'] = pd.cut(filtered_data['team_rank'], bins=bins, labels=labels, right=True)

    # Group by next_fixture_days and rank category, then calculate mean points in the next league match
    grouped_data = filtered_data.groupby(['next_fixture_days', 'rank_category'])[
        'next_team_points'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(x='next_fixture_days', y='next_team_points', hue='rank_category', data=grouped_data,
                palette='viridis', alpha=0.8)

    # Add labels and title
    plt.xlabel('Days Between Cup and League Fixture')
    plt.ylabel('Average Points in League Fixture')
    plt.title('Effect of Days between Cup and League Fixture on Next League Fixture Performance')
    plt.ylim(0, 3)  # Assuming points range between 0 and 3
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add a legend for rank categories
    plt.legend(title='Team Rank')

    # Show and save the plot
    plt.tight_layout()
    plt.show()


def plot_fixture_days_vs_points_regression(data, save_dir):
    # Filter data for next_fixture_days <= 8 and exclude next_fixture_days = 1
    filtered_data = data[(data['next_fixture_days'] <= 8) & (data['next_fixture_days'] != 1)].copy()

    # Define manual bins for team ranks
    bins = [0, 6, 18, 36, float('inf')]  # Boundaries for the rank groups
    labels = ['Top 6', 'League 1', 'League 2', 'League 3']  # Labels for each group

    # Bin the teams into rank categories using manual splits
    filtered_data['rank_category'] = pd.cut(filtered_data['team_rank'], bins=bins, labels=labels, right=True)

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Plotting for each rank category with regression lines
    for rank_category in filtered_data['rank_category'].unique():
        subset = filtered_data[filtered_data['rank_category'] == rank_category]

        # Perform linear regression
        X = subset[['next_fixture_days']]
        y = subset['next_team_points']
        model = LinearRegression().fit(X, y)

        # Predict and plot the regression line
        x_range = pd.DataFrame({'next_fixture_days': np.linspace(X.min(), X.max(), 100).flatten()})
        y_pred = model.predict(x_range)
        plt.plot(x_range, y_pred, label=f'{rank_category}: y={model.coef_[0]:.2f}x + {model.intercept_:.2f}', linewidth=2)

        # Plot the actual points with smaller size
        plt.scatter(X, y, alpha=0.1, s=10)

    # Customize the plot
    plt.xlabel('Days Between Cup and League Fixture')
    plt.ylabel('Average Points in League Fixture')
    plt.title('Effect of Days between Cup and League Fixture on Next League Fixture Performance')

    # Remove the grid
    plt.grid(False)

    # Display the legend with regression coefficients
    plt.legend(title='Regression Lines per Ranking')

    # Show and save the plot
    plt.tight_layout()
    plt.show()


def plot_extra_time_effect_by_rank(data, save_dir):
    # Filter data for next_fixture_days <= 5
    filtered_data = data[data['next_fixture_days'] <= 5].copy()

    # Define manual bins for team ranks
    bins = [0, 6, 18, 36, float('inf')]  # Boundaries for the rank groups
    labels = ['Top 6', 'League 1', 'League 2', 'League 3']  # Labels for each group

    # Bin the teams into rank categories using manual splits
    filtered_data['rank_category'] = pd.cut(filtered_data['team_rank'], bins=bins, labels=labels, right=True)

    # Group by extra_time and rank category, then calculate mean points in the next league fixture
    grouped_data = filtered_data.groupby(['extra_time', 'rank_category'])['next_team_points'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(x='extra_time', y='next_team_points', hue='rank_category', data=grouped_data, palette='viridis', alpha=0.8)

    # Add labels and title
    plt.xlabel('Cup Fixture Duration')
    plt.ylabel('Average Points in Next League Fixture')
    plt.title('Effect of Extra Time in Cup Fixture on Next League Fixture Performance')
    plt.ylim(0, 3)  # Assuming points range between 0 and 3
    plt.grid(True, linestyle='--', alpha=0.6)

    # Change x-axis labels
    ax = plt.gca()
    ax.set_xticklabels(['90 Min', 'Extra Time'])

    # Add a legend for rank categories
    plt.legend(title='Team Rank')

    # Show and save the plot
    plt.tight_layout()
    plt.show()

