import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from cycler import cycler

from utils.load import project_root

# Load custom style
style_path = os.path.join(project_root(), 'utils/styles', 'light.mplstyle')
plt.style.use(style_path)

additional_colors = ['#ff1e56', '#ffac41', '#5a189a', '#4cc9f0', '#7209b7']
color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color'] + additional_colors


def save_plot(plotname, save_dir):
    plt.savefig(os.path.join(save_dir, f'{plotname}.png'))
    plt.close()


def is_categorical(variable_data):
    return len(variable_data.dropna().unique()) <= 3


# Distribution and Count Plots
def plot_variable_distributions(data, outcome_vars, instrumental_vars, treatment_vars, control_vars, save_dir):
    variables = outcome_vars + instrumental_vars + treatment_vars + control_vars
    for var in variables:
        plt.figure(figsize=(10, 6))

        unique_years = data['year'].nunique()
        adjusted_palette = color_palette[:unique_years]  # Adjust palette size

        if is_categorical(data[var]):
            sns.countplot(x=var, data=data, alpha=0.5, hue='year', palette=adjusted_palette)
            plt.title(f'Count of {var} per Year')
            plt.ylabel('Count')
        else:
            for year, year_data in data.groupby('year'):
                clean_data = year_data[var].dropna()
                sns.histplot(clean_data, kde=True, element='step', alpha=0.4, stat='density', label=str(year))
            plt.title(f'Distribution of {var} per Year')
            plt.ylabel('Density')
        plt.xlabel(var)
        if var == 'next_fixture_days':
            plt.xlim(0, 21)
        plt.legend(title='Year', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
        plt.tight_layout()
        save_plot(f'distribution_{var}', save_dir)


def plot_financial_control_variables(data, save_dir):
    # Define bins and labels for team ranking
    bins = [0, 6, 18, 36, float('inf')]
    labels = ['Top 6', 'League 1', 'League 2', 'League 3']

    # Apply binning based on the defined categories
    data['team_rank_bin'] = pd.cut(data['team_rank'], bins=bins, labels=labels, right=True)

    # Calculate mean values per bin and year
    mean_values_per_bin_year = data.groupby(['year', 'team_rank_bin']).agg({
        'team_size': 'mean',
        'mean_age': 'mean',
        'mean_value': 'mean',
        'total_value': 'mean'
    }).reset_index()

    # Set up the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    line_props = {'marker': 'o', 'alpha': 0.5}

    # Plot each financial variable over time for each rank bin
    for year in mean_values_per_bin_year['year'].unique():
        year_data = mean_values_per_bin_year[mean_values_per_bin_year['year'] == year]
        axes[0, 0].plot(year_data['team_rank_bin'], year_data['team_size'], label=f'{year}', **line_props)
        axes[0, 1].plot(year_data['team_rank_bin'], year_data['mean_age'], label=f'{year}', **line_props)
        axes[1, 0].plot(year_data['team_rank_bin'], year_data['mean_value'], label=f'{year}', **line_props)
        axes[1, 1].plot(year_data['team_rank_bin'], year_data['total_value'], label=f'{year}', **line_props)

    # Set titles for subplots
    axes[0, 0].set_title('Mean Team Size')
    axes[0, 1].set_title('Mean Age')
    axes[1, 0].set_title('Mean Market Value')
    axes[1, 1].set_title('Total Market Value')

    # Customize axis labels and add grid
    for ax in axes.flat:
        ax.set_xlabel('Team Rank Category')
        ax.grid(True)

    # Adjust layout and add legend
    plt.subplots_adjust(bottom=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle('Financial Control Variables per Team Rank Category', fontsize=16)

    # Save the plot
    save_plot('distribution_financial_controls_per_year', save_dir)


# Correlation and Relationship Analysis
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


# Perfromance Analysis by Factors
def plot_avg_points_by_cup_round(data, save_dir):
    data['next_team_points'].fillna(0, inplace=True)
    plt.figure(figsize=(12, 6))
    mean_points = data.groupby(['stage', 'team_win'])['next_team_points'].mean().unstack()
    mean_points.columns = ['Loss', 'Win']
    mean_points.plot(kind='bar', stacked=False, color=['#e71d36', '#07f49e'])
    plt.title('Next Fixture Average Points vs. Win/Loss in Cup Fixture')
    plt.ylabel('Average Points')
    plt.xlabel('Round')
    plt.legend(title='Cup Fixture Result')
    plt.tight_layout()
    save_plot('performance_avg_points_by_cup_round', save_dir)


def plot_avg_points_by_cup_round_line(data, save_dir):
    plt.figure(figsize=(12, 6))
    mean_points = data.groupby(['stage', 'team_win'])['next_team_points'].mean().unstack()
    mean_points.columns = ['Loss', 'Win']
    mean_points.plot(kind='line', marker='o', color=['#e71d36', '#07f49e'])
    plt.title('Next League Fixture Performance by Cup Round Result')
    plt.ylabel('Average Points')
    plt.xlabel('Cup Round')
    plt.legend(title='Cup Fixture Result', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot('performance_avg_points_by_cup_round_line', save_dir)


def plot_win_percentage_by_cup_round(data, save_dir):
    plt.figure(figsize=(12, 6))
    win_percentage = data.groupby(['stage', 'team_win'])['next_team_win'].mean().unstack() * 100
    win_percentage.columns = ['Loss', 'Win']
    win_percentage.plot(kind='bar', stacked=False, color=['#e71d36', '#07f49e'])
    plt.title('Next Fixture Win Percentage vs. Win/Loss in Cup Fixture')
    plt.ylabel('Win Percentage (%)')
    plt.xlabel('Round')
    plt.xticks(rotation=90)
    plt.legend(title='Cup Fixture Result')
    plt.tight_layout()
    save_plot('performance_win_percentage_by_cup_round', save_dir)


def plot_league_rank_change_by_cup_round(data, save_dir):
    plt.figure(figsize=(12, 6))
    mean_rank_diff = data.groupby(['stage', 'team_win'])['team_rank_diff'].mean().unstack()
    mean_rank_diff.columns = ['Loss', 'Win']
    mean_rank_diff.plot(kind='bar', stacked=False, color=['#e71d36', '#07f49e'])
    plt.title('Change in League Rank vs. Win/Loss in Cup Fixture')
    plt.ylabel('Average Change in Rank')
    plt.xlabel('Round')
    plt.xticks(rotation=90)
    plt.legend(title='Cup Fixture Result')
    plt.tight_layout()
    save_plot('performance_league_rank_change_by_cup_round', save_dir)


def plot_next_fixture_performance_by_rank_diff(data, save_dir):
    filtered_data = data[data['next_fixture_days'] <= 5].copy()
    plt.figure(figsize=(12, 8))
    bins = [-float('inf'), -20, -5, -1, 1, 5, 20, float('inf')]
    labels = ['Much Better', 'Better', 'Little Better', 'Neutral', 'Little Worse', 'Worse', 'Much Worse']
    filtered_data['rank_diff_binned'] = pd.cut(filtered_data['rank_diff'], bins=bins, labels=labels)
    grouped_data = filtered_data.groupby(['rank_diff_binned', 'team_win'])['next_team_points'].mean().unstack()
    grouped_data.columns = ['Loss', 'Win']
    ax = grouped_data.plot(kind='bar', stacked=False, color=['#e71d36', '#07f49e'])
    plt.ylim(0, 2.5)
    plt.yticks([0.5, 1, 1.5, 2, 2.5])
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    plt.title('Next Fixture Performance per Opponent Strength')
    plt.ylabel('Average Points')
    plt.xlabel('Team strength relative to Opponent')
    plt.legend(title='Cup Fixture Result')
    plt.tight_layout()
    save_plot('performance_next_fixture_by_rank_diff', save_dir)


def plot_rank_change_by_cup_round_line(data, save_dir):
    plt.figure(figsize=(12, 6))
    mean_rank_change = data.groupby(['stage', 'team_win'])['team_rank_diff'].mean().unstack()
    mean_rank_change.columns = ['Lose', 'Win']
    mean_rank_change.plot(kind='line', marker='o', color=['#e71d36', '#07f49e'])
    plt.title('Rank Change by Cup Round Result')
    plt.ylabel('Average Rank Change')
    plt.xlabel('Cup Stage')
    plt.legend(title='Cup Fixture Result', loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot('performance_rank_change_by_cup_stage_line', save_dir)


def plot_league_rank_change_by_opponent_strenght(data, save_dir):
    plt.figure(figsize=(12, 8))
    bins = [-float('inf'), -20, -5, -1, 1, 5, 20, float('inf')]
    labels = ['Much Better', 'Better', 'Little Better', 'Neutral', 'Little Worse', 'Worse', 'Much Worse']
    data['rank_diff_binned'] = pd.cut(data['rank_diff'], bins=bins, labels=labels)
    grouped_data = data.groupby(['rank_diff_binned', 'team_win'])['team_rank_diff'].mean().unstack()
    grouped_data.columns = ['Loss', 'Win']
    ax = grouped_data.plot(kind='bar', stacked=False, color=['#e71d36', '#07f49e'])
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    plt.title('Season-End Rank Change per Opponent Strength by W/L in Cup')
    plt.ylabel('Average Rank Change')
    plt.xlabel('Opponent Strength Relative to Team')
    plt.legend(title='Cup Fixture Result')
    plt.tight_layout()
    save_plot('performance_league_rank_change_by_opponent_strength', save_dir)


def plot_effect_travel_distance(data, save_dir):
    filtered_data = data[data['next_fixture_days'] <= 5].copy()
    bins = [-1, 0, 100, 250, 500, float('inf')]
    labels = ['Home (0 km)', '0-100 km', '100-250 km', '250-500 km', '500+ km']
    filtered_data['distance_bin'] = pd.cut(filtered_data['distance'], bins=bins, labels=labels)

    grouped = filtered_data.groupby(['distance_bin', 'stage']).agg(
        mean_points=('next_team_points', 'mean'),
        count=('next_team_points', 'size')
    ).reset_index()

    aggregated = filtered_data.groupby('distance_bin').agg(
        mean_points=('next_team_points', 'mean'),
        count=('next_team_points', 'size')
    ).reset_index()
    aggregated['stage'] = 'All Rounds'
    combined_data = pd.concat([grouped, aggregated], ignore_index=True)

    g = sns.catplot(
        x='distance_bin',
        y='mean_points',
        col='stage',
        col_order=[1, 2, 3, 4, 5, 'All Rounds'],
        col_wrap=2,
        data=combined_data,
        kind='bar',
        height=5,
        aspect=1.5,
        hue='distance_bin',
        legend=False,
        palette='viridis',
        alpha=0.8
    )

    g.set_axis_labels("Travel Distance for Cup Fixture", "Average Points in Next League Fixture")
    g.set_titles("Round {col_name}")
    g.set(ylim=(0, 3))
    g.fig.suptitle(
        'Effect of Travel Distance on Next League Fixture Performance\n(Only Fixtures with ≤ 5 Days Between)', y=0.975,
        fontsize=16)

    for ax, stage in zip(g.axes.flat, g.col_names):
        stage_data = combined_data[combined_data['stage'] == stage]
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not np.isnan(height):
                    x_val = bar.get_x() + bar.get_width() / 2
                    count = stage_data.loc[
                        (stage_data['distance_bin'] == labels[int(bar.get_x() // bar.get_width())]), 'count'].values[0]
                    if count > 0:
                        ax.text(x_val, height, f'(n={count})', ha='center', va='bottom', fontsize=10, color='white')

    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    save_plot('performance_effect_distance', save_dir)


def plot_effect_fixture_days_regression(data, save_dir):
    filtered_data = data[(data['next_fixture_days'] <= 8) & (data['next_fixture_days'] != 1)].copy()
    bins = [0, 6, 18, 36, float('inf')]
    labels = ['Top 6', 'League 1', 'League 2', 'League 3']
    filtered_data['rank_category'] = pd.cut(filtered_data['team_rank'], bins=bins, labels=labels, right=True)

    plt.figure(figsize=(12, 8))
    for rank_category in filtered_data['rank_category'].unique():
        subset = filtered_data[filtered_data['rank_category'] == rank_category]
        X = subset[['next_fixture_days']]
        y = subset['next_team_points']
        model = LinearRegression().fit(X, y)
        x_range = pd.DataFrame({'next_fixture_days': np.linspace(X.min(), X.max(), 100).flatten()})
        y_pred = model.predict(x_range)
        plt.plot(x_range, y_pred, label=f'{rank_category}: y={model.coef_[0]:.2f}x + {model.intercept_:.2f}',
                 linewidth=2)
        plt.scatter(X, y, alpha=0.1, s=10)

    plt.xlabel('Days Between Cup and League Fixture')
    plt.ylabel('Average Points in League Fixture')
    plt.title('Effect of Days Between Cup and League Fixture on Next League Fixture Performance')
    plt.grid(False)
    plt.legend(title='Regression Lines per Ranking')
    plt.tight_layout()
    save_plot('performance_effect_distance_regression', save_dir)


def plot_effect_fixture_days(data, save_dir):
    filtered_data = data[(data['next_fixture_days'] <= 8) & (data['next_fixture_days'] != 1)].copy()
    bins = [0, 6, 18, 36, float('inf')]
    labels = ['Top 6', 'League 1', 'League 2', 'League 3']
    filtered_data['rank_category'] = pd.cut(filtered_data['team_rank'], bins=bins, labels=labels, right=True)

    grouped_data = filtered_data.groupby(['next_fixture_days', 'rank_category'])[
        'next_team_points'].mean().reset_index()

    plt.figure(figsize=(12, 8))
    sns.barplot(x='next_fixture_days', y='next_team_points', hue='rank_category', data=grouped_data, palette='viridis',
                alpha=0.8)
    plt.xlabel('Days Between Cup and League Fixture')
    plt.ylabel('Average Points in League Fixture')
    plt.title('Effect of Days Between Cup and League Fixture on Next League Fixture Performance')
    plt.ylim(0, 3)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Team Rank')
    plt.tight_layout()
    save_plot('performance_effect_fixture_days', save_dir)


def plot_extra_time_effect_on_performance(data, save_dir):
    filtered_data = data[data['next_fixture_days'] <= 5].copy()
    bins = [0, 6, 18, 36, float('inf')]
    labels = ['Top 6', 'League 1', 'League 2', 'League 3']
    filtered_data['rank_category'] = pd.cut(filtered_data['team_rank'], bins=bins, labels=labels, right=True)

    grouped_data = filtered_data.groupby(['extra_time', 'rank_category'])['next_team_points'].mean().reset_index()

    plt.figure(figsize=(12, 8))
    sns.barplot(x='extra_time', y='next_team_points', hue='rank_category', data=grouped_data, palette='viridis',
                alpha=0.8)
    plt.xlabel('Cup Fixture Duration')
    plt.ylabel('Average Points in Next League Fixture')
    plt.title('Effect of Extra Time in Cup Fixture on Next League Fixture Performance')
    plt.ylim(0, 3)
    plt.grid(True, linestyle='--', alpha=0.6)
    ax = plt.gca()

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['90 Min', 'Extra Time'])

    plt.legend(title='Team Rank')
    plt.tight_layout()
    save_plot('performance_effect_extra_time', save_dir)
