import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from raw_data.loader import get_project_root
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

project_root = get_project_root()

# Ensure plots directory exists
def ensure_country_plots_dir(country):
    country_plots_dir = os.path.join("plots", country)
    os.makedirs(country_plots_dir, exist_ok=True)
    return country_plots_dir

# Function for loading data
def load_csv_data(country, file_name):
    project_root = get_project_root()
    file_path = os.path.join(project_root, 'process_data', country, file_name)
    return pd.read_csv(file_path)

def preprocess_data(fixtures_df, standings_df):
    # Filter and preprocess data
    standings_df = standings_df[standings_df['Year'] > 2011].copy()
    standings_df['PrevYear'] = standings_df['Year'] + 1

    # Merge operations
    stages_df = (fixtures_df[fixtures_df['Year'] > 2011]
                 .merge(standings_df[['PrevYear', 'Team_id', 'NationalRank']],
                        left_on=['Year', 'Opponent_id'],
                        right_on=['PrevYear', 'Team_id'],
                        how='left',
                        suffixes=('', '_opponent'))
                 .rename(columns={'NationalRank': 'Opponent_rank_prev'})
                 .drop(columns=['PrevYear', 'Team_id_opponent'])
                 .merge(standings_df[['PrevYear', 'Team_id', 'NationalRank']],
                        left_on=['Year', 'Team_id'],
                        right_on=['PrevYear', 'Team_id'],
                        how='left')
                 .rename(columns={'NationalRank': 'Team_rank_prev'})
                 .drop(columns=['PrevYear'])
                 .merge(standings_df[['Year', 'Team_id', 'NationalRank']],
                        left_on=['Year', 'Team_id'],
                        right_on=['Year', 'Team_id'],
                        how='left')
                 .rename(columns={'NationalRank': 'Team_rank'})
                 )

    # Handle missing values and create new columns
    stages_df = (stages_df
                 .assign(Team_rank_prev=lambda df: df['Team_rank_prev'].fillna(75),
                         Opponent_rank_prev=lambda df: df['Opponent_rank_prev'].fillna(75),
                         Rank_diff=lambda df: df['Opponent_rank_prev'] - df['Team_rank_prev'],
                         const=1)
                 .dropna(subset=['Stage'])
                 .assign(Stage=lambda df: df['Stage'].astype(int))
                 .query('2012 <= Year <= 2022'))

    # Add maximum stage reached by each team in each year
    max_stage_df = fixtures_df.groupby(['Year', 'Team_id'])['Stage'].max().reset_index()
    max_stage_df.rename(columns={'Stage': 'Team_max_stage'}, inplace=True)

    # Merge max stage information into stages_df
    stages_df = pd.merge(stages_df, max_stage_df, on=['Year', 'Team_id'], how='left')

    return stages_df

# Visualization functions
def plot_histogram_and_kde(df, column, country_plots_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(f'Histogram and KDE of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(country_plots_dir, f'{column}_hist_kde.png'))
    plt.show()

def plot_box_by_stage(df, stage_column, y_column, country_plots_dir):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[stage_column], y=df[y_column])
    plt.title(f'Box Plot of {y_column} by {stage_column}')
    plt.xlabel(stage_column)
    plt.ylabel(y_column)
    plt.savefig(os.path.join(country_plots_dir, f'{y_column}_by_{stage_column}_box.png'))
    plt.show()

def plot_scatter_with_regression(df, x_column, y_column, country_plots_dir):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df[x_column], y=df[y_column], line_kws={"color": "red"})
    plt.title(f'Scatter Plot with Regression Line: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.savefig(os.path.join(country_plots_dir, f'{x_column}_vs_{y_column}_scatter.png'))
    plt.show()

def plot_line_by_year(df, x_column, y_column, hue_column, country_plots_dir):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column, marker='o')
    plt.title(f'Line Plot of {y_column} over {x_column} by {hue_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=hue_column)
    plt.savefig(os.path.join(country_plots_dir, f'{y_column}_over_{x_column}_by_{hue_column}_line.png'))
    plt.show()

def plot_bar_by_stage(df, stage_column, y_column, country_plots_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df[stage_column], y=df[y_column])
    plt.title(f'Bar Plot of {y_column} by {stage_column}')
    plt.xlabel(stage_column)
    plt.ylabel(y_column)
    plt.savefig(os.path.join(country_plots_dir, f'{y_column}_by_{stage_column}_bar.png'))
    plt.show()

def plot_heatmap(df, columns, country_plots_dir):
    plt.figure(figsize=(12, 8))
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Correlation Matrix')
    plt.savefig(os.path.join(country_plots_dir, 'correlation_heatmap.png'))
    plt.show()

def plot_avg_rank_by_stage(stages_df, country_plots_dir):
    # Group by Stage and Team_win, and calculate the average Team_rank
    avg_rank_df = stages_df.groupby(['Stage', 'Team_win'])['Team_rank'].mean().reset_index()

    # Map values for clarity
    avg_rank_df['Match Outcome'] = avg_rank_df['Team_win'].map({0: 'Lost', 1: 'Won'})

    # Plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=avg_rank_df, x='Stage', y='Team_rank', hue='Match Outcome', palette='muted')
    plt.title('Average Team Rank by Stage and Match Outcome')
    plt.xlabel('Stage')
    plt.ylabel('Average Team Rank')
    plt.legend(title='Match Outcome')
    plt.savefig(os.path.join(country_plots_dir, "avg_rank_by_stage.png"))
    plt.show()


def analyze_first_stage_by_stage(stages_df, country_plots_dir):
    unique_stages = stages_df['Stage'].unique()

    for stage in unique_stages:
        df_stage = stages_df[stages_df['Stage'] == stage]

        # Correlation between Rank_diff and Team_win
        correlation = df_stage[['Rank_diff', 'Team_win']].corr().iloc[0, 1]
        print(f'Stage {stage}: Correlation between Rank_diff and Team_win: {correlation}')

        # Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Rank_diff', y='Team_win', data=df_stage, alpha=0.5)
        plt.title(f'Stage {stage}: Rank Difference vs. Probability of Winning')
        plt.xlabel('Rank Difference')
        plt.ylabel('Probability of Winning')
        plt.savefig(os.path.join(country_plots_dir, f'rank_diff_vs_win_scatter_stage_{stage}.png'))
        plt.close()

        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Team_win', y='Rank_diff', data=df_stage)
        plt.title(f'Stage {stage}: Rank Difference by Match Outcome')
        plt.xlabel('Match Outcome (0=Lost, 1=Won)')
        plt.ylabel('Rank Difference')
        plt.savefig(os.path.join(country_plots_dir, f'rank_diff_by_match_outcome_box_stage_{stage}.png'))
        plt.close()

        # First-Stage Regression
        X = sm.add_constant(df_stage[['Rank_diff']])
        first_stage_model = sm.OLS(df_stage['Team_win'], X).fit()
        print(f'Stage {stage} First-Stage Regression Summary:')
        print(first_stage_model.summary())

        # F-Statistic for the instrument
        f_stat = first_stage_model.f_pvalue
        print(f'Stage {stage}: F-statistic of the first stage: {f_stat}')

        # Partial R-Squared
        partial_r2 = first_stage_model.rsquared
        print(f'Stage {stage}: Partial R-squared of Rank_diff: {partial_r2}')

        # Save regression summary to text file
        with open(os.path.join(country_plots_dir, f'first_stage_regression_summary_stage_{stage}.txt'), 'w') as f:
            f.write(first_stage_model.summary().as_text())

def plot_logistic_regression(df, x_column, y_column, stage, country_plots_dir):
    X = sm.add_constant(df[x_column])
    model = sm.Logit(df[y_column], X).fit()

    # Avoid SettingWithCopyWarning by creating a new DataFrame
    df_stage = df.copy()
    df_stage['pred_prob'] = model.predict(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_stage[x_column], y=df_stage[y_column], alpha=0.5)
    sns.lineplot(x=df_stage[x_column], y=df_stage['pred_prob'], color='red')
    plt.title(f'Stage {stage}: Logistic Regression: {x_column} vs Probability of Winning')
    plt.xlabel(x_column)
    plt.ylabel('Probability of Winning')
    plt.savefig(os.path.join(country_plots_dir, f'{x_column}_vs_probability_of_winning_logistic_regression_stage_{stage}.png'))
    plt.show()

def plot_avg_rank_by_stage_and_outcome(df, stage, country_plots_dir):
    avg_opponent_rank_df = df.groupby(['Team_win'])['Opponent_rank_prev'].mean().reset_index()
    avg_opponent_rank_df['Match Outcome'] = avg_opponent_rank_df['Team_win'].map({0: 'Lost', 1: 'Won'})

    plt.figure(figsize=(14, 8))
    sns.barplot(data=avg_opponent_rank_df, x='Match Outcome', y='Opponent_rank_prev', hue='Match Outcome', palette='muted', dodge=False)
    plt.title(f'Stage {stage}: Average Opponent Rank by Match Outcome')
    plt.xlabel('Match Outcome')
    plt.ylabel('Average Opponent Rank')
    plt.savefig(os.path.join(country_plots_dir, f"avg_opponent_rank_by_outcome_stage_{stage}.png"))
    plt.show()

if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'
    fixtures_df = load_csv_data(country, f'{cup}_fixtures.csv')
    standings_df = load_csv_data(country, 'league_standings.csv')
    stages_df = preprocess_data(fixtures_df, standings_df)

    # Save the processed DataFrame
    output_path = os.path.join(project_root, 'process_data', country, f'{cup}_processed.csv')
    stages_df.to_csv(output_path, index=False)

    # Ensure country-specific plots directory exists
    country_plots_dir = ensure_country_plots_dir(country)

    # Distribution of Team_rank
    plot_histogram_and_kde(stages_df, 'Team_rank', country_plots_dir)

    # Relationship Between Team_max_stage and Team_rank
    plot_box_by_stage(stages_df, 'Team_max_stage', 'Team_rank', country_plots_dir)
    plot_scatter_with_regression(stages_df, 'Team_max_stage', 'Team_rank', country_plots_dir)

    # Yearly Trends
    plot_line_by_year(stages_df, 'Year', 'Team_rank', 'Team_max_stage', country_plots_dir)

    # Effect of Participating in Stages
    plot_bar_by_stage(stages_df, 'Stage', 'Team_rank', country_plots_dir)

    # Correlation Heatmap
    plot_heatmap(stages_df, ['Team_rank', 'Team_max_stage', 'Rank_diff', 'Team_win', 'const'], country_plots_dir)

    # Plot avg rank by stage
    plot_avg_rank_by_stage(stages_df, country_plots_dir)

    # First-stage analysis by stage
    analyze_first_stage_by_stage(stages_df, country_plots_dir)

    # Logistic regression and bar plot by stage
    unique_stages = stages_df['Stage'].unique()
    for stage in unique_stages:
        df_stage = stages_df[stages_df['Stage'] == stage]
        plot_logistic_regression(df_stage, 'Opponent_rank_prev', 'Team_win', stage, country_plots_dir)
        plot_avg_rank_by_stage_and_outcome(df_stage, stage, country_plots_dir)
