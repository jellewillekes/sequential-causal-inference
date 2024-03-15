import os
import pandas as pd
from raw_data.loader import get_project_root
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.api import OLS


def load_csv_data(country, file_name):
    project_root = get_project_root()
    file_path = os.path.join(project_root, 'process_data', country, file_name)
    return pd.read_csv(file_path)


# Load data
country = 'England'
cup = 'FA_Cup'
fixtures_df = load_csv_data(country, f'{cup}_fixtures.csv')
standings_df = load_csv_data(country, 'league_standings.csv')

# Preprocess data
fixtures_df = fixtures_df[fixtures_df['Year'] > 2011]
standings_df = standings_df[standings_df['Year'] > 2011]
standings_df['AdjustedYear'] = standings_df['Year'] + 1

# Merge to include national rank and adjusted year
stages_df = pd.merge(fixtures_df, standings_df[['AdjustedYear', 'Team_id', 'NationalRank']],
                     left_on=['Year', 'Opponent_id'], right_on=['AdjustedYear', 'Team_id'], how='left',
                     suffixes=('', '_opponent'))

# Rename columns and drop NaN in national rank
stages_df.dropna(subset=['NationalRank'], inplace=True)
stages_df.rename(columns={'NationalRank': 'RankOpponent'}, inplace=True)

stages_df = pd.merge(stages_df, standings_df[['Year', 'Team_id', 'NationalRank']],
                     left_on=['Year', 'Team_id'], right_on=['Year', 'Team_id'], how='left', suffixes=('',
                                                                                                      '_z'))
stages_df.rename(columns={'NationalRank': 'Rank'}, inplace=True)
stages_df['Rank'] = stages_df['Rank'].fillna(75)
stages_df['Rank_diff'] = (stages_df['Rank'] > stages_df['RankOpponent']).astype(int)

stages_df['const'] = 1  # Add constant for regression

stages_df.dropna(subset=['Stage'], inplace=True)
stages_df['Stage'] = stages_df['Stage'].astype(int)

# Create ReachedRound variable for each round by assuming participation equals reaching that round
for round_number in range(1, max(stages_df['Stage'])):
    stages_df[f'ReachedRound{round_number}'] = stages_df['Stage'].apply(lambda x: 1 if x >= round_number else 0)

# Begin IV regression analysis for each round
results = {}

for round_number in range(1, max(stages_df['Stage'])):
    # Define the treatment (ReachedRound) and instrument (RankOpponent for the previous round)
    df_round = stages_df[stages_df['Stage'] == round_number].copy()
    if not df_round.empty:
        treatment = f'ReachedRound{round_number}'
        instrument = 'Rank_diff'
        df_round.dropna(subset=['Rank_diff'], inplace=True)
        df_round.dropna(subset=['Team_win'], inplace=True)
        # First Stage Regression: predict winning based on opponent's rank
        first_stage = IV2SLS(df_round['Team_win'], df_round[['const']], df_round[[instrument]]).fit()

        # Predicted values (probability of winning this round)
        stages_df[f'{treatment}_pred'] = first_stage.fittedvalues

        # Second Stage Regression: effect of reaching this round on league performance
        df_merged = pd.merge(stages_df, standings_df[['Team_id', 'Year', 'NationalRank']],
                             left_on=['Team_id', 'AdjustedYear'], right_on=['Team_id', 'Year'], how='left',
                             suffixes=('', '_regr'))
        df_merged.dropna(subset=[f'{treatment}_pred', 'Rank'], inplace=True)

        second_stage = OLS(df_merged['Rank'], df_merged[['const', f'{treatment}_pred']]).fit()
        results[f'Stage_{round_number}'] = second_stage.summary()

# Print results
for key, value in results.items():
    print(f"Results for {key}:")
    print(value)
    print("\n---\n")
