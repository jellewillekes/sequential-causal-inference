import os
import pandas as pd
from raw_data.loader import get_project_root

from fuzzywuzzy import process, fuzz


# Ensure plots directory exists
def ensure_country_plots_dir(country):
    country_plots_dir = os.path.join("plots", country)
    os.makedirs(country_plots_dir, exist_ok=True)
    return country_plots_dir


# Function for loading data
def load_csv_data(country, file_name):
    file_path = os.path.join(get_project_root(), 'process_data', country, file_name)
    return pd.read_csv(file_path)


def preprocess_match_data(fixtures_df, standings_df):
    # Filter and preprocess data
    standings_df = standings_df[standings_df['year'] > 2011].copy()
    standings_df['prev_year'] = standings_df['year'] + 1

    # Merge operations
    stages_df = (fixtures_df[fixtures_df['year'] > 2011]
                 .merge(standings_df[['division', 'prev_year', 'team_id', 'national_rank']],
                        left_on=['year', 'opponent_id'],
                        right_on=['prev_year', 'team_id'],
                        how='left',
                        suffixes=('', '_opponent'))
                 .rename(columns={'national_rank': 'opponent_rank_prev'})
                 .drop(columns=['prev_year', 'team_id_opponent'])
                 .merge(standings_df[['prev_year', 'team_id', 'national_rank']],
                        left_on=['year', 'team_id'],
                        right_on=['prev_year', 'team_id'],
                        how='left')
                 .rename(columns={'national_rank': 'team_rank_prev'})
                 .drop(columns=['prev_year'])
                 .merge(standings_df[['year', 'team_id', 'national_rank']],
                        left_on=['year', 'team_id'],
                        right_on=['year', 'team_id'],
                        how='left')
                 .rename(columns={'national_rank': 'team_rank'})
                 )

    # Handle missing values and create new columns
    stages_df = (stages_df
                 .assign(team_rank_prev=lambda df: df['team_rank_prev'].fillna(75),
                         opponent_rank_prev=lambda df: df['opponent_rank_prev'].fillna(75),
                         Rank_diff=lambda df: df['opponent_rank_prev'] - df['team_rank_prev'],
                         const=1)
                 .dropna(subset=['stage'])
                 .assign(stage=lambda df: df['stage'].astype(int))
                 .query('2012 <= year <= 2022'))

    # Add maximum stage reached by each team in each year
    max_stage_df = fixtures_df.groupby(['year', 'team_id'])['stage'].max().reset_index()
    max_stage_df.rename(columns={'stage': 'team_max_stage'}, inplace=True)

    # Merge max stage information into stages_df
    stages_df = pd.merge(stages_df, max_stage_df, on=['year', 'team_id'], how='left')

    return stages_df


def merge_distance_data(match_df, distances_df):
    # Create a new column 'Distance' in match_df, initializing with 0
    match_df['distance'] = 0

    # Filter rows where team_home is 'away'
    away_matches = match_df[match_df['team_home'] == 'away']

    # Create a dictionary for faster lookup of distances
    distance_dict = {}
    for index, row in distances_df.iterrows():
        team1, team2, distance = row['team_name'], row['opponent_name'], row['distance']
        # Add both combinations to the dictionary
        distance_dict[(team1, team2)] = distance
        distance_dict[(team2, team1)] = distance

    # Define a function to lookup distance
    def lookup_distance(row):
        team = row['team_name']
        opponent = row['opponent_name']
        return distance_dict.get((team, opponent), None)

    # Apply the function to the away_matches dataframe
    away_matches['distance'] = away_matches.apply(lookup_distance, axis=1)

    # Update the original match_df with the distances found for away matches
    match_df.update(away_matches[['distance']])

    return match_df


def preprocess_data(country, cup):
    fixtures_df = load_csv_data(country, f'{cup}_fixtures.csv')
    standings_df = load_csv_data(country, 'league_standings.csv')
    match_df = preprocess_match_data(fixtures_df, standings_df)

    distances_df = load_csv_data(country, f'{cup}_distances.csv')
    match_df = merge_distance_data(match_df, distances_df)

    financial_df = load_csv_data(country, f'league_financial_data.csv')

    def get_best_match(row, choices):
        match, score, index = process.extractOne(row['team_name'], choices)
        if score >= 70.5:
            return match, score
        else:
            return None, None

    # Apply the get_best_match function and split the result into two columns
    match_df[['best_match', 'match_ratio']] = match_df.apply(lambda row: get_best_match(row, financial_df['team_name']), axis=1, result_type='expand')

    # Filter out rows where match_ratio is None
    match_df = match_df[match_df['match_ratio'].notna()]

    # Merge the dataframes on 'Year' and 'best_match'
    merged = pd.merge(match_df, financial_df, left_on=['year', 'best_match'], right_on=['year', 'team_name'],
                      suffixes=('_fix', '_fin'), how='left')

    return merged


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    stages_df = preprocess_data(country, cup)

    # Save the processed DataFrame
    # output_path = os.path.join(project_root, 'process_data', country, f'{cup}_processed.csv')

    print(stages_df.head())
