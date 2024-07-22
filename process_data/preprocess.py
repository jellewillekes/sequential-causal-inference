import os
import pandas as pd
import numpy as np
from raw_data.loader import get_project_root

from fuzzywuzzy import process, fuzz
import Levenshtein


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Ensure plots directory exists
def ensure_country_plots_dir(country):
    country_plots_dir = os.path.join("plots", country)
    os.makedirs(country_plots_dir, exist_ok=True)
    return country_plots_dir


# Function for loading data
def load_csv_data(country, file_name):
    file_path = os.path.join(get_project_root(), 'process_data', country, file_name)
    return pd.read_csv(file_path)


def set_non_league_rank(df, divisions=4):
    latest_year = df['year'].max()
    total_teams = df[df['year'] == latest_year]['team_id'].nunique()

    # Calculate the rank for non-league teams
    non_league_rank = total_teams + (total_teams // divisions)

    # Set the national rank for non-league teams
    df['team_rank'] = df['team_rank'].fillna(non_league_rank)
    df['team_rank_prev'] = df['team_rank_prev'].fillna(non_league_rank)
    df['opponent_rank_prev'] = df['opponent_rank_prev'].fillna(non_league_rank)

    return df


def preprocess_match_data(fixtures_df, standings_df):
    # Filter and preprocess data
    standings_df = standings_df[standings_df['year'] > 2010].copy()
    standings_df['prev_year'] = standings_df['year'] + 1

    # Merge operations
    stages_df = (fixtures_df[fixtures_df['year'] > 2010]
                 .merge(standings_df[['prev_year', 'team_id', 'national_rank']],
                        left_on=['year', 'opponent_id'],
                        right_on=['prev_year', 'team_id'],
                        how='left',
                        suffixes=('', '_opponent'))
                 .rename(columns={'national_rank': 'opponent_rank_prev'})
                 .drop(columns=['prev_year', 'team_id_opponent'])
                 .merge(standings_df[['prev_year', 'division', 'team_id', 'national_rank']],
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

    stages_df = set_non_league_rank(stages_df)

    # Handle missing values and create new columns
    stages_df = (stages_df
                 .assign(rank_diff=lambda df: df['opponent_rank_prev'] - df['team_rank_prev'],
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


def merge_financial_data(match_df, financial_df):
    def get_base_name(team_name):
        if ' II' in team_name:
            return team_name.replace(' II', ''), ' II'
        elif ' 2' in team_name:
            return team_name.replace(' 2', ''), ' 2'
        elif ' B' in team_name:
            return team_name.replace(' B', ''), ' B'
        else:
            return team_name, ''

    def get_best_match(row, choices):
        team_name = row['team_name']
        base_name, suffix = get_base_name(team_name)

        # Filter choices to those with matching suffix
        if suffix:
            filtered_choices = [choice for choice in choices
                                if suffix in choice
                                and base_name in choice]
        else:
            filtered_choices = [choice for choice in choices if
                                not any(suffix in choice for suffix in [' II', ' 2', ' B'])]

        if not filtered_choices:
            return None, None

        # Use Levenshtein to find the best match
        best_match = None
        best_score = 0
        for choice in filtered_choices:
            score = Levenshtein.ratio(team_name, choice)
            if score > best_score:
                best_match = choice
                best_score = score

        if best_score >= 0.76:  # Adjust the threshold if needed
            return best_match, best_score
        else:
            return None, None

    # Apply the get_best_match function and split the result into two columns
    match_df[['best_match', 'match_ratio']] = match_df.apply(lambda row: get_best_match(row, financial_df['team_name']),
                                                             axis=1,
                                                             result_type='expand')

    # Filter out rows where match_ratio is None
    match_df = match_df[match_df['match_ratio'].notna()]

    # Merge the dataframes on 'Year' and 'best_match'
    merged = pd.merge(
        match_df,
        financial_df,
        left_on=['year', 'best_match'],
        right_on=['year', 'team_name'],
        suffixes=('_fix', '_fin'),
        how='left'
    )

    # Keep original names from match_df
    suffixes_columns = [col for col in match_df.columns if col in financial_df.columns]
    rename_columns = {col + '_fix': col for col in suffixes_columns}
    merged.rename(columns=rename_columns, inplace=True)

    return merged


def preprocess_data(country, cup):
    fixtures_df = load_csv_data(country, f'{cup}_fixtures.csv')
    standings_df = load_csv_data(country, 'league_standings.csv')
    match_df = preprocess_match_data(fixtures_df, standings_df)

    distances_df = load_csv_data(country, f'{cup}_distances.csv')
    match_df = merge_distance_data(match_df, distances_df)

    financial_df = load_csv_data(country, f'league_financial_data.csv')
    match_df = merge_financial_data(match_df, financial_df)

    # Convert `team_home` to a binary variable
    match_df['team_home'] = match_df['team_home'].apply(lambda x: 1 if x == 'home' else 0)

    # Convert `fixture_length` to a binary variable
    match_df['extra_time'] = match_df['fixture_length'].apply(lambda x: 1 if x > 90 else 0)

    return match_df


def check_names(df):
    # Analysis on Match Ratios:
    name_matches = df[['fixture_id', 'team_name', 'best_match', 'match_ratio', 'team_name_fin']]

    tricky_score = name_matches[name_matches['match_ratio'] < 0.90]
    tricky_score = tricky_score.sort_values(by=['team_name', 'best_match'])

    # Drop duplicates considering the specified columns: team_name, best_match, match_ratio, and team_name_fin
    tricky_score = tricky_score.drop_duplicates(subset=['team_name', 'best_match', 'match_ratio', 'team_name_fin'])

    print(tricky_score.head())


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    stages_df = preprocess_data(country, cup)

    # Save the processed DataFrame
    output_path = os.path.join(get_project_root(), 'process_data', country, f'{cup}_processed.csv')
    stages_df.to_csv(output_path, index=False)
