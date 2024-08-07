import os
import pandas as pd
import Levenshtein

from utils.load import project_root, load_csv


def set_non_league_rank(team_data: pd.DataFrame, divisions: int = 4):
    """
    Set the national rank for non-league teams in the dataframe
    by filling missing rankings based on the total number of teams
    in the latest year and specified divisions.

    Returns:
    - pd.DataFrame: Dataframe with updated team rankings.
    """
    latest_year = team_data['year'].max()
    total_teams_in_latest_year = team_data[team_data['year'] == latest_year]['team_id'].nunique()
    non_league_rank = total_teams_in_latest_year + (total_teams_in_latest_year // divisions)

    team_data['team_rank'] = team_data['team_rank'].fillna(non_league_rank)
    team_data['team_rank_prev'] = team_data['team_rank_prev'].fillna(non_league_rank)
    team_data['opponent_rank_prev'] = team_data['opponent_rank_prev'].fillna(non_league_rank)

    return team_data


def merge_cup_and_league_data(cup_fixtures: pd.DataFrame, league_standings: pd.DataFrame):
    """
    Merge cup fixtures with current year's national rank of the team (`team_rank`),
    previous year's national rank (`team_rank_prev`) and division (`division`)
    of the team, current year's national rank of the team (`team_rank`).

    Sets national ranks for non-league (non-professional) teams,
    adds rank differences between team and opponent to cup fixtures.

    Returns:
    - pd.DataFrame: Merged dataframe with cup and league data.
    """
    league_standings = league_standings[league_standings['year'] > 2010].copy()
    league_standings['prev_year'] = league_standings['year'] + 1

    merged_cup_fixtures = (cup_fixtures[cup_fixtures['year'] > 2010]
                           # Merge opponent's previous year national rank
                           .merge(league_standings[['prev_year', 'team_id', 'national_rank']],
                                  left_on=['year', 'opponent_id'],
                                  right_on=['prev_year', 'team_id'],
                                  how='left',
                                  suffixes=('', '_opponent'))
                           .rename(columns={'national_rank': 'opponent_rank_prev'})
                           .drop(columns=['prev_year', 'team_id_opponent'])
                           # Merge team's previous year national rank and division
                           .merge(league_standings[['prev_year', 'division', 'team_id', 'national_rank']],
                                  left_on=['year', 'team_id'],
                                  right_on=['prev_year', 'team_id'],
                                  how='left')
                           .rename(columns={'national_rank': 'team_rank_prev'})
                           .drop(columns=['prev_year'])
                           # Merge team's current year national rank
                           .merge(league_standings[['year', 'team_id', 'national_rank']],
                                  left_on=['year', 'team_id'],
                                  right_on=['year', 'team_id'],
                                  how='left')
                           .rename(columns={'national_rank': 'team_rank'}))

    merged_cup_fixtures = set_non_league_rank(merged_cup_fixtures)

    merged_cup_fixtures = (merged_cup_fixtures
                           .assign(rank_diff=lambda df: df['opponent_rank_prev'] - df['team_rank_prev'],
                                   const=1)
                           .dropna(subset=['stage'])
                           .assign(stage=lambda df: df['stage'].astype(int))
                           .query('2012 <= year <= 2022')
                           .assign(team_better=lambda df: (df['team_rank'] < df['opponent_rank_prev']).astype(int)))

    max_stage_data = cup_fixtures.groupby(['year', 'team_id'])['stage'].max().reset_index()
    max_stage_data.rename(columns={'stage': 'team_max_stage'}, inplace=True)
    merged_data = pd.merge(merged_cup_fixtures, max_stage_data, on=['year', 'team_id'], how='left')

    return merged_data


def merge_with_next_fixture_data(cup_fixtures, league_fixtures):
    """
    Merge cup fixtures dataframe with next league fixture data,
    adding information about the next opponent, fixture date,
    days until next fixture, match result, and points.

    Returns:
    - pd.DataFrame: Merged dataframe with next fixture data.
    """
    cup_fixtures['fixture_date'] = pd.to_datetime(cup_fixtures['fixture_date'])
    league_fixtures['fixture_date'] = pd.to_datetime(league_fixtures['fixture_date'])

    cup_fixtures = cup_fixtures.sort_values(by=['team_id', 'fixture_date']).reset_index(drop=True)
    league_fixtures = league_fixtures.sort_values(by=['team_id', 'fixture_date']).reset_index(drop=True)

    result_rows = []

    for index, row in cup_fixtures.iterrows():
        team_id = row['team_id']
        fixture_date = row['fixture_date']

        next_matches = league_fixtures[
            (league_fixtures['team_id'] == team_id) & (league_fixtures['fixture_date'] > fixture_date)]

        if not next_matches.empty:
            next_match = next_matches.iloc[0]
            next_fixture_date = next_match['fixture_date']
            next_opponent_name = next_match['opponent_name']
            next_fixture_days = (next_fixture_date - fixture_date).days
            next_team_win = next_match['team_win']
            next_team_points = next_match['team_points_match']

            result_rows.append({
                **row,
                'next_fixture_date': next_fixture_date,
                'next_opponent_name': next_opponent_name,
                'next_fixture_days': next_fixture_days,
                'next_team_win': next_team_win,
                'next_team_points': next_team_points
            })

    result = pd.DataFrame(result_rows)

    return result


def merge_with_distance_data(cup_fixtures, distance_data):
    """
    Merge cup fixtures dataframe with distance data, adding travel distance
    information for away matches.

    Returns:
    - pd.DataFrame: Merged dataframe with distance data.
    """
    cup_fixtures['distance'] = 0
    away_matches = cup_fixtures[cup_fixtures['team_home'] == 'away']

    distance_dict = {}
    for index, row in distance_data.iterrows():
        team1, team2, distance = row['team_name'], row['opponent_name'], row['distance']
        distance_dict[(team1, team2)] = distance
        distance_dict[(team2, team1)] = distance

    def lookup_distance(row):
        team = row['team_name']
        opponent = row['opponent_name']
        return distance_dict.get((team, opponent), None)

    away_matches['distance'] = away_matches.apply(lookup_distance, axis=1)
    cup_fixtures.update(away_matches[['distance']])

    return cup_fixtures


def merge_with_financial_data(cup_fixtures, financial_data):
    """
    Merge cup fixtures dataframe with financial data (squad size, mean age,
    foreign players, mean market value, total market value) using
    the best matching team names based on Levenshtein similarity scores as
    names of teams are not exactly similar based on data sources.

    Returns:
    - pd.DataFrame: Merged dataframe with financial data.
    """

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

        if suffix:
            filtered_choices = [choice for choice in choices if suffix in choice and base_name in choice]
        else:
            filtered_choices = [choice for choice in choices if
                                not any(suffix in choice for suffix in [' II', ' 2', ' B'])]

        if not filtered_choices:
            return None, None

        best_match = None
        best_score = 0
        for choice in filtered_choices:
            score = Levenshtein.ratio(team_name, choice)
            if score > best_score:
                best_match = choice
                best_score = score

        if best_score >= 0.76:
            return best_match, best_score
        else:
            return None, None

    cup_fixtures[['best_match', 'match_ratio']] = cup_fixtures.apply(
        lambda row: get_best_match(row, financial_data['team_name']),
        axis=1,
        result_type='expand')
    cup_fixtures = cup_fixtures[cup_fixtures['match_ratio'].notna()]

    merged_cup_fixtures = pd.merge(
        cup_fixtures,
        financial_data,
        left_on=['year', 'best_match'],
        right_on=['year', 'team_name'],
        suffixes=('_fix', '_fin'),
        how='left'
    )

    suffixes_columns = [col for col in cup_fixtures.columns if col in financial_data.columns]
    rename_columns = {col + '_fix': col for col in suffixes_columns}
    merged_cup_fixtures.rename(columns=rename_columns, inplace=True)

    return merged_cup_fixtures


def preprocess_data(country: str, cup: str):
    """
    Preprocess the data for a given country and cup by merging and enhancing data
    from various sources including cup fixtures, league standings, next fixtures,
    distances, and financial information.

    Returns:
    - pd.DataFrame: Preprocessed dataframe with combined data from various sources.
    """
    cup_fixtures = load_csv(os.path.join(project_root(), 'data', 'process_data', country, f'{cup}_fixtures.csv'))
    league_standings = load_csv(os.path.join(project_root(), 'data', 'process_data', country, 'league_standings.csv'))
    league_fixtures = load_csv(os.path.join(project_root(), 'data', 'process_data', country, 'league_fixtures.csv'))
    distance_data = load_csv(os.path.join(project_root(), 'data', 'process_data', country, f'{cup}_distances.csv'))
    financial_data = load_csv(
        os.path.join(project_root(), 'data', 'process_data', country, 'league_financial_data.csv'))

    merged_cup_fixtures = merge_cup_and_league_data(cup_fixtures, league_standings)
    merged_cup_fixtures = merge_with_next_fixture_data(merged_cup_fixtures, league_fixtures)
    merged_cup_fixtures = merge_with_distance_data(merged_cup_fixtures, distance_data)
    merged_cup_fixtures = merge_with_financial_data(merged_cup_fixtures, financial_data)

    merged_cup_fixtures['team_home'] = merged_cup_fixtures['team_home'].apply(lambda x: 1 if x == 'home' else 0)
    merged_cup_fixtures['extra_time'] = merged_cup_fixtures['fixture_length'].apply(lambda x: 1 if x > 90 else 0)

    return merged_cup_fixtures


def check_name_matches(data):
    """
    Analyze and print teams with low match ratios from Levenshtein matching
    to identify potential discrepancies in the merged financial data.
    """
    name_matches = data[['fixture_id', 'team_name', 'best_match', 'match_ratio', 'team_name_fin']]
    tricky_matches = name_matches[name_matches['match_ratio'] < 0.90]
    tricky_matches = tricky_matches.sort_values(by=['team_name', 'best_match'])
    tricky_matches = tricky_matches.drop_duplicates(subset=['team_name', 'best_match', 'match_ratio', 'team_name_fin'])

    print(tricky_matches.head())


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    processed_data = preprocess_data(country, cup)
    output_path = os.path.join(project_root(), 'data', 'process_data', country, f'{cup}_processed.csv')
    processed_data.to_csv(output_path, index=False)