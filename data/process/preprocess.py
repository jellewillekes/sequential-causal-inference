import os
import pandas as pd

from datetime import timedelta

from utils.load import project_root, load_csv


def set_non_league_rank(team_data: pd.DataFrame, divisions: int = 4):
    """
    Set the national rank for non-league teams in the dataframe
    by filling missing rankings based on the total number of teams
    in the latest year and specified divisions.

    Returns:
    - pd.DataFrame: Dataframe with updated team rankings.
    """
    divisions = team_data['team_division'].max()
    latest_year = team_data['year'].max()
    total_teams_in_latest_year = team_data[team_data['year'] == latest_year]['team_id'].nunique()
    non_national_rank = total_teams_in_latest_year + (total_teams_in_latest_year // divisions)

    team_data['team_rank'] = team_data['team_rank'].fillna(non_national_rank)
    team_data['team_rank_prev'] = team_data['team_rank_prev'].fillna(non_national_rank)
    team_data['opponent_rank_prev'] = team_data['opponent_rank_prev'].fillna(non_national_rank)

    total_teams_lowest_division = \
        team_data[(team_data['year'] == latest_year) & (team_data['team_division'] == divisions)]['team_id'].nunique()
    non_league_rank = total_teams_lowest_division // 2

    team_data['team_league_rank'] = team_data['team_league_rank'].fillna(non_league_rank)
    team_data['team_league_rank_prev'] = team_data['team_league_rank_prev'].fillna(non_league_rank)
    team_data['opponent_league_rank_prev'] = team_data['opponent_league_rank_prev'].fillna(non_league_rank)

    team_data['team_division'] = team_data['team_division'].apply(lambda x: divisions + 1 if pd.isna(x) else x)
    team_data['opponent_division'] = team_data['opponent_division'].apply(lambda x: divisions + 1 if pd.isna(x) else x)

    return team_data


def find_next_cup_round(winning_team_id, current_round_date, cup_fixtures):
    """
    Find the next cup round date for the winning team based on their cup fixtures.
    """
    next_round_fixtures = cup_fixtures[
        (cup_fixtures['team_id'] == winning_team_id) &
        (cup_fixtures['fixture_date'] > current_round_date)
        ].sort_values(by='fixture_date')

    if not next_round_fixtures.empty:
        return next_round_fixtures.iloc[0]['fixture_date']
    return None


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
    league_standings = league_standings.rename(columns={'position': 'league_rank'})

    merged_cup_fixtures = (cup_fixtures[cup_fixtures['year'] > 2010]
                           # Merge opponent's previous year national rank
                           .merge(league_standings[['prev_year', 'team_id', 'national_rank', 'league_rank',
                                                    'division']],
                                  left_on=['year', 'opponent_id'],
                                  right_on=['prev_year', 'team_id'],
                                  how='left',
                                  suffixes=('', '_opponent'))
                           .rename(columns={'national_rank': 'opponent_rank_prev',
                                            'league_rank': 'opponent_league_rank_prev',
                                            'division': 'opponent_division'})
                           .drop(columns=['prev_year', 'team_id_opponent'])
                           # Merge team's previous year national rank and division
                           .merge(
        league_standings[['prev_year', 'division', 'team_id', 'national_rank', 'league_rank']],
        left_on=['year', 'team_id'],
        right_on=['prev_year', 'team_id'],
        how='left')
                           .rename(columns={'national_rank': 'team_rank_prev',
                                            'league_rank': 'team_league_rank_prev',
                                            'division': 'team_division'})
                           .drop(columns=['prev_year'])
                           # Merge team's current year national rank
                           .merge(league_standings[['year', 'team_id', 'national_rank', 'league_rank']],
                                  left_on=['year', 'team_id'],
                                  right_on=['year', 'team_id'],
                                  how='left')
                           .rename(columns={'national_rank': 'team_rank',
                                            'league_rank': 'team_league_rank'}))

    merged_cup_fixtures = set_non_league_rank(merged_cup_fixtures)

    merged_cup_fixtures = (merged_cup_fixtures
                           .assign(rank_diff=lambda df: df['opponent_rank_prev'] - df['team_rank_prev'],
                                   team_rank_diff=lambda df: df['team_rank'] - df['team_rank_prev'])
                           .dropna(subset=['stage'])
                           .assign(stage=lambda df: df['stage'].astype(int))
                           .query('2012 <= year <= 2023')
                           .assign(team_better=lambda df: (df['team_rank'] < df['opponent_rank_prev']).astype(int)))

    max_stage_data = cup_fixtures.groupby(['year', 'team_id'])['stage'].max().reset_index()
    max_stage_data.rename(columns={'stage': 'team_max_stage'}, inplace=True)
    merged_data = pd.merge(merged_cup_fixtures, max_stage_data, on=['year', 'team_id'], how='left')

    return merged_data


def merge_with_next_fixture_data(cup_fixtures, league_fixtures):
    # Ensure that dates are in datetime format
    cup_fixtures['fixture_date'] = pd.to_datetime(cup_fixtures['fixture_date'])
    league_fixtures['fixture_date'] = pd.to_datetime(league_fixtures['fixture_date'])

    result_rows = []

    def find_next_cup_round(team_id, current_fixture_date, cup_fixtures):
        next_cup_fixtures = cup_fixtures[
            (cup_fixtures['team_id'] == team_id) &
            (cup_fixtures['fixture_date'] > current_fixture_date)
        ].sort_values(by='fixture_date')

        if not next_cup_fixtures.empty:
            return next_cup_fixtures.iloc[0]['fixture_date']
        else:
            return None

    fixture_groups = cup_fixtures.groupby('fixture_id')

    for fixture_id, fixture_data in fixture_groups:
        fixture_date = fixture_data.iloc[0]['fixture_date']
        teams_in_fixture = fixture_data['team_id'].unique()
        winning_team_id = fixture_data[fixture_data['team_win'] == 1]['team_id'].values[0]

        for team_id in teams_in_fixture:
            next_matches_after_round_t = league_fixtures[
                (league_fixtures['team_id'] == team_id) &
                (league_fixtures['fixture_date'] > fixture_date)
            ]

            if not next_matches_after_round_t.empty:
                next_match_after_t = next_matches_after_round_t.iloc[0]
                next_fixture_date_after_t = next_match_after_t['fixture_date']
                next_fixture_days_after_t = (next_fixture_date_after_t - fixture_date).days
                next_team_points_after_t = next_match_after_t['team_points_match']
            else:
                next_fixture_date_after_t = None
                next_fixture_days_after_t = None
                next_team_points_after_t = None

            next_fixture_date_after_t1 = None
            next_fixture_days_after_t1 = None
            next_team_points_after_t1 = None

            if team_id == winning_team_id:
                next_cup_round_date = find_next_cup_round(team_id, fixture_date, cup_fixtures)
            else:
                next_cup_round_date = find_next_cup_round(winning_team_id, fixture_date, cup_fixtures)
                # Check if next_cup_round_date is not None before subtracting timedelta
                if next_cup_round_date is not None:
                    next_cup_round_date = next_cup_round_date - timedelta(days=0)
                else:
                    print(f"Next cup round date is None for team_id: {team_id} and fixture_date: {fixture_date}")

            if next_cup_round_date is not None:
                next_matches_after_round_t1 = league_fixtures[
                    (league_fixtures['team_id'] == team_id) &
                    (league_fixtures['fixture_date'] > next_cup_round_date)
                ]

                if not next_matches_after_round_t1.empty:
                    next_match_after_t1 = next_matches_after_round_t1.iloc[0]
                    next_fixture_date_after_t1 = next_match_after_t1['fixture_date']
                    next_fixture_days_after_t1 = (next_fixture_date_after_t1 - next_cup_round_date).days
                    next_team_points_after_t1 = next_match_after_t1['team_points_match']

            result_rows.append({
                'fixture_id': fixture_id,
                'team_id': team_id,
                'next_fixture_date_round': next_fixture_date_after_t,
                'next_fixture_days_round': next_fixture_days_after_t,
                'next_team_points_round': next_team_points_after_t,
                'next_fixture_date_round_plus': next_fixture_date_after_t1,
                'next_fixture_days_round_plus': next_fixture_days_after_t1,
                'next_team_points_round_plus': next_team_points_after_t1
            })

    result_df = pd.DataFrame(result_rows)
    merged_cup_fixtures = cup_fixtures.merge(result_df, on=['fixture_id', 'team_id'], how='left')

    return merged_cup_fixtures


def merge_with_distance_data(cup_fixtures, distance_data):
    """
    Merge cup fixtures dataframe with distance data, adding travel distance
    information for away matches.

    Returns:
    - pd.DataFrame: Merged dataframe with distance data.
    """
    cup_fixtures['distance'] = 0
    away_matches = cup_fixtures[cup_fixtures['team_home'] == 'away'].copy()

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
    cup_fixtures.loc[away_matches.index, 'distance'] = away_matches['distance']

    return cup_fixtures


def merge_with_financial_data(cup_fixtures, financial_data, team_mapping):
    """
    Merge cup fixtures dataframe with financial data using a custom mapping
    of team names. Custom mappings can be found in settings folder.

    Returns:
    - pd.DataFrame: Merged dataframe with financial data.
    """
    cup_fixtures = cup_fixtures.merge(
        team_mapping,
        left_on='team_name',
        right_on='cup_name',
        how='left'
    )

    cup_fixtures.drop(columns=['cup_name'], inplace=True)

    merged_cup_fixtures = pd.merge(
        cup_fixtures,
        financial_data,
        left_on=['year', 'financial_name'],
        right_on=['year', 'team_name'],
        how='left',
        suffixes=('', '_fin')
    )

    merged_cup_fixtures.drop(columns=['financial_name', 'team_name_fin'], inplace=True)

    return merged_cup_fixtures


def preprocess_data(country: str, cup: str):
    """
    Preprocess the data for a given country and cup by merging and enhancing data
    from various sources including cup fixtures, league standings, next fixtures,
    distances, and financial information.

    Returns:
    - pd.DataFrame: Preprocessed dataframe with combined data from various sources.
    """
    cup_fixtures = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_fixtures.csv'))
    league_standings = load_csv(os.path.join(project_root(), 'data', 'process', country, 'league_standings.csv'))
    league_fixtures = load_csv(os.path.join(project_root(), 'data', 'process', country, 'league_fixtures.csv'))
    distance_data = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_distance_data.csv'))
    financial_data = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_financial_data.csv'))
    team_mapping = load_csv(os.path.join(project_root(), 'settings', country, f'{cup}_team_mapping.csv'))

    merged_cup_fixtures = merge_cup_and_league_data(cup_fixtures, league_standings)
    merged_cup_fixtures = merge_with_next_fixture_data(merged_cup_fixtures, league_fixtures)
    merged_cup_fixtures = merge_with_distance_data(merged_cup_fixtures, distance_data)
    merged_cup_fixtures = merge_with_financial_data(merged_cup_fixtures, financial_data, team_mapping)

    merged_cup_fixtures['team_home'] = merged_cup_fixtures['team_home'].apply(lambda x: 1 if x == 'home' else 0)
    merged_cup_fixtures['extra_time'] = merged_cup_fixtures['fixture_length'].apply(lambda x: 1 if x > 90 else 0)

    # Save the final preprocessed data to a CSV file
    output_path = os.path.join(project_root(), 'data', 'process', country, f'{cup}_processed.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_cup_fixtures.to_csv(output_path, index=False)

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
    country = 'England'
    cup = 'FA_Cup'
    processed_data = preprocess_data(country, cup)

    print(processed_data)
