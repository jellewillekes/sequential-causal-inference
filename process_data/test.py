import pandas as pd
import os
from raw_data.loader import get_project_root


def load_csv_data(country, file_name):
    project_root = get_project_root()
    file_path = os.path.join(project_root, 'process_data', country, file_name)
    return pd.read_csv(file_path)


# Usage
country = 'Netherlands'
fixtures_df = load_csv_data(country, 'KNVB_Beker_fixtures.csv')
standings_df = load_csv_data(country, 'league_standings.csv')


def get_opponent_previous_season_rank(fixtures_df, standings_df):
    # Create a copy of the standings DataFrame and adjust the season to get the previous year's rankings
    previous_season_standings = standings_df.copy()
    previous_season_standings['Season'] += 1

    # Merge the fixtures data with the previous season's standings based on the opponent's name and year
    merged_df = fixtures_df.merge(
        previous_season_standings,
        left_on=['Opponent_name', 'Year'],
        right_on=['Team', 'Season'],
        how='left',  # Ensures we keep all fixtures even if no matching standing is found
        suffixes=('', '_Prev')
    )

    # Select the relevant columns, including the opponent's national rank from the previous season
    opponent_ranks_df = merged_df[['Year', 'Stage', 'Team_name', 'Team_id', 'Opponent_name', 'NationalRank_Prev']]
    opponent_ranks_df.rename(columns={'NationalRank_Prev': 'Opponent_Rank_Previous_Season'}, inplace=True)

    return opponent_ranks_df



def get_team_league_ranks(standings_df):
    # Select relevant columns
    result_df = standings_df[['Season', 'Team', 'Team_id', 'NationalRank']]
    result_df.rename(columns={'NationalRank': 'League_Rank'}, inplace=True)

    return result_df


# Run functions to get the desired datasets
team_league_ranks_df = get_team_league_ranks(standings_df)
opponent_ranks_df = get_opponent_previous_season_rank(fixtures_df, standings_df)
