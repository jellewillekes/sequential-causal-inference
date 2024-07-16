import os
import json
import pandas as pd
import yaml

# Assuming your custom loader functions are correctly defined elsewhere
from raw_data.loader import get_project_root, load_mappings_from_yaml, load_stages, load_league_mappings


def compile_standings_for_country(country):
    all_standings = []
    project_root = get_project_root()
    leagues = load_league_mappings(country)

    for league, details in leagues.items():
        if 'standings' in details['data_types']:
            for season in range(details['season_start'], details['season_end'] + 1):
                standings_path = os.path.join(project_root, 'raw_data', country, league, str(season),
                                              'standings_data.json')
                if os.path.isfile(standings_path):
                    with open(standings_path, 'r') as file:
                        standings_data = json.load(file)
                        for entry in standings_data['response'][0]['league']['standings'][0]:
                            standings_info = {
                                'League': league,
                                'Division': details['division'],
                                'Season': season,
                                'Position': entry['rank'],
                                'Team': entry['team']['name'],
                                'Points': entry['points'],
                                'Form': entry['form'],
                                'Played': entry['all']['played'],
                                'Win': entry['all']['win'],
                                'Draw': entry['all']['draw'],
                                'Lose': entry['all']['lose'],
                                'GoalsDiff': entry['goalsDiff'],
                                'GoalsFor': entry['all']['goals']['for'],
                                'GoalsAgainst': entry['all']['goals']['against'],
                            }
                            all_standings.append(standings_info)

    df_standings = pd.DataFrame(all_standings)
    return calculate_national_rank(df_standings)


def calculate_national_rank(df):
    """Adjusts team positions to calculate a national rank across divisions for each season."""
    # Calculate offsets for each division
    offsets = calculate_offsets(df)

    # Apply offsets to calculate National Rank
    df['NationalRank'] = df.apply(lambda row: row['Position'] + offsets[(row['Season'], row['Division'])], axis=1)

    # Sort DataFrame by 'Season' and 'NationalRank'
    df_sorted = df.sort_values(by=['Season', 'NationalRank']).reset_index(drop=True)
    return df_sorted


def calculate_offsets(df):
    """Calculate offsets based on the number of teams in higher divisions per season."""
    offsets = {}
    for season in df['Season'].unique():
        max_position = 0
        for division in sorted(df[df['Season'] == season]['Division'].unique()):
            offsets[(season, division)] = max_position
            max_position += df[(df['Season'] == season) & (df['Division'] == division)]['Position'].max()
    return offsets


# Main execution
country = 'Netherlands'
df_final = compile_standings_for_country(country)
print(df_final[['Season', 'Division', 'Position', 'Team', 'NationalRank']])
