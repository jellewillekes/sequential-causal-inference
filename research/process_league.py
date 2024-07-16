import os
import json
import pandas as pd
import yaml

from raw_data.loader import get_project_root, load_mappings_from_yaml, load_stages, load_league_mappings


def process_leagues(country):
    all_standings = []
    project_root = get_project_root()
    leagues = load_league_mappings(country)

    for league, details in leagues.items():
        print(league)
        if 'standings' in details['data_types']:
            for season in range(details['season_start'], details['season_end']):
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

    return pd.DataFrame(all_standings)


# Usage example:
country = 'Netherlands'
df_standings = process_leagues(country)

df = df_standings.sort_values(by=['Season', 'Division', 'Position'])

# Function to calculate the offsets for each division based on the number of teams in higher divisions
def calculate_offsets(df):
    offsets = {}
    for season in df['Season'].unique():
        season_data = df[df['Season'] == season]
        max_position = 0
        for division in sorted(season_data['Division'].unique()):
            offsets[(season, division)] = max_position
            max_position += season_data[season_data['Division'] == division]['Position'].max()
    return offsets

# Calculate the offsets for each division
offsets = calculate_offsets(df)

# Apply the offsets to calculate the NationalRank
df['Rank'] = df.apply(lambda row: row['Position'] + offsets[(row['Season'], row['Division'])], axis=1)

# Sort by 'Season' and 'NationalRank'
df = df.sort_values(by=['Season', 'Rank'])

# Reset index to clean up the DataFrame
df = df.reset_index(drop=True)

# Now, df has an additional column 'NationalRank' which represents the overall national ranking
print(df[['Season', 'Division', 'Position', 'Team', 'NationalRank']])
