import os
import json
import pandas as pd
from utils.load import project_root, load_league_mappings


def process_standings_data(entry, league, division, season):
    return {
        'league': league,
        'division': division,
        'year': season,
        'position': entry['rank'],
        'team_name': entry['team']['name'],
        'team_id': entry['team']['id'],
        'points': entry['points'],
        'played': entry['all']['played'],
        'win': entry['all']['win'],
        'draw': entry['all']['draw'],
        'lose': entry['all']['lose'],
        'goals_diff': entry['goalsDiff'],
        'goals_for': entry['all']['goals']['for'],
        'goals_against': entry['all']['goals']['against'],
    }


def calculate_offsets(df):
    offsets = {}
    for Year in df['year'].unique():
        max_position = 0
        for division in sorted(df[df['year'] == Year]['division'].unique()):
            offsets[(Year, division)] = max_position
            max_position += df[(df['year'] == Year) & (df['division'] == division)]['position'].max()
    return offsets


def calculate_national_rank(df):
    df = df.sort_values(by=['year', 'division', 'position'])
    offsets = calculate_offsets(df)
    df['national_rank'] = df.apply(lambda row: row['position'] + offsets[(row['year'], row['division'])], axis=1)
    return df.sort_values(by=['year', 'national_rank']).reset_index(drop=True)


def save_to_csv(df, country, filename):
    save_path = os.path.join(project_root(), 'data', 'process', country, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


def compile_standings(country):
    leagues = load_league_mappings(country)
    all_standings = []

    for league, details in leagues.items():
        if 'standings' in details['data_types']:
            for season in range(details['season_start'], details['season_end'] + 1):
                standings_path = os.path.join(project_root(), 'raw_data', country, league, str(season),
                                              'standings_data.json')
                if os.path.isfile(standings_path):
                    with open(standings_path, 'r') as file:
                        standings_data = json.load(file)
                        if standings_data['response'] and standings_data['response'][0]['league']['standings']:
                            for entry in standings_data['response'][0]['league']['standings'][0]:
                                standings_info = process_standings_data(entry, league, details['division'], season)
                                all_standings.append(standings_info)
                        else:
                            print(f"No standings data available for {league} in season {season}")

    df_standings = pd.DataFrame(all_standings)
    df_final = calculate_national_rank(df_standings)
    save_to_csv(df_final, country, 'league_standings.csv')
    return df_final


def construct_match_data(season, round_name, fixture, team_type, team_win):
    team_key = 'home' if team_type == 'home' else 'away'
    opponent_key = 'away' if team_type == 'home' else 'home'

    return {
        'year': season,
        'fixture_date': fixture['fixture']['date'],
        'league': fixture['league']['name'],
        'round': round_name,
        'fixture_location': fixture['fixture']['venue']['city'],
        'team_name': fixture['teams'][team_key]['name'],
        'team_id': fixture['teams'][team_key]['id'],
        'opponent_name': fixture['teams'][opponent_key]['name'],
        'opponent_id': fixture['teams'][opponent_key]['id'],
        'team_win': team_win,
        'team_goals': fixture['goals'][team_key],
        'opponent_goals': fixture['goals'][opponent_key],
    }


def process_season_fixtures(fixtures_data, season):
    season_matches = []
    for fixture in fixtures_data['response']:
        round_name = fixture['league']['round']

        home_winner = fixture['teams']['home'].get('winner')
        away_winner = fixture['teams']['away'].get('winner')

        home_team_win = int(home_winner) if home_winner is not None else None
        away_team_win = int(away_winner) if away_winner is not None else None

        match_data_home = construct_match_data(season, round_name, fixture, 'home', home_team_win)
        match_data_away = construct_match_data(season, round_name, fixture, 'away', away_team_win)

        season_matches.append(match_data_home)
        season_matches.append(match_data_away)

    return season_matches


def compile_fixtures(country):
    leagues = load_league_mappings(country)
    all_fixtures = []

    for league, details in leagues.items():
        if 'fixtures' in details['data_types'] and details['division'] != 'NaN':
            for season in range(details['season_start'], details['season_end'] + 1):
                fixtures_path = os.path.join(project_root(), 'raw_data', country, league, str(season),
                                             'fixtures_data.json')
                if os.path.isfile(fixtures_path):
                    with open(fixtures_path, 'r') as file:
                        fixtures_data = json.load(file)
                        all_fixtures.extend(process_season_fixtures(fixtures_data, season))

    df_fixtures = pd.DataFrame(all_fixtures)
    df_fixtures['team_points_match'] = df_fixtures['team_win'].apply(
        lambda x: 3 if x == 1 else (1 if pd.isna(x) else 0)
    )

    save_to_csv(df_fixtures, country, 'league_fixtures.csv')
    return df_fixtures


def construct_league_data(country):
    fixtures_final = compile_fixtures(country)
    standings_final = compile_standings(country)
    print(fixtures_final.head())
    print(standings_final.head())


# Usage example
if __name__ == "__main__":
    country = 'England'
    construct_league_data(country)
