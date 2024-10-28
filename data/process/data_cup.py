from utils.load import project_root, load_mappings_from_yaml
import os
import json
import pandas as pd


def process_season_fixtures(fixtures_data, season, stages):
    season_matches = []
    for fixture in fixtures_data['response']:
        round_name = fixture['league']['round']

        # Only proceed if the round is in the stages mapping
        if round_name not in stages:
            continue

        stage_number = stages[round_name]

        home_winner = fixture['teams']['home'].get('winner')
        away_winner = fixture['teams']['away'].get('winner')

        home_team_win = int(home_winner) if home_winner is not None else None
        away_team_win = int(away_winner) if away_winner is not None else None

        match_data_home = construct_fixtures_data(season, round_name, stage_number, fixture, 'home', home_team_win)
        match_data_away = construct_fixtures_data(season, round_name, stage_number, fixture, 'away', away_team_win)

        season_matches.append(match_data_home)
        season_matches.append(match_data_away)

    return season_matches


def construct_fixtures_data(season, round_name, stage_number, fixture, team_type, team_win):
    team_key = 'home' if team_type == 'home' else 'away'
    opponent_key = 'away' if team_type == 'home' else 'home'

    venue_name = fixture['fixture']['venue']['name']

    return {
        'year': season,
        'round': round_name,
        'stage': stage_number,
        'fixture_id': fixture['fixture']['id'],
        'fixture_date': fixture['fixture']['date'],
        'team_name': fixture['teams'][team_key]['name'],
        'team_id': fixture['teams'][team_key]['id'],
        'opponent_name': fixture['teams'][opponent_key]['name'],
        'opponent_id': fixture['teams'][opponent_key]['id'],
        'team_win': team_win,
        'team_home': team_type,
        'fixture_length': fixture['fixture']['status']['elapsed'],
        'fixture_location': venue_name,
    }


def save_to_csv(df, country, cup):
    project_root_path = project_root()
    save_path = os.path.join(project_root_path, 'data', 'process', country, f'{cup}_fixtures.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


def construct_cup_data(country, cup):
    mappings_file = os.path.join('settings', f'mapping_{country.lower()}.yaml')
    mappings = load_mappings_from_yaml(mappings_file)
    league_info = mappings.get(cup)
    if not league_info:
        raise ValueError(f"No mapping found for cup: {cup} in country: {country}")

    season_start = league_info['season_start']
    season_end = league_info['season_end']
    stages = league_info['rounds']

    all_fixtures = []
    project_root_path = project_root()

    for season in range(season_start, season_end + 1):
        season_path = os.path.join(project_root_path, 'data', 'raw', country, cup, str(season), 'fixtures_data.json')
        if os.path.isfile(season_path):
            with open(season_path, 'r') as file:
                fixtures_data = json.load(file)
                all_fixtures.extend(process_season_fixtures(fixtures_data, season, stages))

    cup_fixtures = pd.DataFrame(all_fixtures)

    # Remove fixtures without a winner, then there will be a replay.
    cup_fixtures = cup_fixtures.dropna(subset=['team_win'])

    # Only keep fixtures that are random based on mapping
    start_round = mappings[cup]['start_round']
    cup_fixtures = cup_fixtures[cup_fixtures['stage'] <= start_round]

    save_to_csv(cup_fixtures, country, cup)
    print(cup_fixtures['year'].max())


if __name__ == "__main__":
    country = 'Portugal'
    cup = 'Taca_de_Portugal'
    construct_cup_data(country, cup)
