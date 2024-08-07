import os
import http.client
import json
import time
import yaml
from collections import defaultdict
import ssl
import certifi


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mappings_from_yaml(filename):
    file_path = os.path.join(get_project_root(), filename)
    with open(file_path, 'r') as file:
        mappings = yaml.safe_load(file)
    return mappings


def load_league_mappings(country):
    with open(os.path.join(get_project_root(), 'settings', f'mapping_{country.lower()}.yaml'), 'r') as file:
        league_mappings = yaml.safe_load(file)
    return league_mappings.get(country)



def request_data(country, league_name, league_id, season, request_counter, start_time):
    if request_counter >= 10:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            print(f"Rate limit approached, sleeping for {60 - elapsed_time} seconds")
            time.sleep(60 - elapsed_time)
        request_counter = 0
        start_time = time.time()

    # Adjusted to use the project root for directory paths
    directory_path = os.path.join(get_project_root(), 'raw', country, league_name, str(season))
    file_path = os.path.join(directory_path, 'injuries_data.json')

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
        print(f"Loaded data from existing file for {country} {league_name} {season}.")
    else:
        context = ssl.create_default_context(cafile=certifi.where())
        conn = http.client.HTTPSConnection("v3.football.api-sports.io", context=context)
        headers = {
            'x-rapidapi-host': "v3.football.api-sports.io",
            'x-rapidapi-key': "483cf201220068a29dbebab0fed58226"  # Replace with your actual API key
        }
        conn.request("GET", f"/injuries?league={league_id}&season={season}", headers=headers)
        res = conn.getresponse()
        data = res.read()

        data_dict = json.loads(data.decode("utf-8"))

        os.makedirs(directory_path, exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=4)
        print(f"Requested new data and saved to file for {country} {league_name} {season}.")

    request_counter += 1
    return data_dict, request_counter, start_time


def aggregate_missed_fixtures_per_team(data):
    missed_fixtures_per_team = defaultdict(int)
    for entry in data:
        if entry['player']['type'] == "Missing Fixture":
            team_name = entry['team']['name']
            missed_fixtures_per_team[team_name] += 1
    return missed_fixtures_per_team


if __name__ == "__main__":
    country = 'Germany'

    mappings = load_mappings_from_yaml(os.path.join('settings', f'mapping_{country.lower()}.yaml'))

    request_counter = 0
    start_time = time.time()

    all_missed_fixtures_per_team_season = defaultdict(lambda: defaultdict(int))

    for league_name, config in mappings[country].items():
        league_id = config['id']

        for season in range(2010, 2023):
            print(f"Processing {league_name} for the {season} season.")
            result, request_counter, start_time = request_data(country, league_name, str(league_id), str(season),
                                                               request_counter, start_time)
            missed_fixtures_per_team = aggregate_missed_fixtures_per_team(result['response'])
            for team, count in missed_fixtures_per_team.items():
                all_missed_fixtures_per_team_season[team][season] += count

    # Print the total number of missed fixtures per team per season
    for team, seasons in all_missed_fixtures_per_team_season.items():
        for season, missed_count in seasons.items():
            print(f"{team} - {season}: {missed_count} missed fixtures")
