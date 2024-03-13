import os
import http.client
import json
import time
import yaml


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mappings_from_yaml(filename):
    file_path = os.path.join(get_project_root(), filename)
    with open(file_path, 'r') as file:
        mappings = yaml.safe_load(file)
    return mappings


def load_stages(country):
    # Load the stage order from the YAML file
    with open(os.path.join(get_project_root(), 'settings', 'stages.yaml'), 'r') as file:
        stages_data = yaml.safe_load(file)
    return stages_data.get(country)


def request_data(league_name, league_id, season, request_counter, start_time):
    if request_counter >= 10:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            print(f"Rate limit approached, sleeping for {60 - elapsed_time} seconds")
            time.sleep(60 - elapsed_time)
        request_counter = 0
        start_time = time.time()

    # Adjusted to use the project root for directory paths
    directory_path = os.path.join(get_project_root(), 'raw_data', league_name, season)
    file_path = os.path.join(directory_path, 'league_data.json')

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
        print(f"Loaded data from existing file for {league_name} {season}.")
    else:
        conn = http.client.HTTPSConnection("v3.football.api-sports.io")
        headers = {
            'x-rapidapi-host': "v3.football.api-sports.io",
            'x-rapidapi-key': "483cf201220068a29dbebab0fed58226"  # Replace with your actual API key
        }
        conn.request("GET", f"/standings?league={league_id}&season={season}", headers=headers)
        res = conn.getresponse()
        data = res.read()

        data_dict = json.loads(data.decode("utf-8"))

        os.makedirs(directory_path, exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=4)
        print(f"Requested new data and saved to file for {league_name} {season}.")

    request_counter += 1
    return data_dict, request_counter, start_time


if __name__ == "__main__":
    mappings = load_mappings_from_yaml(os.path.join('settings', 'mapping_england.yaml'))

    request_counter = 0
    start_time = time.time()

    for league_name, league_id in mappings.items():
        for season in range(2010, 2023):
            if league_name in mappings:
                print(f"Processing {league_name} for the {season} season.")
                result, request_counter, start_time = request_data(league_name, str(league_id), str(season),
                                                                   request_counter, start_time)
            else:
                print(f"League not found in the mappings: {league_name}")
