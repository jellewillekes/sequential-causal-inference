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


def request_data(country, competition_name, competition_id, season, data_type, request_counter, start_time):
    if request_counter >= 10:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            print(f"Rate limit approached, sleeping for {60 - elapsed_time} seconds")
            time.sleep(60 - elapsed_time)
        request_counter = 0
        start_time = time.time()

    # Update directory path to include country and competition name
    directory_path = os.path.join(get_project_root(), 'raw_data', country, competition_name.replace(" ", "_"),
                                  str(season))
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, f"{data_type}_data.json")

    if data_type == "standings":
        api_endpoint = f"/standings?league={competition_id}&season={season}"
    elif data_type == "fixtures":
        api_endpoint = f"/fixtures?league={competition_id}&season={season}"

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
        print(f"Loaded data from existing file for {country}, {competition_name} {season}, {data_type}.")
    else:
        conn = http.client.HTTPSConnection("v3.football.api-sports.io")
        headers = {
            'x-rapidapi-host': "v3.football.api-sports.io",
            'x-rapidapi-key': "483cf201220068a29dbebab0fed58226"
        }
        conn.request("GET", api_endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()

        data_dict = json.loads(data.decode("utf-8"))

        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=4)
        print(f"Requested new data and saved to file for {country}, {competition_name} {season}, {data_type}.")

    request_counter += 1
    return data_dict, request_counter, start_time


if __name__ == "__main__":
    mappings = load_mappings_from_yaml('settings/mapping_competitions.yaml')

    request_counter = 0
    start_time = time.time()

    # Specify the country to process
    specified_country = "Germany"  # Change as needed

    if specified_country in mappings:
        country_info = mappings[specified_country]
        for competition_name, info in country_info.items():
            competition_id = info['id']
            season_start = info['season_start']
            season_end = info['season_end']
            data_types = info['data_types']

            for season in range(season_start, season_end + 1):
                for data_type in data_types:
                    print(
                        f"Processing {specified_country}, {competition_name} for the {season} season, requesting {data_type}.")
                    result, request_counter, start_time = request_data(specified_country, competition_name,
                                                                       competition_id, season, data_type,
                                                                       request_counter, start_time)
    else:
        print(f"No data available for {specified_country}")
