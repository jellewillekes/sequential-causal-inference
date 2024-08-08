import os
import http.client
import json
import time
import logging
import ssl
import certifi
from utils.load import load_mappings_from_yaml, load_api_key, project_root

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def request_data(country, league_name, league_id, season, request_counter, start_time, api_key):
    directory_path = os.path.join(project_root(), 'data', 'raw', country, league_name, season)
    file_path = os.path.join(directory_path, 'league_data.json')

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
        logging.info(f"Loaded data from existing file for {league_name} {season}.")
        return data_dict, request_counter, start_time, True
    else:
        # Rate limiting
        if request_counter >= 10:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                logging.info(f"Rate limit approached, sleeping for {60 - elapsed_time} seconds")
                time.sleep(60 - elapsed_time)
            request_counter = 0
            start_time = time.time()

        try:
            context = ssl.create_default_context(cafile=certifi.where())
            conn = http.client.HTTPSConnection("v3.football.api-sports.io", context=context)
            headers = {
                'x-rapidapi-host': "v3.football.api-sports.io",
                'x-rapidapi-key': api_key
            }
            conn.request("GET", f"/standings?league={league_id}&season={season}", headers=headers)
            res = conn.getresponse()
            data = res.read()

            data_dict = json.loads(data.decode("utf-8"))

            # Check for errors in the response
            if data_dict.get("errors"):
                logging.error(f"Error in response for {league_name} {season}: {data_dict['errors']}")
                return None, request_counter, start_time, False

            # Check if the response is empty
            if not data_dict['response']:
                logging.info(f"No data in response for {league_name} {season}.")
                return None, request_counter, start_time, True

            os.makedirs(directory_path, exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump(data_dict, file, indent=4)
            logging.info(f"Requested new data and saved to file for {league_name} {season}.")

            request_counter += 1
        except Exception as e:
            logging.error(f"Error requesting data for {league_name} {season}: {e}")
            return None, request_counter, start_time, False

        return data_dict, request_counter, start_time, True


def request_raw_data(country):
    mappings_file = os.path.join('settings', f'mapping_{country.lower()}.yaml')
    mappings = load_mappings_from_yaml(mappings_file)
    api_key = load_api_key(os.path.join(project_root(), 'credentials', 'api_key.txt'))

    request_counter = 0
    start_time = time.time()

    for league_name, league_info in mappings.items():
        league_id = league_info['id']
        season_start = league_info['season_start']
        season_end = league_info['season_end']

        if 'standings' in league_info['data_types']:
            for season in range(season_start, season_end + 1):
                logging.info(f"Processing {league_name} for the {season} season.")
                result, request_counter, start_time, continue_processing = request_data(country,
                                                                                        league_name,
                                                                                        str(league_id),
                                                                                        str(season),
                                                                                        request_counter,
                                                                                        start_time,
                                                                                        api_key)
                if not continue_processing:
                    logging.info(f"Stopping further requests due to error in response for {league_name} {season}.")
                    return


if __name__ == "__main__":
    country = 'Netherlands'  # Example country
    request_raw_data(country)
