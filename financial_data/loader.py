import os
import http.client
import json
import time
import yaml

import pandas as pd

from financial_data.scrape import scrape_league_data


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


def load_csv_data(country, file_name):
    file_path = os.path.join(get_project_root(), 'financial_data', country, file_name)
    return pd.read_csv(file_path)


def request_financial_data(country):
    mappings = load_mappings_from_yaml(os.path.join('settings', f'mapping_{country.lower()}.yaml'))

    data_seasons = []

    for league_name, config in mappings[country].items():
        start = config['season_start']
        end = config['season_end']
        transfermarkt_name = config['transfermarkt_name']

        # Skip if transfermarkt_name is 'none'
        if transfermarkt_name != 'none':
            for season in range(start, end):
                print(f"Processing {league_name} for the {season} season.")
                data_season = scrape_league_data(league_name=league_name, transfermarkt_name=transfermarkt_name, year=season)
                data_seasons.append(data_season)

    financial_data = pd.concat(data_seasons, ignore_index=True)

    return financial_data


if __name__ == "__main__":
    country = 'Germany'

    # financial_data = load_csv_data(country, 'league_financial_data.csv')

    # Example usage
    financial_data = request_financial_data(country)

    # Save financial_data to CSV in the Germany folder under financial_data
    output_path = os.path.join(get_project_root(), 'process_data', country, 'league_financial_data.csv')
    financial_data.to_csv(output_path, index=False)

    print(financial_data.head())
