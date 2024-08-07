import os
import pandas as pd

from data.financial.scrape import scrape_league_data
from utils.load import project_root, load_mappings_from_yaml


def request_financial_data(country):
    mappings = load_mappings_from_yaml(os.path.join(project_root(),
                                                    'settings',
                                                    f'mapping_{country.lower()}.yaml'))

    data_seasons = []

    for league_name, config in mappings.items():
        print(f'Requesting financial data for {league_name}')
        start = config['season_start']
        end = config['season_end']
        transfermarkt_name = config['transfermarkt_name']

        if transfermarkt_name != 'none':
            for season in range(start, end + 1):
                print(f"Processing {league_name} for the {season} season.")
                data_season = scrape_league_data(league_name=league_name,
                                                 transfermarkt_name=transfermarkt_name,
                                                 year=season)
                data_seasons.append(data_season)

    financial_data = pd.concat(data_seasons, ignore_index=True)

    # Save financial data to CSV
    output_path = os.path.join(project_root(),
                               'data',
                               'process',
                               country,
                               f'{country}_financial_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    financial_data.to_csv(output_path, index=False)
    print(f"Saved financial data to {output_path}")

    return financial_data


if __name__ == "__main__":
    country = 'Netherlands'
    financial_data = request_financial_data(country)
    print(financial_data.head())
