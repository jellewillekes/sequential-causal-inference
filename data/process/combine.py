import os
import pandas as pd
from utils.load import project_root, load_mappings_from_yaml, load_processed_data
from data.process.imputation import impute_data

mapping = load_mappings_from_yaml('settings/mapping.yaml')


def generate_country_code_mapping(mapping):
    country_codes = {}
    for idx, country in enumerate(mapping['countries'].keys(), start=1):
        country_codes[country] = idx
    return country_codes


def load_and_process_cup_data():
    all_data = []

    country_codes = generate_country_code_mapping(mapping)

    for country, cup in mapping['countries'].items():
        data = load_processed_data(country, cup)

        data = impute_data(data, method='minmax')

        data['country_name'] = country
        data['country_code'] = country_codes[country]

        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)

    combined_data = combined_data[combined_data['next_fixture_days'] <= 5]

    combined_data = combined_data.dropna(subset='team_size')

    output_path = os.path.join(project_root(), 'data/process/combined', 'combined_cup_processed.csv')
    combined_data.to_csv(output_path, index=False)

    print(f"Combined data saved to {output_path}")


if __name__ == "__main__":
    load_and_process_cup_data()
