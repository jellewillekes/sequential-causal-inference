import os
import pandas as pd

from utils.load import project_root, load_mappings_from_yaml, load_processed_data
from data.process.imputation import impute_data


mapping = load_mappings_from_yaml('settings/mapping.yaml')


def load_and_process_cup_data():
    all_data = []

    for country, cup in mapping['countries'].items():
        data = load_processed_data(country, cup)

        data = impute_data(data, method='minmax')
        data['country'] = country

        # Append the processed data to the list
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)

    output_path = os.path.join(project_root(), 'data/process/combined', 'combined_cup_data.csv')
    combined_data.to_csv(output_path, index=False)

    print(f"Combined data saved to {output_path}")


if __name__ == "__main__":
    load_and_process_cup_data()
