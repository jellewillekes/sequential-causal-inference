import os
import pandas as pd
import time
from utils.load import project_root
from data.distance.core import calculate_distance


def calculate_distances(fixtures, country):
    unique_combinations = (fixtures[['team_name', 'opponent_name']]
                           .drop_duplicates()
                           .assign(
        pair=lambda df: df.apply(lambda row: tuple(sorted([row['team_name'], row['opponent_name']])), axis=1))
                           .drop_duplicates(subset=['pair'])
                           .drop(columns=['pair'])
                           .sort_values(by=['team_name', 'opponent_name'])
                           .reset_index(drop=True))

    unique_combinations['distance'] = None

    for i, row in unique_combinations.iterrows():
        print(f'Processing row {i + 1}/{len(unique_combinations)}')
        try:
            distance = calculate_distance(row['team_name'], row['opponent_name'], country)
            unique_combinations.at[i, 'distance'] = distance
        except Exception as e:
            print(f"Error calculating distance for index {i}: {e}")
            save_intermediate_csv(unique_combinations, country, i)
            time.sleep(60)  # Wait a minute before retrying

    return unique_combinations


def save_intermediate_csv(df, country, row_index):
    project_root_path = project_root()
    save_path = os.path.join(project_root_path, 'data', 'process', country, f'{country}_distance_data_incomplete.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved incomplete data at row {row_index} to {save_path}")


def save_to_csv(df, country, cup):
    project_root_path = project_root()
    save_path = os.path.join(project_root_path, 'data', 'process', country, f'{country}_distance_data.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved complete data to {save_path}")


def process_cup_fixtures(country, cup):
    project_root_path = project_root()
    fixtures_path = os.path.join(project_root_path, 'data', 'process', country, f'{cup}_fixtures.csv')

    fixtures_data = pd.read_csv(fixtures_path)
    fixtures_with_distances = calculate_distances(fixtures_data, country)
    save_to_csv(fixtures_with_distances, country, cup)
    return fixtures_with_distances


def request_distance_data(country, cup):
    fixtures_df = process_cup_fixtures(country, cup)
    print(fixtures_df.head())


# Usage example
if __name__ == "__main__":
    country = 'Netherlands'
    cup = 'KNVB_Beker'
    request_distance_data(country, cup)
