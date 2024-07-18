import os
import pandas as pd
from raw_data.loader import get_project_root, load_stages
from distance_data.distance import calculate_distance
import time

class DistanceAnalysis:
    def __init__(self, country, cup):
        self.country = country
        self.cup = cup
        self.project_root = get_project_root()
        self.stages = load_stages(country)

    def process_cup_fixtures(self):
        fixtures_path = os.path.join(self.project_root, 'process_data', self.country,
                                     f'{self.cup}_fixtures.csv')

        fixtures_data = pd.read_csv(fixtures_path)
        fixtures_with_distances = self.calculate_distances(fixtures_data)
        self.save_to_csv(fixtures_with_distances)
        return fixtures_with_distances

    def calculate_distances(self, fixtures):
        unique_combinations = (fixtures[['team_name', 'opponent_name']]
                               .drop_duplicates()
                               .assign(
            Pair=lambda df: df.apply(lambda row: tuple(sorted([row['team_name'], row['opponent_name']])), axis=1))
                               .drop_duplicates(subset=['Pair'])
                               .drop(columns=['Pair'])
                               .sort_values(by=['team_name', 'opponent_name'])
                               .reset_index(drop=True))

        unique_combinations['Distance'] = None

        for i, row in unique_combinations.iterrows():
            print(f'Processing row {i+1}/{len(unique_combinations)}')
            try:
                distance = calculate_distance(row['team_name'], row['opponent_name'], self.country)
                unique_combinations.at[i, 'distance'] = distance
            except Exception as e:
                print(f"Error calculating distance for index {i}: {e}")
                # Save progress to CSV before retrying
                self.save_intermediate_csv(unique_combinations, i)
                time.sleep(60)  # Wait a minute before retrying

        return unique_combinations

    def save_intermediate_csv(self, df, row_index):
        save_path = os.path.join(self.project_root, 'process_data', self.country,
                                 f'{self.cup}_distances_incomplete.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved incomplete data at row {row_index} to {save_path}")

    def save_to_csv(self, df):
        save_path = os.path.join(self.project_root, 'process_data', self.country,
                                 f'{self.cup}_distances.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved complete data to {save_path}")


# Usage example
if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    cup_analysis = DistanceAnalysis(country, cup)
    fixtures_df = cup_analysis.process_cup_fixtures()
    print(fixtures_df.head())
