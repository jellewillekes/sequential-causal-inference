import pandas as pd
import os
from raw_data.loader import get_project_root


def load_csv_data(country, file_name):
    project_root = get_project_root()
    file_path = os.path.join(project_root, 'process_data', country, file_name)
    return pd.read_csv(file_path)


# Usage
country = 'Netherlands'
fixtures_df = load_csv_data(country, 'KNVB_Beker_fixtures.csv')
standings_df = load_csv_data(country, 'league_standings.csv')

# Increase 'Season' in standings_df by 1 to represent it as the previous year's standing
standings_df['AdjustedYear'] = standings_df['Year'] + 1

# Now, merge using 'Year' from fixtures_df with the adjusted 'AdjustedSeason' in standings_df
merged_df = pd.merge(fixtures_df,
                     standings_df[['AdjustedYear', 'Team_id', 'NationalRank']],
                     left_on=['Year', 'Opponent_id'],
                     right_on=['AdjustedYear', 'Team_id'],
                     how='left',
                     suffixes=('', '_opponent'))

filtered_df = merged_df[(merged_df['NationalRank_opponent'].isna()) & (merged_df['Stage'] > 2)]


merged_df.dropna(subset=['NationalRank_opponent'], inplace=True)

print('test')
