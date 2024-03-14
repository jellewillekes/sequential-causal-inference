import os
import json
import pandas as pd
from raw_data.loader import get_project_root, load_league_mappings


class LeagueAnalysis:
    def __init__(self, country):
        self.country = country
        self.project_root = get_project_root()
        self.leagues = load_league_mappings(country)

    def compile_standings(self):
        all_standings = []
        for league, details in self.leagues.items():
            if 'standings' in details['data_types']:
                for season in range(details['season_start'], details['season_end'] + 1):
                    standings_path = os.path.join(self.project_root, 'raw_data', self.country, league, str(season),
                                                  'standings_data.json')
                    if os.path.isfile(standings_path):
                        with open(standings_path, 'r') as file:
                            standings_data = json.load(file)
                            for entry in standings_data['response'][0]['league']['standings'][0]:
                                standings_info = self.process_standings_data(entry, league, details['division'], season)
                                all_standings.append(standings_info)
        df_standings = pd.DataFrame(all_standings)
        df_final = self.calculate_national_rank(df_standings)
        self.save_to_csv(df_final)
        return df_final

    def process_standings_data(self, entry, league, division, season):
        return {
            'League': league,
            'Division': division,
            'Year': season,
            'Position': entry['rank'],
            'Team': entry['team']['name'],
            'Team_id': entry['team']['id'],
            'Points': entry['points'],
            'Form': entry['form'],
            'Played': entry['all']['played'],
            'Win': entry['all']['win'],
            'Draw': entry['all']['draw'],
            'Lose': entry['all']['lose'],
            'GoalsDiff': entry['goalsDiff'],
            'GoalsFor': entry['all']['goals']['for'],
            'GoalsAgainst': entry['all']['goals']['against'],
        }

    def calculate_national_rank(self, df):
        df = df.sort_values(by=['Year', 'Division', 'Position'])
        offsets = self.calculate_offsets(df)
        df['NationalRank'] = df.apply(lambda row: row['Position'] + offsets[(row['Year'], row['Division'])], axis=1)
        return df.sort_values(by=['Year', 'NationalRank']).reset_index(drop=True)

    def calculate_offsets(self, df):
        offsets = {}
        for Year in df['Year'].unique():
            max_position = 0
            for division in sorted(df[df['Year'] == Year]['Division'].unique()):
                offsets[(Year, division)] = max_position
                max_position += df[(df['Year'] == Year) & (df['Division'] == division)]['Position'].max()
        return offsets

    def save_to_csv(self, df):
        save_path = os.path.join(self.project_root, 'process_data', self.country, 'league_standings.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)


def run_analysis(country):
    analysis = LeagueAnalysis(country)
    df_final = analysis.compile_standings()
    print(df_final[['Year', 'Division', 'Position', 'Team', 'NationalRank']].head(5))


if __name__ == "__main__":
    country = 'England'
    run_analysis(country)
