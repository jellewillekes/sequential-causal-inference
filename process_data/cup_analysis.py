from raw_data.loader import get_project_root, load_stages
import os
import json
import pandas as pd


class CupAnalysis:
    def __init__(self, country, cup):
        self.country = country
        self.cup = cup
        self.project_root = get_project_root()
        self.stages = load_stages(country)

    def process_cup_fixtures(self, season_start, season_end):
        all_matches = []

        for season in range(season_start, season_end + 1):
            season_path = os.path.join(self.project_root, 'raw_data', self.country, self.cup, str(season),
                                       'fixtures_data.json')

            if os.path.isfile(season_path):
                with open(season_path, 'r') as file:
                    fixtures_data = json.load(file)
                    all_matches.extend(self.process_season_fixtures(fixtures_data, season))

        matches_df = pd.DataFrame(all_matches)
        self.save_to_csv(matches_df)
        return matches_df

    def process_season_fixtures(self, fixtures_data, season):
        season_matches = []
        for fixture in fixtures_data['response']:
            round_name = fixture['league']['round']
            stage_number = self.stages.get(round_name)

            home_winner = fixture['teams']['home'].get('winner')
            away_winner = fixture['teams']['away'].get('winner')

            home_team_win = int(home_winner) if home_winner is not None else None
            away_team_win = int(away_winner) if away_winner is not None else None

            match_data_home = self.construct_match_data(season, round_name, stage_number, fixture, 'home',
                                                        home_team_win)
            match_data_away = self.construct_match_data(season, round_name, stage_number, fixture, 'away',
                                                        away_team_win)

            season_matches.append(match_data_home)
            season_matches.append(match_data_away)

        return season_matches

    def construct_match_data(self, season, round_name, stage_number, fixture, team_type, team_win):
        team_key = 'home' if team_type == 'home' else 'away'
        opponent_key = 'away' if team_type == 'home' else 'home'

        return {
            'Year': season,
            'Round': round_name,
            'Stage': stage_number,
            'Team_name': fixture['teams'][team_key]['name'],
            'Team_id': fixture['teams'][team_key]['id'],
            'Opponent_name': fixture['teams'][opponent_key]['name'],
            'Opponent_id': fixture['teams'][opponent_key]['id'],
            'Team_win': team_win
        }

    def save_to_csv(self, df):
        save_path = os.path.join(self.project_root, 'process_data', self.country,
                                 f'{self.cup}_fixtures.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)


# Usage example
if __name__ == "__main__":
    country = 'Netherlands'
    cup = 'KNVB_Beker'
    season_start = 2011
    season_end = 2023

    cup_analysis = CupAnalysis(country, cup)
    fixtures_df = cup_analysis.process_cup_fixtures(season_start, season_end)
    print(fixtures_df.head())
