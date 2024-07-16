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
                            # Check if 'response' and 'league']['standings' lists are not empty
                            if standings_data['response'] and standings_data['response'][0]['league']['standings']:
                                for entry in standings_data['response'][0]['league']['standings'][0]:
                                    standings_info = self.process_standings_data(entry, league, details['division'],
                                                                                 season)
                                    all_standings.append(standings_info)
                            else:
                                print(f"No standings data available for {league} in season {season}")

        df_standings = pd.DataFrame(all_standings)
        df_final = self.calculate_national_rank(df_standings)
        self.save_to_csv(df_final, 'league_standings.csv')
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

    def save_to_csv(self, df, filename):
        save_path = os.path.join(self.project_root, 'process_data', self.country, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    def compile_fixtures(self):
        all_fixtures = []
        for league, details in self.leagues.items():
            if 'fixtures' in details['data_types']:
                for season in range(details['season_start'], details['season_end'] + 1):
                    standings_path = os.path.join(self.project_root, 'raw_data', self.country, league, str(season),
                                                  'fixtures_data.json')
                    if os.path.isfile(standings_path):
                        with open(standings_path, 'r') as file:
                            fixtures_data = json.load(file)
                            all_fixtures.extend(self.process_season_fixtures(fixtures_data, season))

        df_fixtures = pd.DataFrame(all_fixtures)

        def add_team_points(df):
            df['Team_points'] = df['Team_win'].apply(lambda x: 3 if x == 1 else (1 if pd.isna(x) else 0))
            return df

        df_fixtures = add_team_points(df_fixtures)

        self.save_to_csv(df_fixtures, 'league_fixtures.csv')

        return df_fixtures

    def process_season_fixtures(self, fixtures_data, season):
        season_matches = []
        for fixture in fixtures_data['response']:
            round_name = fixture['league']['round']

            home_winner = fixture['teams']['home'].get('winner')
            away_winner = fixture['teams']['away'].get('winner')

            home_team_win = int(home_winner) if home_winner is not None else None
            away_team_win = int(away_winner) if away_winner is not None else None

            match_data_home = self.construct_match_data(season, round_name, fixture, 'home',
                                                        home_team_win)
            match_data_away = self.construct_match_data(season, round_name, fixture, 'away',
                                                        away_team_win)

            season_matches.append(match_data_home)
            season_matches.append(match_data_away)

        return season_matches

    def construct_match_data(self, season, round_name, fixture, team_type, team_win):
        team_key = 'home' if team_type == 'home' else 'away'
        opponent_key = 'away' if team_type == 'home' else 'home'

        return {
            'Year': season,
            'Date': fixture['fixture']['date'],
            'Round': round_name,
            'City': fixture['fixture']['venue']['city'],
            'Team_name': fixture['teams'][team_key]['name'],
            'Team_id': fixture['teams'][team_key]['id'],
            'Opponent_name': fixture['teams'][opponent_key]['name'],
            'Opponent_id': fixture['teams'][opponent_key]['id'],
            'Team_win': team_win,
            'Team_goals': fixture['goals'][team_key],
            'Opponent_goals': fixture['goals'][opponent_key],
        }


def run_analysis(country):
    analysis = LeagueAnalysis(country)
    fixtures_final = analysis.compile_fixtures()
    # print('test')
    # standings_final = analysis.compile_standings()
    # print(standings_final[['Year', 'Division', 'Position', 'Team', 'NationalRank']].head(5))


if __name__ == "__main__":
    country = 'Netherlands'
    run_analysis(country)
