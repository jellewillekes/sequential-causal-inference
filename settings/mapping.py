import os
import pandas as pd
import Levenshtein
from utils.load import project_root, load_csv


def extract_unique_team_names(country: str, cup: str):
    # Load cup fixtures and financial data
    cup_fixtures = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_fixtures.csv'))
    financial_data = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_financial_data.csv'))

    # Extract unique team names from both datasets
    unique_cup_teams = cup_fixtures['team_name'].unique()
    unique_financial_teams = financial_data['team_name'].unique()

    return unique_cup_teams, unique_financial_teams


def generate_team_mapping(unique_cup_teams, unique_financial_teams):
    """
    Generate a mapping between cup team names and financial team names based on Levenshtein distance.
    Returns a DataFrame with columns: ['cup_team_name', 'financial_team_name', 'match_ratio'].
    """
    mapping_results = []

    for cup_team in unique_cup_teams:
        best_match = None
        best_score = 0

        for fin_team in unique_financial_teams:
            score = Levenshtein.ratio(cup_team, fin_team)
            if score > best_score:
                best_match = fin_team
                best_score = score

        mapping_results.append({
            'cup_team_name': cup_team,
            'financial_team_name': best_match,
            'match_ratio': best_score
        })

    mapping_df = pd.DataFrame(mapping_results)
    return mapping_df


if __name__ == "__main__":
    country = 'Netherlands'
    cup = 'KNVB_Beker'

    unique_cup_teams, unique_financial_teams = extract_unique_team_names(country, cup)

    print('Cup')
    print(unique_cup_teams)

    print('Financial')
    print(unique_financial_teams)

    # Generate the mapping between cup and financial teams
    team_mapping_df = generate_team_mapping(unique_cup_teams, unique_financial_teams)

    print("\nGenerated Team Mapping:")
    print(team_mapping_df)

    # Save the mapping to a CSV file if needed
    output_path = os.path.join(project_root(), 'settings', country, f'{cup}_team_mapping.csv')
    team_mapping_df.to_csv(output_path, index=False)
    print(f"Mapping saved to {output_path}")
