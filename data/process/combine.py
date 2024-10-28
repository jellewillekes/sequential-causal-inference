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


def generate_summary_statistics(combined_data):
    # Calculate the latest year in the dataset
    latest_year = combined_data['year'].max()

    # Initialize an empty list to store the summary rows
    summary_rows = []

    # List of unique countries/cups
    countries_cups = combined_data['country_name'].unique()

    for country in countries_cups:
        country_data = combined_data[combined_data['country_name'] == country]

        # Get unique rounds
        stages = country_data['stage'].unique()

        for round_name in stages:
            round_data = country_data[country_data['stage'] == round_name]

            # Get the latest year data
            latest_year_data = round_data[round_data['year'] == latest_year]

            # Number of matches (fixture_id) in the latest year
            matches_latest_year = latest_year_data['fixture_id'].nunique()

            # Average number of matches (fixture_id) over all years
            avg_matches_all_years = round_data.groupby('year')['fixture_id'].nunique().mean()

            # Number of teams in the latest year
            teams_latest_year = latest_year_data['team_id'].nunique()

            # Average number of teams per round over all years
            avg_teams_all_years = round_data.groupby('year')['team_id'].nunique().mean()

            # Total matches (all years)
            total_matches_all_years = round_data['fixture_id'].nunique()

            # Year span
            year_span = f"{round_data['year'].min()}-{round_data['year'].max()}"

            # Add the row to the summary list
            summary_rows.append({
                'Country': country,
                'Round': round_name,
                'Stage': round_data['stage'].iloc[0],  # assuming stage is consistent within a round
                'Latest Year': latest_year,
                f'Matches ({latest_year})': matches_latest_year,
                'Avg. Matches (All Years)': avg_matches_all_years,
                f'Teams ({latest_year})': teams_latest_year,
                'Avg. Teams (All Years)': avg_teams_all_years,
                'Year Span': year_span,
                'Total Matches (All Years)': total_matches_all_years
            })

    # Convert the summary list to a DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Calculate the grand totals for the entire dataset
    grand_totals = pd.DataFrame([{
        'Country': 'Grand Total',
        'Round': '',
        'Stage': '',
        'Latest Year': '',
        f'Matches ({latest_year})': summary_df[f'Matches ({latest_year})'].sum(),
        'Avg. Matches (All Years)': summary_df['Avg. Matches (All Years)'].mean(),
        f'Teams ({latest_year})': summary_df[f'Teams ({latest_year})'].sum(),
        'Avg. Teams (All Years)': summary_df['Avg. Teams (All Years)'].mean(),
        'Year Span': '',
        'Total Matches (All Years)': summary_df['Total Matches (All Years)'].sum()
    }])

    # Concatenate the summary_df and grand_totals
    summary_df = pd.concat([summary_df, grand_totals], ignore_index=True)
    summary_df = summary_df.sort_values(by=['Country', 'Round'], ascending=[True, True])

    return summary_df


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

    combined_data = combined_data.dropna(subset=['team_size', 'distance'])

    output_path = os.path.join(project_root(), 'data/process/combined', 'combined_cup_processed_win.csv')
    combined_data.to_csv(output_path, index=False)

    print(f"Combined data saved to {output_path}")

    summary_statistics = generate_summary_statistics(combined_data)
    output_path = os.path.join(project_root(), 'data/process/combined', 'cup_summary_statistics.csv')
    summary_statistics.to_csv(output_path, index=False)
    print(f"Summary statistics saved to {output_path}")


if __name__ == "__main__":
    load_and_process_cup_data()
