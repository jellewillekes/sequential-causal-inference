import os
import pandas as pd
import numpy as np
from data.preprocess.imputation import impute_data
from raw.loader import project_root

project_root = project_root()


def load_processed_data(country, cup):
    file_path = os.path.join(project_root, 'data/process', country, f'{cup}_processed.csv')
    return pd.read_csv(file_path)


def preprocess_data(df, outcome_var, treatment_var, instrument_var, round_specific_controls, season_specific_controls,
                    num_rounds):
    # Create separate columns for each round for the treatment, instrument, and round-specific control variables
    for round_num in range(1, num_rounds + 1):
        df[f'{treatment_var}_round{round_num}'] = np.where(df['stage'] == round_num, df[treatment_var], np.nan)
        df[f'{instrument_var}_round{round_num}'] = np.where(df['stage'] == round_num, df[instrument_var], np.nan)

        # Create round-specific control variables
        for control in round_specific_controls:
            df[f'{control}_round{round_num}'] = np.where(df['stage'] == round_num, df[control], np.nan)

    # Sort the dataframe to ensure correct order for filling values
    df.sort_values(by=['team_id', 'year', 'stage'], inplace=True)

    # Forward fill treatment and instrument variables to handle noncompliance
    for round_num in range(1, num_rounds + 1):
        # Forward fill the treatment and set to 0 if not present
        df[f'{treatment_var}_round{round_num}'] = df.groupby(['team_id', 'year'])[
            f'{treatment_var}_round{round_num}'].ffill().fillna(0)
        # Forward fill the instrument variable
        df[f'{instrument_var}_round{round_num}'] = df.groupby(['team_id', 'year'])[
            f'{instrument_var}_round{round_num}'].ffill()

    # Set round-specific control variables to zero after team is eliminated
    for round_num in range(2, num_rounds + 1):
        # If team was eliminated in the previous round, set current round's controls to zero
        df.loc[df[f'{treatment_var}_round{round_num - 1}'] == 0, [f'{treatment_var}_round{round_num}',
                                                                  f'{instrument_var}_round{round_num}']] = 0
        for control in round_specific_controls:
            df.loc[df[f'{treatment_var}_round{round_num - 1}'] == 0, f'{control}_round{round_num}'] = 0

    # Pivot the data to have one row per team-season
    df_pivot = df.pivot_table(index=['team_id', 'year'],
                              values=[f'{treatment_var}_round{round_num}' for round_num in range(1, num_rounds + 1)] +
                                     [f'{instrument_var}_round{round_num}' for round_num in range(1, num_rounds + 1)] +
                                     [f'{control}_round{round_num}' for control in round_specific_controls for round_num
                                      in range(1, num_rounds + 1)] +
                                     season_specific_controls + [outcome_var, 'team_name', 'fixture_id'],
                              aggfunc='max').reset_index()

    # Ensure no remaining NaNs for season-specific controls and outcome variable
    df_pivot.dropna(subset=season_specific_controls + [outcome_var], inplace=True)

    return df_pivot


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'
    outcome_var = 'team_rank'
    treatment_var = 'team_win'
    instrument_var = 'team_better'
    round_specific_controls = ['team_home', 'distance']  # Control variables that change each round
    season_specific_controls = ['team_size', 'mean_age', 'foreigners',
                                'total_value']  # Control variables that stay the same for the season
    num_rounds = 6  # Adjust based on the number of rounds in the competition

    # Load the processed DataFrame
    stages_df = load_processed_data(country, cup)

    # Impute missing data if necessary
    stages_df = impute_data(stages_df, method='minmax')

    # Preprocess the data
    preprocessed_df = preprocess_data(stages_df, outcome_var, treatment_var, instrument_var, round_specific_controls,
                                      season_specific_controls, num_rounds)

    nan_rows = preprocessed_df[preprocessed_df.isnull().any(axis=1)]
    print(nan_rows)
    preprocessed_df = preprocessed_df.dropna()

    # Save the preprocessed data to a CSV file
    output_path = os.path.join(project_root, 'causality/factorial_iv', country, f'{cup}_preprocessed.csv')
    preprocessed_df.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to {output_path}")
