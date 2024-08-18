import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from econml.grf import CausalForest
from sklearn.model_selection import train_test_split
from utils.load import project_root


def load_processed_data(country, cup):
    file_path = os.path.join(project_root(), 'data/process', country, f'{cup}_processed.csv')
    return pd.read_csv(file_path)


def run_causal_forest_analysis(df):
    # Selecting relevant columns for the analysis
    X = df[['team_home', 'rank_diff', 'next_fixture_days', 'distance',
            'team_size', 'mean_age', 'foreigners', 'total_value']].values
    T = df['team_win'].values  # Treatment: Did the team win the cup match?
    Y = df['next_team_points'].values  # Outcome: Points in the next league match
    round_stages = df['stage'].values  # Round stages for per-round analysis

    # Ensure T and Y are 2D as required by econml
    T = T.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    # Handling missing data: Create a mask for rows without missing values across all relevant columns
    combined_data = pd.DataFrame(np.hstack((X, T, Y)))
    mask = ~combined_data.isnull().any(axis=1)

    # Apply the mask to all arrays
    X, T, Y, round_stages = X[mask], T[mask], Y[mask], round_stages[mask]

    # Split the data into training and testing sets
    X_train, X_test, T_train, T_test, Y_train, Y_test, stage_train, stage_test = train_test_split(
        X, T, Y, round_stages, test_size=0.2, random_state=42)

    # Initialize the Causal Forest model
    model = CausalForest()

    # Fit the model: order of arguments is X, T, Y
    model.fit(X_train, T_train, Y_train)

    # Estimate the CATE using the predict method
    cate = model.predict(X_test)

    # Estimate the ATE by averaging the CATE values
    ate = np.mean(cate)

    # Output the overall ATE
    print(f"Overall ATE: {ate}")

    # Analyze ATE and CATE per round
    for stage in sorted(set(stage_test)):
        mask = stage_test == stage
        stage_cate = model.predict(X_test[mask])

        stage_ate = np.mean(stage_cate)
        print(f"Stage {stage} - ATE: {stage_ate}")

        # Plotting CATE distribution
        plt.figure()
        plt.hist(stage_cate, bins=30, edgecolor='black')
        plt.title(f'CATE Distribution for Stage {stage}')
        plt.xlabel('CATE')
        plt.ylabel('Frequency')
        plt.show()

    return ate, cate


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    # Load the processed DataFrame
    stages_df = load_processed_data(country, cup)

    # Run Causal Forest Analysis
    ate, cate = run_causal_forest_analysis(stages_df)

    # Print overall ATE
    print(f"Overall ATE: {ate}")
