import os
import pandas as pd
import numpy as np
from raw_data.loader import get_project_root
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# Function for loading data
def load_csv_data(country, file_name):
    project_root = get_project_root()
    file_path = os.path.join(project_root, 'process_data', country, file_name)
    return pd.read_csv(file_path)


# Function for preprocessing data
def preprocess_data(fixtures_df, standings_df):
    fixtures_df = fixtures_df[fixtures_df['Year'] > 2011]
    standings_df = standings_df[standings_df['Year'] > 2011]
    standings_df.loc[:, 'AdjustedYear'] = standings_df['Year'] + 1  # Updated for SettingWithCopyWarning

    # Merging to create stages_df with opponent and team rankings
    stages_df = pd.merge(fixtures_df, standings_df[['AdjustedYear', 'Team_id', 'NationalRank']],
                         left_on=['Year', 'Opponent_id'], right_on=['AdjustedYear', 'Team_id'], how='left',
                         suffixes=('', '_opponent'))
    stages_df.rename(columns={'NationalRank': 'RankOpponent'}, inplace=True)

    stages_df = pd.merge(stages_df, standings_df[['Year', 'Team_id', 'NationalRank']],
                         left_on=['Year', 'Team_id'], right_on=['Year', 'Team_id'], how='left')
    stages_df.rename(columns={'NationalRank': 'Rank'}, inplace=True)

    # Handling missing values
    stages_df['Rank'] = stages_df['Rank'].fillna(75)
    stages_df['Rank_diff'] = (stages_df['Rank'] > stages_df['RankOpponent']).astype(int)
    stages_df['const'] = 1  # Add constant for regression
    stages_df.dropna(subset=['Stage'], inplace=True)
    stages_df['Stage'] = stages_df['Stage'].astype(int)

    # Adding columns for reached round
    for round_number in range(1, max(stages_df['Stage']) + 1):
        stages_df[f'ReachedRound{round_number}'] = stages_df['Stage'].apply(lambda x: 1 if x >= round_number else 0)

    stages_df = stages_df[stages_df["Year"] == 2017]

    # Data cleaning for 'inf' or 'NaN' values before returning the cleaned DataFrame
    features = ['Rank_diff', 'const', 'Rank', 'RankOpponent']  # Adjust this list based on the features you're using
    stages_df.dropna(subset=features, inplace=True)  # Drop rows with NaN in specified features
    stages_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    stages_df.dropna(subset=features, inplace=True)  # Drop rows with NaN in specified features again, if needed

    # Return the cleaned and preprocessed DataFrame
    return stages_df


# Function to check for multicollinearity
def check_multicollinearity(stages_df, features):
    # Prepare the feature matrix X, excluding the 'const' column for correlation and eigenvalue calculations
    X = stages_df[features].dropna(axis=1)
    X_without_const = X.drop(columns=['const'], errors='ignore')  # Drop 'const' if present

    # Calculate VIF
    X_with_const = add_constant(X)  # Add constant here for VIF calculation
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    print("VIF Results:\n", vif_data)

    # VIF Explanation
    print("\nVIF Explained:\nA Variance Inflation Factor (VIF) provides a measure of the increase in variance of a regression coefficient due to multicollinearity. VIF values greater than 10 may indicate problematic multicollinearity.")

    # Calculate condition index
    U, S, V = np.linalg.svd(X_with_const)
    condition_indices = S.max() / S
    condition_data = pd.DataFrame(condition_indices, columns=["Condition Index"])
    print("\nCondition Indices:\n", condition_data)

    # Condition Index Explanation
    print("\nCondition Index Explained:\nCondition Index assesses multicollinearity by examining the singular values from a singular value decomposition of the data matrix. Indices greater than 30 suggest multicollinearity may be inflating the variance of parameter estimates.")

    # Correlation Matrix
    corr_matrix = X_without_const.corr()
    print("\nCorrelation Matrix:\n", corr_matrix)

    # Correlation Matrix Explanation
    print("\nCorrelation Matrix Explained:\nThe correlation matrix shows pairwise correlations between variables. Correlation coefficients close to 1 or -1 indicate a strong linear relationship, suggesting potential multicollinearity issues.")

    # Eigenvalues of the Correlation Matrix
    eigenvalues, _ = np.linalg.eig(corr_matrix.fillna(0))  # Fill NaN with 0 for eigenvalue calculation
    print("\nEigenvalues of Correlation Matrix:\n", eigenvalues)

    # Eigenvalues Explanation
    print("\nEigenvalues Explained:\nEigenvalues of the correlation matrix close to 0 indicate multicollinearity. The presence of small eigenvalues suggests that the data matrix has nearly dependent columns, implying multicollinearity.")



# Function for 2SLS analysis
def perform_2SLS_analysis(stages_df):
    results = {}
    for round_number in range(1, max(stages_df['Stage']) + 1):
        df_round = stages_df[stages_df['Stage'] == round_number].copy()
        if not df_round.empty:
            treatment = f'ReachedRound{round_number}'
            instrument = 'Rank_diff'
            df_round.dropna(subset=['Rank_diff', 'Team_win'], inplace=True)
            first_stage = IV2SLS(df_round[f'ReachedRound{round_number}'], df_round[['const']],
                                 df_round[['Rank_diff']]).fit()
            stages_df[f'{treatment}_pred'] = first_stage.fittedvalues

            df_merged = pd.merge(stages_df, standings_df[['Team_id', 'Year', 'NationalRank']],
                                 left_on=['Team_id', 'AdjustedYear'], right_on=['Team_id', 'Year'], how='left')
            df_merged.dropna(subset=[f'{treatment}_pred', 'Rank'], inplace=True)

            second_stage = OLS(df_merged['Rank'], df_merged[['const', f'{treatment}_pred']]).fit()
            results[f'Stage_{round_number}'] = second_stage.summary()
    return results


# Main execution flow
if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'
    fixtures_df = load_csv_data(country, f'{cup}_fixtures.csv')
    standings_df = load_csv_data(country, 'league_standings.csv')
    stages_df = preprocess_data(fixtures_df, standings_df)
    # Assuming 'Rank_diff' and 'const' are the main features used in your model
    check_multicollinearity(stages_df, ['Rank_diff', 'const', 'Rank', 'RankOpponent'])
    results = perform_2SLS_analysis(stages_df)
    for key, value in results.items():
        print(f"Results for {key}:")
        print(value)
        print("\n---\n")
