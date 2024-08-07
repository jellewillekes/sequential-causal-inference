import os
import pandas as pd
import numpy as np
from raw_data.loader import project_root
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# Function for loading data
def load_csv_data(country, file_name):
    project_root = project_root()
    file_path = os.path.join(project_root, 'process', country, file_name)
    return pd.read_csv(file_path)

def preprocess_data(fixtures_df, standings_df):
    # Adjust Year in standings_df to refer to the previous year
    standings_df['PreviousYear'] = standings_df['Year'] - 1

    # Merge fixtures with standings to get NationalRank for Team_id for the previous year
    fixtures_df = pd.merge(fixtures_df, standings_df[['Team_id', 'NationalRank', 'PreviousYear']],
                           left_on=['Team_id', 'Year'], right_on=['Team_id', 'PreviousYear'],
                           suffixes=('', '_Team'))

    # Merge fixtures with standings to get NationalRank for Opponent_id for the previous year
    fixtures_df = pd.merge(fixtures_df, standings_df[['Team_id', 'NationalRank', 'PreviousYear']],
                           left_on=['Opponent_id', 'Year'], right_on=['Team_id', 'PreviousYear'],
                           suffixes=('', '_Opponent'))

    # Calculate the difference in NationalRank
    fixtures_df['RankDiff'] = fixtures_df['NationalRank_Opponent'] - fixtures_df['NationalRank_Team']

    # Create a new DataFrame with only the necessary columns
    regression_df = fixtures_df[['Year', 'Team_id', 'Opponent_id', 'Team_win', 'RankDiff']].dropna()

    return regression_df


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
        df_round = df_round[df_round['Year'] == 2018].copy()
        if not df_round.empty:
            treatment = f'Team_win'
            instrument = 'RankDiff'
            outcome = 'NationalRank'
            df_round.dropna(subset=[instrument, treatment], inplace=True)
            first_stage = IV2SLS(df_round[treatment], df_round[['const']],
                                 df_round[[instrument]]).fit()
            df_round[f'{treatment}_pred'] = first_stage.fittedvalues

            second_stage = OLS(df_round[outcome], df_round[['const', f'{treatment}_pred']]).fit()
            results[f'Stage_{round_number}'] = second_stage.summary()
    return results


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'
    fixtures_df = load_csv_data(country, f'{cup}_fixtures.csv')
    standings_df = load_csv_data(country, 'league_standings.csv')

    regression_df = preprocess_data(fixtures_df, standings_df)
    print(regression_df.head())
