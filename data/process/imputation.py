import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def minmax_impute(match_df):
    non_league_division = match_df['team_division'].max()
    max_division = non_league_division - 1
    lowest_known_division = match_df['team_division'].min()

    # Define columns to impute
    min_value_columns = ['team_size', 'foreigners', 'mean_value', 'total_value']
    max_value_columns = ['mean_age']

    # Function to adjust lower mean with a Z-score adjustment
    def adjusted_lower_mean_zscore(df, col, percentile=0.05, z_score=-1):
        if df[col].notna().sum() == 0:  # Handle cases where no data is available
            return np.nan
        lower_bound = df[col].quantile(percentile)
        subset = df[df[col] <= lower_bound][col]
        lower_mean = subset.mean()
        lower_std = subset.std()
        adjusted_mean = lower_mean + z_score * lower_std
        return max(adjusted_mean, 0)

    for col in min_value_columns:
        # Impute missing values with the mean for league teams
        means = match_df.groupby(['year', 'team_division'])[col].transform('mean')
        match_df[col] = match_df[col].fillna(means)

        # Debugging: Print out missing values before non-league imputation
        print(f"Missing {col} before non-league imputation:", match_df[col].isna().sum())

        # Adjust non-league teams and NaN division values
        non_league_or_nan_teams = (match_df['team_division'] == non_league_division) | (
            match_df['team_division'].isna())
        match_df.loc[non_league_or_nan_teams, col] = match_df[non_league_or_nan_teams].apply(
            lambda row:
            adjusted_lower_mean_zscore(
                match_df[(match_df['year'] == row['year']) & (match_df['team_division'] == max_division)], col
            )
            if row['year'] in match_df['year'].unique() else np.nan,
            axis=1
        )

        # Fallback: Take the mean of the lowest known division for any remaining NaNs
        lowest_division_mean = match_df[match_df['team_division'] == lowest_known_division][col].mean()
        match_df[col] = match_df[col].fillna(lowest_division_mean)

        # Debugging: Print out missing values after fallback imputation
        print(f"Missing {col} after using lowest division mean:", match_df[col].isna().sum())

    # Handle max_value_columns, specifically 'mean_age'
    for col in max_value_columns:
        # Impute missing values with the mean for league teams
        means = match_df.groupby(['year', 'team_division'])[col].transform('mean')
        match_df[col] = match_df[col].fillna(means)

        # Fallback: Use the mean of the lowest known division for any remaining NaNs
        lowest_division_mean = match_df[match_df['team_division'] == lowest_known_division][col].mean()
        match_df[col] = match_df[col].fillna(lowest_division_mean)

        # Debugging: Print out missing values after fallback imputation
        print(f"Missing {col} after using lowest division mean:", match_df[col].isna().sum())

    return match_df


def drop_nan_impute(match_df):
    print(match_df)

    return match_df


def impute_data(match_df, method='minmax'):
    if method == 'minmax':
        match_df = minmax_impute(match_df)

    elif method == 'drop':
        match_df = match_df.dropna(subset=['team_division', 'league'])
    else:
        match_df = match_df

    return match_df


def regression_impute(df, target_col, impute_col, fill_value, non_league_value, log_transform=False):
    """
    Perform regression imputation by extrapolation for a target column based on an impute column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    target_col (str): The name of the column to be imputed.
    impute_col (str): The name of the column to use for the imputation.
    fill_value: The value to replace NaNs in the impute column.
    non_league_value: The value that indicates non-league teams.
    log_transform: Apply log transformation if True.

    Returns:
    pd.DataFrame: DataFrame with the imputed values.
    """
    # Avoid SettingWithCopyWarning
    df = df.copy()

    # Prepare data by replacing NaN in impute_col with a fill value
    df[impute_col] = df[impute_col].fillna(fill_value)

    # Separate league and non-league teams
    league_teams = df[(df[impute_col] != non_league_value) & df[target_col].notna()]
    non_league_teams = df[df[impute_col] == non_league_value]

    # Prepare the regression model
    X = league_teams[[impute_col]]
    y = league_teams[target_col]

    # Apply log transformation if specified
    if log_transform:
        y = np.log1p(y)  # log1p is used to handle zero values

    # Train the regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict and impute the missing values for non-league teams (division 4)
    predicted_values = model.predict([[non_league_value]])

    # Plot division vs. target_col with regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='League Teams')
    plt.plot([1, 2, 3, non_league_value], model.predict([[1], [2], [3], [non_league_value]]), color='red',
             label='Regression Line')
    plt.scatter([non_league_value], predicted_values, color='green', label='Predicted Value for Division 4', zorder=5)
    plt.xlabel(impute_col)
    plt.ylabel(target_col)
    plt.title(f'Division vs. {target_col} with Regression Line and Predicted Value')
    plt.legend()
    plt.show()

    # Revert log transformation if applied
    if log_transform:
        predicted_values = np.expm1(predicted_values)  # expm1 to revert log1p transformation

    df.loc[non_league_teams.index, target_col] = predicted_values[0]

    return df


def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def exponential_decay_impute(df, target_col, impute_col, fill_value, non_league_value, log_transform=False):
    """
    Perform imputation using an exponential decay model for a target column based on an impute column.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    target_col (str): The name of the column to be imputed.
    impute_col (str): The name of the column to use for the imputation.
    fill_value: The value to replace NaNs in the impute column.
    non_league_value: The value that indicates non-league teams.
    log_transform: Apply log transformation if True.

    Returns:
    pd.DataFrame: DataFrame with the imputed values.
    """
    # Avoid SettingWithCopyWarning
    df = df.copy()

    # Prepare data by replacing NaN in impute_col with a fill value
    df[impute_col] = df[impute_col].fillna(fill_value)

    # Separate league and non-league teams
    league_teams = df[(df[impute_col] != non_league_value) & df[target_col].notna()]
    non_league_teams = df[df[impute_col] == non_league_value]

    # Prepare the data for curve fitting
    X = league_teams[impute_col].values
    y = league_teams[target_col].values

    # Apply log transformation if specified
    if log_transform:
        y = np.log1p(y)  # log1p is used to handle zero values

    # Fit the exponential decay model
    params, _ = curve_fit(exponential_decay, X, y, maxfev=10000)

    # Predict and impute the missing values for non-league teams (division 4)
    predicted_value = exponential_decay(non_league_value, *params)

    # Revert log transformation if applied
    if log_transform:
        predicted_value = np.expm1(predicted_value)  # expm1 to revert log1p transformation
        y = np.expm1(y)  # Revert y values for correct plotting

    # Plot division vs. target_col with exponential decay curve
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='League Teams')
    X_plot = np.linspace(1, non_league_value, 100)
    plt.plot(X_plot, exponential_decay(X_plot, *params), color='red', label='Exponential Decay Model')
    plt.scatter([non_league_value], [predicted_value], color='green', label='Predicted Value for Division 4', zorder=5)
    plt.xlabel(impute_col)
    plt.ylabel(target_col)
    plt.title(f'Division vs. {target_col} with Exponential Decay Model and Predicted Value')
    plt.legend()
    plt.show()

    df.loc[non_league_teams.index, target_col] = predicted_value

    return df
