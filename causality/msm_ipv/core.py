import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.load import project_root
from data.process.imputation import impute_data

from pygam import LogisticGAM, s, f


def load_processed_data(country, cup):
    file_path = os.path.join(project_root(), 'data/process', country, f'{cup}_processed.csv')
    return pd.read_csv(file_path)


def ensure_country_plots_dir(country):
    country_plots_dir = os.path.join("plots", country, "Causal_Effect")
    os.makedirs(country_plots_dir, exist_ok=True)
    return country_plots_dir


def create_lagged_variables(data, group_vars, treatment_var):
    data = data.sort_values(by=group_vars + ['stage'])
    data[f'{treatment_var}_lagged'] = data.groupby(group_vars)[treatment_var].shift(1)
    data = data.dropna(subset=[f'{treatment_var}_lagged'])
    return data


def estimate_propensity_scores(data, treatment_var, control_vars):
    X = data[control_vars]
    y = data[treatment_var]

    categorical_vars = ['team_win_lagged', 'division', 'team_home', 'extra_time']
    continuous_vars = ['team_size', 'foreigners', 'mean_age', 'total_value', 'distance', 'next_fixture_days']

    # Scale the continuous variables
    scaler = StandardScaler()
    X_continuous = scaler.fit_transform(X[continuous_vars])

    # Reconstruct the DataFrame with scaled continuous variables and original categorical variables
    X_scaled = pd.DataFrame(X_continuous, columns=continuous_vars)
    X_scaled[categorical_vars] = X[categorical_vars].reset_index(drop=True)

    # Define the GAM model with regularization (increase 'lam' to add more regularization)
    gam = LogisticGAM(
        f(0) + f(1) + f(2) + f(3) +  # Categorical variables
        s(4) + s(5) + s(6) + s(7) + s(8) + s(9)  # Continuous variables
    ).fit(X_scaled, y)

    # You can adjust 'lam' parameter to regularize the spline terms
    gam = LogisticGAM(
        f(0) + f(1) + f(2) + f(3) +  # Categorical variables
        s(4, lam=1.0) + s(5, lam=1.0) + s(6, lam=1.0) + s(7, lam=1.0) + s(8, lam=1.0) + s(9, lam=1.0)  # Regularized continuous variables
    )

    try:
        gam.fit(X_scaled, y)
        # Predict propensity scores
        propensity_scores = gam.predict_proba(X_scaled)
        plot_gam_terms(gam, X_scaled)
    except Exception as e:
        print(f"GAM did not converge: {e}")
        propensity_scores = np.full_like(y, 0.5, dtype=float)

    return propensity_scores


def plot_gam_terms(gam, X_scaled):
    """Plot the smooth terms from the fitted GAM model."""
    for i, term in enumerate(gam.terms):
        from pygam.terms import SplineTerm
        if isinstance(term, SplineTerm):  # Check if the term is a SplineTerm (smooth term)
            plt.figure()
            XX = gam.generate_X_grid(term=i)
            plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
            plt.title(f'Partial Dependence of {X_scaled.columns[i]}')
            plt.xlabel(X_scaled.columns[i])
            plt.ylabel('Partial Dependence')
            plt.show()


def compute_iptw_weights(data, treatment_var, propensity_scores):
    treatment = data[treatment_var]
    weights = treatment / propensity_scores + (1 - treatment) / (1 - propensity_scores)
    return weights


def perform_msm_analysis(data, outcome_var, treatment_var, control_vars, stage):
    propensity_scores = estimate_propensity_scores(data, treatment_var, control_vars)
    weights = compute_iptw_weights(data, treatment_var, propensity_scores)

    plot_propensity_scores(data, propensity_scores, treatment_var, stage)

    X = sm.add_constant(data[[treatment_var] + control_vars])

    # Check for multicollinearity and remove highly collinear variables
    if X.shape[1] > 1:
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        print(vif)

    try:
        model = sm.WLS(data[outcome_var], X, weights=weights).fit()
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e}")
        model = None

    # Residual analysis
    if model:
        plot_residuals(model, stage)

    return model


def plot_propensity_scores(data, propensity_scores, treatment_var, stage):
    """Plot the distribution of propensity scores for treated and control groups for a given stage."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(propensity_scores[data[treatment_var] == 1], fill=True, label='Treated', color='blue')
    sns.kdeplot(propensity_scores[data[treatment_var] == 0], fill=True, label='Control', color='red')
    plt.title(f'Propensity Score Distribution for Round {stage}')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def plot_residuals(model, stage):
    """Plot residuals to check for normality and homoscedasticity."""
    residuals = model.resid
    fitted = model.fittedvalues

    plt.figure(figsize=(10, 6))

    # Residuals vs Fitted
    plt.subplot(1, 2, 1)
    plt.scatter(fitted, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs Fitted (Round {stage})')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    sm.qqplot(residuals, line='45', fit=True)
    plt.title(f'Q-Q Plot of Residuals (Round {stage})')

    plt.tight_layout()
    plt.show()


def analyze_msm_by_stage(stages_df, outcome_var, treatment_var, control_vars, display="none"):
    unique_stages = stages_df['stage'].unique()
    unique_stages.sort()
    results = []
    summaries = {}

    for stage in unique_stages:
        if "summary" in display:
            print(f'Running Round {stage}')
        df_stage = stages_df[stages_df['stage'] == stage].copy()
        msm_model = perform_msm_analysis(df_stage, outcome_var, treatment_var, control_vars, stage)

        if msm_model:
            if "summary" in display:
                print(msm_model.summary())
            results.append({
                'stage': stage,
                'msm_effect': msm_model.params[treatment_var],
                'std_error': msm_model.bse[treatment_var],
                'p_value': msm_model.pvalues[treatment_var],
                'r_squared': msm_model.rsquared
            })
            summaries[stage] = msm_model.summary().as_text()
        else:
            results.append({
                'stage': stage,
                'msm_effect': None,
                'std_error': None,
                'p_value': None,
                'r_squared': None
            })

    return results, summaries


def plot_causal_effect(results, country_plots_dir, display="none", filename="causal_effect_by_stage.png"):
    results_df = pd.DataFrame(results)
    if "plot" in display:
        plt.figure(figsize=(10, 6))
        plt.errorbar(results_df['stage'], results_df['msm_effect'], yerr=results_df['std_error'], fmt='o', capsize=5)
        plt.title('Causal Effect of Advancing in Each Round on Final League Performance')
        plt.xlabel('Round')
        plt.ylabel('Causal Effect (MSM Estimate)')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'
    display = "summary plot"

    stages_df = load_processed_data(country, cup)
    stages_df = stages_df[stages_df['next_fixture_days'] < 6]

    stages_df = impute_data(stages_df, method='minmax')
    country_plots_dir = ensure_country_plots_dir(country)
    stages_df = create_lagged_variables(stages_df, group_vars=['team_id', 'year'], treatment_var='team_win')
    outcome_var = 'next_team_points'
    treatment_var = 'team_win'
    control_vars = ['team_win_lagged', 'division', 'team_home', 'extra_time', 'team_size', 'foreigners', 'mean_age',
                    'total_value', 'distance', 'next_fixture_days']

    results, summaries = analyze_msm_by_stage(stages_df, outcome_var, treatment_var, control_vars, display)

    results_df = pd.DataFrame(results)
    print(results_df)

    plot_causal_effect(results, country_plots_dir, display)
