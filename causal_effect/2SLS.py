import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from raw_data.loader import get_project_root

from process_data.imputation import impute_data

project_root = get_project_root()


def load_processed_data(country, cup):
    project_root = get_project_root()
    file_path = os.path.join(project_root, 'process_data', country, f'{cup}_processed.csv')
    return pd.read_csv(file_path)


# Ensure plots directory exists
def ensure_country_plots_dir(country):
    country_plots_dir = os.path.join("plots", country)
    os.makedirs(country_plots_dir, exist_ok=True)
    return country_plots_dir


def perform_2sls_analysis(data, outcome_var, instr_var, treatment_var, control_vars):
    # First stage Regression
    X1 = sm.add_constant(data[[instr_var] + control_vars])
    first_stage_model = sm.OLS(data[treatment_var], X1).fit()
    data['D_hat'] = first_stage_model.predict(X1)

    print('First-stage Regression Summary:')
    print(first_stage_model.summary())

    # Second stage Regression
    X2 = sm.add_constant(data[['D_hat'] + control_vars])
    second_stage_model = sm.OLS(data[outcome_var], X2).fit()

    print('Second-stage Regression Summary:')
    print(second_stage_model.summary())

    return second_stage_model


def analyze_2sls_by_stage(stages_df, outcome_var, instr_var, treatment_var, control_vars):
    unique_stages = stages_df['stage'].unique()
    unique_stages.sort()

    results = []

    for stage in unique_stages:
        print(f'Running Stage {stage}')
        df_stage = stages_df[stages_df['stage'] == stage].copy()

        # Perform 2SLS analysis for each stage
        second_stage_model = perform_2sls_analysis(df_stage, outcome_var, instr_var, treatment_var, control_vars)
        results.append((stage, second_stage_model.params['D_hat'], second_stage_model.bse['D_hat']))

    return results


def plot_causal_effect(results, country_plots_dir):
    results_df = pd.DataFrame(results, columns=['stage', 'Causal_Effect', 'Std_Error'])
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['stage'], results_df['Causal_Effect'], yerr=results_df['Std_Error'], fmt='o', capsize=5)
    plt.title('Causal Effect of Advancing in Each Round on Final League Performance')
    plt.xlabel('stage')
    plt.ylabel('Causal Effect')
    plt.grid(True)
    plt.savefig(os.path.join(country_plots_dir, 'causal_effect_by_stage.png'))
    plt.show()


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    # Load the processed DataFrame
    stages_df = load_processed_data(country, cup)

    stages_df = impute_data(stages_df, method='drop')

    # Ensure country-specific plots directory exists
    country_plots_dir = ensure_country_plots_dir(country)

    # Define variables for the 2SLS analysis
    outcome_var = 'team_rank'
    instr_var = 'opponent_rank_prev'
    treatment_var = 'team_win'
    control_vars = ['team_rank_prev', 'team_size', 'foreigners', 'mean_age', 'mean_value', 'total_value', 'distance']

    # Perform the 2SLS analysis by stage
    results = analyze_2sls_by_stage(stages_df, outcome_var, instr_var, treatment_var, control_vars)

    # Plot the causal effect for all rounds
    plot_causal_effect(results, country_plots_dir)
