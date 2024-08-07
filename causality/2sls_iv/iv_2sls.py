import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from raw.loader import project_root
from data.preprocess.imputation import impute_data

project_root = project_root()


def load_processed_data(country, cup):
    file_path = os.path.join(project_root(), 'data/process', country, f'{cup}_processed.csv')
    return pd.read_csv(file_path)


def ensure_country_plots_dir(country):
    country_plots_dir = os.path.join("plots", country, "Causal_Effect")  # Modify this line
    os.makedirs(country_plots_dir, exist_ok=True)
    return country_plots_dir


def perform_2sls_analysis(data, outcome_var, instr_var, treatment_var, control_vars, display="none"):
    # First stage Regression
    X1 = sm.add_constant(data[[instr_var] + control_vars])
    first_stage_model = sm.OLS(data[treatment_var], X1).fit()
    data['D_hat'] = first_stage_model.predict(X1)

    if "summary" in display:
        print('First-stage Regression Summary:')
        print(first_stage_model.summary())

    # Second stage Regression
    X2 = sm.add_constant(data[['D_hat'] + control_vars])
    second_stage_model = sm.OLS(data[outcome_var], X2).fit()

    if "summary" in display:
        print('Second-stage Regression Summary:')
        print(second_stage_model.summary())

    return {
        'second_stage_model': second_stage_model,
        'first_stage_model': first_stage_model,
        'second_stage_params': second_stage_model.params,
        'second_stage_bse': second_stage_model.bse,
        'second_stage_pvalues': second_stage_model.pvalues,
        'second_stage_rsquared': second_stage_model.rsquared,
        'first_stage_fvalue': first_stage_model.fvalue,
        'first_stage_pvalue': first_stage_model.f_pvalue,
    }


def analyze_2sls_by_stage(stages_df, outcome_var, instr_var, treatment_var, control_vars, display="none"):
    unique_stages = stages_df['stage'].unique()
    unique_stages.sort()

    results = []
    summaries = {}

    for stage in unique_stages:
        if "summary" in display:
            print(f'Running Stage {stage}')
        df_stage = stages_df[stages_df['stage'] == stage].copy()

        # Perform 2SLS analysis for each stage
        analysis_result = perform_2sls_analysis(df_stage, outcome_var, instr_var, treatment_var, control_vars, display)
        results.append({
            'stage': stage,
            '2sls_iv': analysis_result['second_stage_params']['D_hat'],
            'std_error': analysis_result['second_stage_bse']['D_hat'],
            'p_value': analysis_result['second_stage_pvalues']['D_hat'],
            'r_squared': analysis_result['second_stage_rsquared'],
            'f_stat': analysis_result['first_stage_fvalue'],
            'f_p_value': analysis_result['first_stage_pvalue'],
        })
        summaries[stage] = {
            'first_stage_summary': analysis_result['first_stage_model'].summary().as_text(),
            'second_stage_summary': analysis_result['second_stage_model'].summary().as_text()
        }

    return results, summaries


def plot_causal_effect(results, country_plots_dir, display="none", filename="causal_effect_by_stage.png"):
    results_df = pd.DataFrame(results)
    if "plot" in display:
        plt.figure(figsize=(10, 6))
        plt.errorbar(results_df['stage'], results_df['2sls_iv'], yerr=results_df['std_error'], fmt='o', capsize=5)
        plt.title('Causal Effect of Advancing in Each Round on Final League Performance')
        plt.xlabel('stage')
        plt.ylabel('Causal Effect')
        plt.grid(True)
        plt.savefig(os.path.join(country_plots_dir, filename))
        plt.show()


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'
    display = " "  # Set to "summary plot" to display both summaries and plots

    # Load the processed DataFrame
    stages_df = load_processed_data(country, cup)

    stages_df = impute_data(stages_df, method='minmax')

    # Ensure country-specific plots directory exists
    country_plots_dir = ensure_country_plots_dir(country)

    # Define variables for the 2SLS analysis
    outcome_var = 'team_rank'
    instr_var = 'team_better'
    treatment_var = 'team_win'
    control_vars = ['team_rank_prev', 'team_size', 'foreigners', 'mean_age', 'mean_value', 'total_value', 'distance']

    # Perform the 2SLS analysis by stage
    results, summaries = analyze_2sls_by_stage(stages_df, outcome_var, instr_var, treatment_var, control_vars, display)

    # Plot the causal effect for all rounds
    plot_causal_effect(results, country_plots_dir, display)
