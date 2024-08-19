import os
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils.load import project_root
from data.process.imputation import impute_data


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


def plot_causal_effect(variables, results, country_plots_dir, display="none", filename="causal_effect_by_stage.png"):
    def format_controls(control_vars):
        # Split the control variables into chunks of 4 and join with line breaks
        lines = []
        for i in range(0, len(control_vars), 4):
            line = ", ".join(control_vars[i:i + 4])
            lines.append(line)
        return "\n            ".join(lines)  # Indent continuation lines to align with "Controls:"

    results_df = pd.DataFrame(results)
    if "plot" in display:
        plt.figure(figsize=(12, 8))  # Adjust figure size to ensure there's space for both legend and annotation

        # Plot each round's causal effect and add to the legend individually
        for index, row in results_df.iterrows():
            plt.errorbar(row['stage'], row['2sls_iv'], yerr=row['std_error'], fmt='o', capsize=5, alpha=0.6,
                         label=f"Round {int(row['stage'])}: p-value: {row['p_value']:.2f}, F-stat: {row['f_stat']:.1f}, "
                               f"$R^2$: {row['r_squared']:.2f}")

        # Extract information from the variables dictionary
        outcome_var = variables.get('outcome_var')
        treatment_var = variables.get('treatment_var')
        instrument_var = variables.get('instrument_var')
        control_vars = variables.get('control_vars', [])

        if outcome_var == 'team_rank_diff':
            plot_title = 'Causal Effect of Advancing per Round on League Standing'
        elif outcome_var == 'next_team_points':
            plot_title = 'Causal Effect of Advancing per Round on Next Fixture Performance'
        else:
            plot_title = 'Causal Effect of Advancing per Round on Performance'

        # Format control variables into multiple lines if needed
        formatted_controls = format_controls(control_vars)

        # Create the annotation text with manual line breaks and left alignment
        annotation_text = (f"Outcome:    {outcome_var}\n"
                           f"Treatment:  {treatment_var}\n"
                           f"Instrument: {instrument_var}\n"
                           f"Controls:   {formatted_controls}")

        # Adjust the placement of the legend to make space for the annotation
        plt.legend(loc='upper left', bbox_to_anchor=(0, -0.25),
                   fontsize='small', frameon=True, prop={'family': 'monospace'})

        # Add the text annotation below the plot on the right side
        plt.annotate(annotation_text, xy=(0.55, -0.35), xycoords='axes fraction',
                     ha='left', va='top', fontsize='medium', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", edgecolor='black', alpha=0.3, facecolor='white'))

        plt.title(plot_title)
        plt.xlabel('Round')
        plt.ylabel('Causal Effect (LATE)')
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust layout to make space for both elements
        plt.savefig(os.path.join(country_plots_dir, filename), bbox_inches='tight')
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
