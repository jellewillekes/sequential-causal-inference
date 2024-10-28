import os
import pandas as pd
import statsmodels.api as sm
from utils.load import project_root


def load_processed_data(country, cup):
    file_path = os.path.join(project_root(), 'data/process', country, f'{cup}_processed.csv')
    return pd.read_csv(file_path)


def ensure_results_dir(country):
    results_dir = os.path.join("results", country, "2SLS_Results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def filter_by_market_value(data, percentile=20):
    """Split data into top and bottom 20% by market value (total_value)."""
    top_cutoff = data['total_value'].quantile(1 - percentile / 100)
    bottom_cutoff = data['total_value'].quantile(percentile / 100)

    top_market_value = data[data['total_value'] >= top_cutoff]
    bottom_market_value = data[data['total_value'] <= bottom_cutoff]

    return top_market_value, bottom_market_value


def perform_2sls_analysis(data, outcome_var, instr_vars, treatment_var, control_vars, display="none"):
    X1 = sm.add_constant(data[instr_vars + control_vars])
    first_stage_model = sm.OLS(data[treatment_var], X1).fit()
    data['D_hat'] = first_stage_model.predict(X1)

    if "summary" in display:
        print('First-stage Regression Summary:')
        print(first_stage_model.summary())

    X2 = sm.add_constant(data[['D_hat'] + control_vars])
    second_stage_model = sm.OLS(data[outcome_var], X2).fit()

    if "summary" in display:
        print('Second-stage Regression Summary:')
        print(second_stage_model.summary())

    first_stage_stats = {
        'instrument_coefficients': first_stage_model.params[instr_vars].values,
        'instrument_std_errors': first_stage_model.bse[instr_vars].values,
        'instrument_t_values': first_stage_model.tvalues[instr_vars].values,
        'instrument_p_values': first_stage_model.pvalues[instr_vars].values,
        'r_squared': first_stage_model.rsquared,
        'f_stat': first_stage_model.fvalue,
        'f_p_value': first_stage_model.f_pvalue,
        'nobs': first_stage_model.nobs
    }

    second_stage_stats = {
        'endogenous_coefficient': second_stage_model.params['D_hat'],
        'endogenous_std_error': second_stage_model.bse['D_hat'],
        'endogenous_t_value': second_stage_model.tvalues['D_hat'],
        'endogenous_p_value': second_stage_model.pvalues['D_hat'],
        'r_squared': second_stage_model.rsquared,
        'nobs': second_stage_model.nobs
    }

    return first_stage_stats, second_stage_stats


def analyze_2sls_by_stage(stages_df, outcome_var, instr_vars, treatment_var, control_vars_list, display="none"):
    unique_stages = stages_df['stage'].unique()
    unique_stages.sort()

    results = []

    for stage in unique_stages:
        df_stage = stages_df[stages_df['stage'] == stage].copy()

        for i, current_controls in enumerate(control_vars_list):
            if "summary" in display:
                print(f'Running Stage {stage}, Model {i + 1} with controls: {current_controls}')

            first_stage_stats, second_stage_stats = perform_2sls_analysis(df_stage, outcome_var, instr_vars,
                                                                          treatment_var, current_controls, display)

            for j, instr in enumerate(instr_vars):
                results.append({
                    'stage': stage,
                    'model': f'Model {i + 1}',
                    'instrument': instr,
                    'first_stage_coefficient': first_stage_stats['instrument_coefficients'][j],
                    'first_stage_std_error': first_stage_stats['instrument_std_errors'][j],
                    'first_stage_t_value': first_stage_stats['instrument_t_values'][j],
                    'first_stage_p_value': first_stage_stats['instrument_p_values'][j],
                    'first_stage_r_squared': first_stage_stats['r_squared'],
                    'first_stage_f_stat': first_stage_stats['f_stat'],
                    'first_stage_f_p_value': first_stage_stats['f_p_value'],
                    'first_stage_nobs': first_stage_stats['nobs'],
                    'second_stage_coefficient': second_stage_stats['endogenous_coefficient'],
                    'second_stage_std_error': second_stage_stats['endogenous_std_error'],
                    'second_stage_t_value': second_stage_stats['endogenous_t_value'],
                    'second_stage_p_value': second_stage_stats['endogenous_p_value'],
                    'second_stage_r_squared': second_stage_stats['r_squared'],
                    'second_stage_nobs': second_stage_stats['nobs']
                })

    return results


def count_nans(data, columns):
    nan_counts = data[columns].isna().sum()
    nan_summary = pd.DataFrame(nan_counts, columns=['NaN Count'])
    return nan_summary


if __name__ == "__main__":
    country = 'combined'
    cup = 'combined_cup'
    display = "summary"

    outcome_var = 'next_team_points_round'

    cup_fixtures = load_processed_data(country, cup)

    if outcome_var == 'next_team_points_round_plus':
        cup_fixtures = cup_fixtures[cup_fixtures['next_fixture_days_round_plus'] <= 5]
        cup_fixtures = cup_fixtures.dropna(subset=['distance', 'next_fixture_days_round_plus'])
        days_var = 'next_fixture_days_round_plus'
    else:
        cup_fixtures = cup_fixtures[cup_fixtures['next_fixture_days_round'] <= 5]
        cup_fixtures = cup_fixtures.dropna(subset=['distance', 'next_fixture_days_round'])
        days_var = 'next_fixture_days_round'

    instr_vars = ['opponent_league_rank_prev', 'opponent_division']
    treatment_var = 'team_win'
    control_vars_list = [
        [],  # Model 1: No control variables
        ['team_league_rank_prev'],  # Model 2
        ['team_league_rank_prev', 'distance'],  # Model 3
        ['team_league_rank_prev', 'distance', days_var],  # Model 4
        ['team_league_rank_prev', 'distance', days_var, 'extra_time'],  # Model 5
        ['team_league_rank_prev', 'distance', days_var, 'extra_time', 'team_size', 'total_value',
         'mean_age'],  # Model 6
        ['team_league_rank_prev', 'distance', days_var, 'extra_time', 'team_size', 'total_value',
         'mean_age',
         'country_code'],  # Model 7
    ]

    # Split data by market value (Top 20% and Bottom 20%)
    top_market_value, bottom_market_value = filter_by_market_value(cup_fixtures)

    # Analyze Top 20%
    print("Analyzing Top 20% Market Value Teams:")
    results_top = analyze_2sls_by_stage(top_market_value, outcome_var, instr_vars, treatment_var, control_vars_list, display)
    results_df_top = pd.DataFrame(results_top)
    top_file_path = os.path.join("results", country, "2SLS_Results", f"combined_2sls_top20_market_value_{outcome_var}.csv")
    results_df_top.to_csv(top_file_path, index=False)
    print(f"Top 20% results saved to {top_file_path}")

    # Analyze Bottom 20%
    print("Analyzing Bottom 20% Market Value Teams:")
    results_bottom = analyze_2sls_by_stage(bottom_market_value, outcome_var, instr_vars, treatment_var, control_vars_list, display)
    results_df_bottom = pd.DataFrame(results_bottom)
    bottom_file_path = os.path.join("results", country, "2SLS_Results", f"combined_2sls_bottom20_market_value_{outcome_var}.csv")
    results_df_bottom.to_csv(bottom_file_path, index=False)
    print(f"Bottom 20% results saved to {bottom_file_path}")

