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


def filter_by_team_size(data, percentile=20):
    """Split data into top and bottom 20% by team size."""
    top_cutoff = data['team_size'].quantile(1 - percentile / 100)
    bottom_cutoff = data['team_size'].quantile(percentile / 100)

    top_team_size = data[data['team_size'] >= top_cutoff]
    bottom_team_size = data[data['team_size'] <= bottom_cutoff]

    return top_team_size, bottom_team_size


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


def get_top_bottom_teams_by_value(data, country_col, value_col, top_pct=0.2, bottom_pct=0.2):
    data = data.copy()
    top_teams = []
    bottom_teams = []
    border_values = {}

    for country in data[country_col].unique():
        country_data = data[data[country_col] == country]
        top_value = country_data[value_col].quantile(1 - top_pct)
        bottom_value = country_data[value_col].quantile(bottom_pct)
        border_values[country] = {'top_value': top_value, 'bottom_value': bottom_value}

        top_teams.append(country_data[country_data[value_col] >= top_value])
        bottom_teams.append(country_data[country_data[value_col] <= bottom_value])

    top_teams = pd.concat(top_teams)
    bottom_teams = pd.concat(bottom_teams)

    return top_teams, bottom_teams, border_values


def run_analysis(cup_fixtures, outcome_var, instr_vars, treatment_var, control_vars_list, value_col, value_type):
    # Split data into top and bottom 20% by specified value (market value or team size)
    top_teams, bottom_teams, value_borders = get_top_bottom_teams_by_value(
        cup_fixtures, country_col='country_code', value_col=value_col, top_pct=0.2, bottom_pct=0.2)

    # Print the border values for each country
    print(f"\n{value_type} borders per country (top 20% and bottom 20%):")
    for country, values in value_borders.items():
        print(f"{country}: Top 20% >= {values['top_value']}, Bottom 20% <= {values['bottom_value']}")

    # Perform 2SLS analysis for top 20% teams
    print(f"\nRunning 2SLS analysis for top 20% {value_type} teams:")
    top_results = analyze_2sls_by_stage(top_teams, outcome_var, instr_vars, treatment_var, control_vars_list)

    # Perform 2SLS analysis for bottom 20% teams
    print(f"\nRunning 2SLS analysis for bottom 20% {value_type} teams:")
    bottom_results = analyze_2sls_by_stage(bottom_teams, outcome_var, instr_vars, treatment_var, control_vars_list)

    # Save results to CSV
    results_top_df = pd.DataFrame(top_results)
    results_bottom_df = pd.DataFrame(bottom_results)

    top_results_file_path = os.path.join("results", "combined", "2SLS_Results", f"top_20_{value_type}_2sls_results.csv")
    results_top_df.to_csv(top_results_file_path, index=False)

    bottom_results_file_path = os.path.join("results", "combined", "2SLS_Results",
                                            f"bottom_20_{value_type}_2sls_results.csv")
    results_bottom_df.to_csv(bottom_results_file_path, index=False)

    print(f"\nTop 20% {value_type} results saved to {top_results_file_path}")
    print(f"Bottom 20% {value_type} results saved to {bottom_results_file_path}")


if __name__ == "__main__":
    country = 'combined'
    cup = 'combined_cup'
    display = "summary"

    outcome_var = 'next_team_points_round'

    cup_fixtures = load_processed_data(country, cup)
    cup_fixtures = cup_fixtures.dropna(subset='distance')

    instr_vars = ['opponent_league_rank_prev', 'opponent_division']
    treatment_var = 'team_win'
    control_vars_list = [
        [],  # Model 1: No control variables
        ['team_league_rank_prev'],  # Model 2
        ['team_league_rank_prev', 'distance'],  # Model 3
        ['team_league_rank_prev', 'distance', 'next_fixture_days'],  # Model 4
        ['team_league_rank_prev', 'distance', 'next_fixture_days', 'extra_time'],  # Model 5
        ['team_league_rank_prev', 'distance', 'next_fixture_days', 'extra_time', 'team_size', 'total_value',
         'mean_age'],  # Model 6
        ['team_league_rank_prev', 'distance', 'next_fixture_days', 'extra_time', 'team_size', 'total_value', 'mean_age',
         'country_code'],  # Model 7
    ]

    # Split data by market value (Top 20% and Bottom 20%)
    top_market_value, bottom_market_value = filter_by_market_value(cup_fixtures)

    # Analyze Top 20%
    print("Analyzing Top 20% Market Value Teams:")
    results_top = analyze_2sls_by_stage(top_market_value, outcome_var, instr_vars, treatment_var, control_vars_list,
                                        display)
    results_df_top = pd.DataFrame(results_top)
    top_file_path = os.path.join("results", country, "2SLS_Results",
                                 f"combined_2sls_top20_market_value_{outcome_var}.csv")
    results_df_top.to_csv(top_file_path, index=False)
    print(f"Top 20% results saved to {top_file_path}")

    # Analyze Bottom 20%
    print("Analyzing Bottom 20% Market Value Teams:")
    results_bottom = analyze_2sls_by_stage(bottom_market_value, outcome_var, instr_vars, treatment_var,
                                           control_vars_list, display)
    results_df_bottom = pd.DataFrame(results_bottom)
    bottom_file_path = os.path.join("results", country, "2SLS_Results",
                                    f"combined_2sls_bottom20_market_value_{outcome_var}.csv")
    results_df_bottom.to_csv(bottom_file_path, index=False)
    print(f"Bottom 20% results saved to {bottom_file_path}")

    # Split data by team size (Top 20% and Bottom 20%)
    top_team_size, bottom_team_size = filter_by_team_size(cup_fixtures)

    # Analyze Top 20%
    print("Analyzing Top 20% Team Size:")
    results_top = analyze_2sls_by_stage(top_team_size, outcome_var, instr_vars, treatment_var, control_vars_list,
                                        display)
    results_df_top = pd.DataFrame(results_top)
    top_file_path = os.path.join("results", country, "2SLS_Results", f"combined_2sls_top20_team_size_{outcome_var}.csv")
    results_df_top.to_csv(top_file_path, index=False)
    print(f"Top 20% results saved to {top_file_path}")

    # Analyze Bottom 20%
    print("Analyzing Bottom 20% Team Size:")
    results_bottom = analyze_2sls_by_stage(bottom_team_size, outcome_var, instr_vars, treatment_var, control_vars_list,
                                           display)
    results_df_bottom = pd.DataFrame(results_bottom)
    bottom_file_path = os.path.join("results", country, "2SLS_Results",
                                    f"combined_2sls_bottom20_team_size_{outcome_var}.csv")
    results_df_bottom.to_csv(bottom_file_path, index=False)
    print(f"Bottom 20% results saved to {bottom_file_path}")
