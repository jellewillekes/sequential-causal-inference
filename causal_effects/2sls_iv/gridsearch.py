import os
import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler
from iv_2sls import load_processed_data, ensure_country_plots_dir, analyze_2sls_by_stage, plot_causal_effect
from data.preprocess.imputation import impute_data


def ensure_country_plots_dir(country):
    country_plots_dir = os.path.join("plots", country, "Gridsearch")  # Modify this line
    os.makedirs(country_plots_dir, exist_ok=True)
    return country_plots_dir


def standardize_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def upscale_causal_effects(causal_effects, scaler, column):
    std = scaler.scale_[scaler.feature_names_in_ == column][0]
    return causal_effects * std


def grid_search(country, cup):
    # Load the processed DataFrame
    stages_df = load_processed_data(country, cup)
    stages_df = impute_data(stages_df, method='drop')

    # Define variables for standardization
    all_vars = ['team_rank', 'opponent_rank_prev', 'rank_diff', 'distance', 'team_win',
                'team_size', 'foreigners', 'mean_age', 'total_value', 'team_home', 'extra_time']

    # Standardize the data and keep the scaler
    stages_df, scaler = standardize_data(stages_df, all_vars)

    # Ensure country-specific plots directory exists
    country_plots_dir = ensure_country_plots_dir(country)

    # Define variables for the 2SLS analysis
    outcome_var = 'team_rank'
    instr_vars = ['rank_diff', 'team_better', 'distance']
    treatment_var = 'team_win'
    all_control_vars = ['team_size', 'foreigners', 'mean_age', 'total_value', 'distance', 'team_home', 'extra_time']

    # Generate all combinations of control variables with 2, 3, 4, 5, 6, and 7 elements
    control_var_combinations = []
    for r in range(2, 8):
        control_var_combinations.extend(itertools.combinations(all_control_vars, r))

    results = []
    grid_id = 0  # Initialize grid_id

    for instr_var in instr_vars:
        for control_var_combo in control_var_combinations:
            # Ensure the instrument variable is not included in the control variables
            if instr_var in control_var_combo:
                continue

            control_var_combo = list(control_var_combo)
            grid_id += 1  # Increment grid_id for each combination
            print(f'Testing Instrument: {instr_var} with Controls: {control_var_combo}, Grid ID: {grid_id}')
            result, _ = analyze_2sls_by_stage(stages_df, outcome_var, instr_var, treatment_var, control_var_combo,
                                              display="")
            for stage_result in result:
                # Upscale the causal effect
                stage_result['causal_effect'] = upscale_causal_effects(stage_result['2sls_iv'], scaler, treatment_var)

                # Add grid_id, instrument, and control_vars to the result
                stage_result.update({
                    'grid_id': grid_id,
                    'instrument': instr_var,
                    'control_vars': control_var_combo,
                    'significant': (stage_result['f_stat'] > 8 and stage_result['f_p_value'] < 0.10 and stage_result[
                        'p_value'] < 0.10)
                })
                results.append(stage_result)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv(os.path.join(country_plots_dir, 'grid_search_results.csv'), index=False)

        # Calculate the number of significant stages for each grid_id
        grid_scores = results_df.groupby('grid_id')['significant'].sum().to_dict()

        # Print all combinations with the highest number of significant stages
        max_significant_stages = max(grid_scores.values())
        best_grids = [grid for grid, score in grid_scores.items() if score == max_significant_stages]

        for grid in best_grids:
            best_results = results_df[results_df['grid_id'] == grid]
            best_control_vars = best_results.iloc[0]['control_vars']
            best_instr_var = best_results.iloc[0]['instrument']

            print(f"Grid ID: {grid}")
            print(f"Instrument: {best_instr_var}")
            print(f"Control Variables: {best_control_vars}")
            print(f"Number of significant stages: {max_significant_stages}")
            for index, row in best_results.iterrows():
                print(
                    f"Stage: {row['stage']}, Causal Effect: {row['causal_effect']}, Std Error: {row['std_error']}, P-Value: {row['p_value']}, R-Squared: {row['r_squared']}, F-Stat: {row['f_stat']}, F P-Value: {row['f_p_value']}, Significant: {row['significant']}")

            # Run the analysis again with the current best combination
            best_result_analysis, best_summaries = analyze_2sls_by_stage(stages_df, outcome_var, best_instr_var,
                                                                         treatment_var,
                                                                         best_control_vars, display="summary plot")

            # Save the plot with a unique filename
            plot_filename = f"grid_{grid}_instr_{best_instr_var}_controls_{'_'.join(best_control_vars)}.png"
            plot_causal_effect(best_result_analysis, country_plots_dir, display="plot", filename=plot_filename)

            # Save the results for all stages
            result_filename = f"grid_{grid}_results.csv"
            result_filepath = os.path.join(country_plots_dir, result_filename)
            best_results.to_csv(result_filepath, index=False)

            # Save the regression summaries for the best grid
            summary_filename = f"grid_{grid}_best_summaries.txt"
            with open(os.path.join(country_plots_dir, summary_filename), 'w') as f:
                for stage, summary in best_summaries.items():
                    f.write(f"Stage: {stage}\n")
                    f.write("First-stage Regression Summary:\n")
                    f.write(summary['first_stage_summary'] + "\n\n")
                    f.write("Second-stage Regression Summary:\n")
                    f.write(summary['second_stage_summary'] + "\n\n")

            print("Best Result Saved:")
            print(f"Grid ID: {grid}")
            print(f"Instrument: {best_instr_var}")
            print(f"Control Variables: {best_control_vars}")
            for index, row in best_results.iterrows():
                print(
                    f"Stage: {row['stage']}, Causal Effect: {row['causal_effect']}, Std Error: {row['std_error']}, P-Value: {row['p_value']}, R-Squared: {row['r_squared']}, F-Stat: {row['f_stat']}, F P-Value: {row['f_p_value']}, Significant: {row['significant']}")
    else:
        print("No significant results found.")


if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    grid_search(country, cup)
