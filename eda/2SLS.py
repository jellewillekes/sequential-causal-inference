import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from raw_data.loader import get_project_root
import statsmodels.api as sm

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


def analyze_2sls_by_stage(stages_df):
    unique_stages = stages_df['Stage'].unique()
    results = []

    for stage in unique_stages:
        df_stage = stages_df[stages_df['Stage'] == stage].copy()

        # First Stage Regression
        X1 = sm.add_constant(df_stage[['Rank_diff']])
        first_stage_model = sm.OLS(df_stage['Team_win'], X1).fit()
        df_stage['D_hat'] = first_stage_model.predict(X1)

        print(f'Stage {stage} First-Stage Regression Summary:')
        print(first_stage_model.summary())

        # Second Stage Regression
        X2 = sm.add_constant(df_stage[['D_hat', 'Team_rank_prev']])
        second_stage_model = sm.OLS(df_stage['Team_rank'], X2).fit()
        results.append((stage, second_stage_model.params['D_hat'], second_stage_model.bse['D_hat']))

        print(f'Stage {stage} Second-Stage Regression Summary:')
        print(second_stage_model.summary())

    return results

def plot_causal_effect(results, country_plots_dir):
    results_df = pd.DataFrame(results, columns=['Stage', 'Causal_Effect', 'Std_Error'])
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['Stage'], results_df['Causal_Effect'], yerr=results_df['Std_Error'], fmt='o', capsize=5)
    plt.title('Causal Effect of Advancing in Each Round on Final League Performance')
    plt.xlabel('Stage')
    plt.ylabel('Causal Effect')
    plt.grid(True)
    plt.savefig(os.path.join(country_plots_dir, 'causal_effect_by_stage.png'))
    plt.show()

if __name__ == "__main__":
    country = 'Germany'
    cup = 'DFB_Pokal'

    # Load the processed DataFrame
    stages_df = load_processed_data(country, cup)

    # Ensure country-specific plots directory exists
    country_plots_dir = ensure_country_plots_dir(country)

    # Perform the 2SLS analysis by stage
    results = analyze_2sls_by_stage(stages_df)

    # Plot the causal effect for all rounds
    plot_causal_effect(results, country_plots_dir)