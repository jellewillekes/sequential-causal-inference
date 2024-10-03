import os
import pandas as pd
import numpy as np
from scipy import stats
from utils.load import project_root


def load_processed_data(country, cup):
    file_path = os.path.join(project_root(), 'data/process', country, f'{cup}_processed.csv')
    return pd.read_csv(file_path)


def create_bins(df, column_name, labels=['Low', 'Medium', 'High']):
    """Create bins for continuous variables, with a special case for distance."""

    if column_name == 'distance':
        # Step 1: Create a new bin column for the distance
        df[f'{column_name}_bins'] = 'Home'  # Assign 'Home' for distances equal to zero

        # Step 2: For non-zero values, apply qcut and assign the categories
        non_zero_distances = df[column_name] > 0
        df.loc[non_zero_distances, f'{column_name}_bins'] = pd.qcut(
            df.loc[non_zero_distances, column_name], q=3, labels=labels
        )
    else:
        # Default binning for non-distance columns
        df[f'{column_name}_bins'] = pd.cut(df[column_name], bins=3, labels=labels)

    return df


def perform_f_test(data, outcome_var, treatment_var):
    """Performs an F-test (Welch's t-test) to compare means for two groups (e.g., Cup Win vs No Cup Win)."""
    group1 = data[data[treatment_var] == 1][outcome_var]
    group2 = data[data[treatment_var] == 0][outcome_var]
    f_value, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test for unequal variances
    return f_value, p_value


def summary_statistics(data, group_var, treatment_var, outcome_var, categories=None, analysis_type=None):
    """Generates summary statistics: mean, standard error, count, and F-test p-value."""
    summary_stats = []

    if categories is not None:
        for category in categories:
            data_category = data[data[group_var] == category]
            grouped_data = data_category.groupby(treatment_var)[outcome_var].agg(['mean', 'std', 'count']).reset_index()
            _, p_value = perform_f_test(data_category, outcome_var, treatment_var)

            for i, row in grouped_data.iterrows():
                std_err = row['std'] / np.sqrt(row['count']) if row['count'] > 0 else np.nan
                summary_stats.append({
                    'Analysis Type': analysis_type,
                    'Variable': group_var,
                    'Category': category,
                    'Group': 'Win' if row[treatment_var] == 1 else 'No Win',
                    'Mean (SE)': f"{row['mean']:.2f} ({std_err:.2f})",
                    'Observations': row['count'],
                    'p-value': p_value
                })
    else:
        # For continuous variables without categories
        grouped_data = data.groupby(treatment_var)[outcome_var].agg(['mean', 'std', 'count']).reset_index()
        _, p_value = perform_f_test(data, outcome_var, treatment_var)

        for i, row in grouped_data.iterrows():
            std_err = row['std'] / np.sqrt(row['count']) if row['count'] > 0 else np.nan
            summary_stats.append({
                'Analysis Type': analysis_type,
                'Variable': group_var,
                'Category': 'All',
                'Group': 'Win' if row[treatment_var] == 1 else 'No Win',
                'Mean (SE)': f"{row['mean']:.2f} ({std_err:.2f})",
                'Observations': row['count'],
                'p-value': p_value
            })

    return summary_stats


def summary_statistics_outcome_only(data, outcome_var, treatment_var, analysis_type=None):
    """Generates summary statistics without F-tests for the given outcome variable."""
    summary_stats = []
    grouped_data = data.groupby(treatment_var)[outcome_var].agg(['mean', 'std', 'count']).reset_index()

    for i, row in grouped_data.iterrows():
        std_err = row['std'] / np.sqrt(row['count']) if row['count'] > 0 else np.nan
        summary_stats.append({
            'Analysis Type': analysis_type,
            'Variable': outcome_var,
            'Category': 'All',
            'Group': 'Win' if row[treatment_var] == 1 else 'No Win',
            'Mean (SE)': f"{row['mean']:.2f} ({std_err:.2f})",
            'Observations': row['count'],
            'p-value': None  # No p-value as requested
        })

    return summary_stats


def save_summary_stats_to_csv(summary_stats, filename):
    """Saves summary statistics to CSV for future LaTeX table creation."""
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(filename, index=False)
    print(f"Summary statistics saved to {filename}")


if __name__ == "__main__":
    country = 'combined'
    cup = 'combined_cup'

    # Load and preprocess data
    cup_fixtures = load_processed_data(country, cup)

    # Filter for participation and cup win analyses
    participation_outcome_var = 'next_team_points_round_plus'
    participation_treatment_var = 'team_win'
    cup_win_outcome_var = 'next_team_points_round'
    cup_win_treatment_var = 'team_win'

    # Dynamically assign the correct recovery days column for each analysis
    recovery_days_participation_var = 'next_fixture_days_round_plus'
    recovery_days_cup_win_var = 'next_fixture_days_round'

    # Filter data
    cup_fixtures_participation = cup_fixtures[cup_fixtures[recovery_days_participation_var] <= 5].dropna(
        subset=['distance', recovery_days_participation_var])

    cup_fixtures_cup_win = cup_fixtures[cup_fixtures[recovery_days_cup_win_var] <= 5].dropna(
        subset=['distance', recovery_days_cup_win_var])

    # New: Print counts for Participation vs Non-Participation
    participation_counts = cup_fixtures_participation[participation_treatment_var].value_counts()
    print(f"Participation vs Non-Participation counts:\n{participation_counts}")
    print(f"Total observations for Participation analysis: {len(cup_fixtures_participation)}\n")

    # New: Print counts for Cup Win vs No Cup Win
    cup_win_counts = cup_fixtures_cup_win[cup_win_treatment_var].value_counts()
    print(f"Cup Win vs No Cup Win counts:\n{cup_win_counts}")
    print(f"Total observations for Cup Win analysis: {len(cup_fixtures_cup_win)}\n")

    # Variables to analyze
    variables = ['total_value', 'mean_value', 'team_size', 'foreigners', 'mean_age', 'distance']
    extra_time = 'extra_time'

    # Create bins for each variable
    for var in variables:
        cup_fixtures_participation = create_bins(cup_fixtures_participation, var)
        cup_fixtures_cup_win = create_bins(cup_fixtures_cup_win, var)

    # Process participation and cup win summary statistics for each variable
    summary_stats_all = []

    for var in variables:
        # Adding 'Home' to the categories for distance
        categories = ['Home', 'Low', 'Medium', 'High'] if var == 'distance' else ['Low', 'Medium', 'High']

        # Participation analysis
        participation_summary_stats = summary_statistics(cup_fixtures_participation, f'{var}_bins',
                                                         participation_treatment_var,
                                                         var, categories,
                                                         'Participation')
        summary_stats_all.extend(participation_summary_stats)

        # Cup win analysis
        cup_win_summary_stats = summary_statistics(cup_fixtures_cup_win, f'{var}_bins', cup_win_treatment_var,
                                                   var, categories, 'Cup Win')
        summary_stats_all.extend(cup_win_summary_stats)

    # For recovery_days and extra_time without bins (mean comparison only)
    recovery_days_participation = summary_statistics(cup_fixtures_participation, recovery_days_participation_var,
                                                     participation_treatment_var, recovery_days_participation_var, None,
                                                     'Participation')
    summary_stats_all.extend(recovery_days_participation)

    extra_time_participation = summary_statistics(cup_fixtures_participation, extra_time,
                                                  participation_treatment_var, extra_time, None, 'Participation')
    summary_stats_all.extend(extra_time_participation)

    recovery_days_cup_win = summary_statistics(cup_fixtures_cup_win, recovery_days_cup_win_var, cup_win_treatment_var,
                                               recovery_days_cup_win_var, None, 'Cup Win')
    summary_stats_all.extend(recovery_days_cup_win)

    extra_time_cup_win = summary_statistics(cup_fixtures_cup_win, extra_time, cup_win_treatment_var,
                                            extra_time, None, 'Cup Win')
    summary_stats_all.extend(extra_time_cup_win)

    # New: Calculate summary stats for next_team_points_round and next_team_points_round_plus
    next_points_participation_stats = summary_statistics_outcome_only(cup_fixtures_participation,
                                                                      'next_team_points_round_plus',
                                                                      participation_treatment_var,
                                                                      'Participation')
    summary_stats_all.extend(next_points_participation_stats)

    next_points_cup_win_stats = summary_statistics_outcome_only(cup_fixtures_cup_win,
                                                                'next_team_points_round',
                                                                cup_win_treatment_var,
                                                                'Cup Win')
    summary_stats_all.extend(next_points_cup_win_stats)

    # Save all summary statistics to one CSV
    save_summary_stats_to_csv(summary_stats_all, "combined_summary_stats_with_analysis_type.csv")

