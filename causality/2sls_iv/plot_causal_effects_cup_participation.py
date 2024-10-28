import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'results/combined/2SLS_Results/combined_2sls_results_next_team_points_round_plus.csv'
df_results = pd.read_csv(file_path)

# Filter the dataset for "Model 7" only and limit the stages to rounds 2 to 6
df_model7_filtered = df_results[(df_results['model'] == 'Model 7') & (df_results['stage'].between(2, 6))]

# Set plot size for a professional-looking figure
plt.figure(figsize=(10, 6))

# Plot causal effect (second stage coefficient) with error bars (second stage std error)
causal_effect_line, = plt.plot(df_model7_filtered['stage'], df_model7_filtered['second_stage_coefficient'],
                               color='black', label='Causal Effect', linestyle='-', marker='o')
plt.errorbar(df_model7_filtered['stage'], df_model7_filtered['second_stage_coefficient'],
             yerr=df_model7_filtered['second_stage_std_error'], fmt='o-', capsize=5, color='black')

# Highlight points where F-stat > 10 and set the size of both crosses equally (including Round 4)
significant_stages_filtered = df_model7_filtered[df_model7_filtered['first_stage_f_stat'] > 9]
cross_markers = plt.scatter(significant_stages_filtered['stage'], significant_stages_filtered['second_stage_coefficient'],
                            color='red', marker='x', s=200)

# Adding labels and title
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Cup Round', fontsize=12)
plt.ylabel('Causal Effect (Points in Next League Fixture)', fontsize=12)
plt.title('Causal Effect of Cup Participation on League Performance', fontsize=14)

# Set integer x-ticks for the 'stage' variable
plt.xticks(range(2, 7))  # Adjust this range as needed based on your data

# Adding a legend showing Causal Effect as black line and F-stat > 10 as red crosses
plt.legend([causal_effect_line, cross_markers], ['Causal Effect', 'F-stat > 9'], loc='upper right')

# Saving or displaying the plot
plt.tight_layout()
plt.savefig('causal_effect_cup_participation.png', dpi=300)
plt.show()
