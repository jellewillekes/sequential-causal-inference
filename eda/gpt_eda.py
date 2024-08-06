import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the provided CSV file
file_path = '/mnt/data/DFB_Pokal_processed.csv'
df = pd.read_csv(file_path)

# Define key variables
key_vars = ['team_win', 'rank_diff', 'team_better', 'team_rank', 'next_team_points']

# 1. Distribution of Key Variables
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for i, var in enumerate(key_vars):
    axs[i//3, i%3].hist(df[var].dropna(), bins=30, edgecolor='k')
    axs[i//3, i%3].set_title(f'Distribution of {var}')
    axs[i//3, i%3].set_xlabel(var)
    axs[i//3, i%3].set_ylabel('Frequency')
fig.tight_layout()
plt.savefig('/mnt/data/distribution_histograms.png')
plt.show()

# Define stages to analyze
stages = sorted(df['stage'].unique())

# Calculate average next team points for winning and losing each stage
average_points_stage = []
for stage in stages:
    avg_points_win = df[(df['stage'] == stage) & (df['team_win'] == 1)]['next_team_points'].mean()
    avg_points_lose = df[(df['stage'] == stage) & (df['team_win'] == 0)]['next_team_points'].mean()
    average_points_stage.append((stage, avg_points_win, avg_points_lose))
average_points_df = pd.DataFrame(average_points_stage, columns=['stage', 'avg_points_win', 'avg_points_lose'])

# 2. Effect of Winning on Next League Match Performance
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(average_points_df['stage'], average_points_df['avg_points_win'], marker='o', label='Win in Stage')
ax.plot(average_points_df['stage'], average_points_df['avg_points_lose'], marker='x', label='Lose in Stage')
ax.set_title('Average Next League Game Points by Cup Stage Result')
ax.set_xlabel('Cup Stage')
ax.set_ylabel('Average Points')
ax.legend()
plt.tight_layout()
plt.savefig('/mnt/data/average_points_stage.png')
plt.show()

# Calculate average rank change for winning and losing each stage
average_rank_change_stage = []
for stage in stages:
    avg_rank_change_win = df[(df['stage'] == stage) & (df['team_win'] == 1)]['team_rank_diff'].mean()
    avg_rank_change_lose = df[(df['stage'] == stage) & (df['team_win'] == 0)]['team_rank_diff'].mean()
    average_rank_change_stage.append((stage, avg_rank_change_win, avg_rank_change_lose))
average_rank_change_df = pd.DataFrame(average_rank_change_stage, columns=['stage', 'avg_rank_change_win', 'avg_rank_change_lose'])

# 3. Effect of Winning on Season-End Rank Change
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(average_rank_change_df['stage'], average_rank_change_df['avg_rank_change_win'], marker='o', label='Win in Stage')
ax.plot(average_rank_change_df['stage'], average_rank_change_df['avg_rank_change_lose'], marker='x', label='Lose in Stage')
ax.set_title('Average Rank Change by Cup Stage Result')
ax.set_xlabel('Cup Stage')
ax.set_ylabel('Average Rank Change')
ax.legend()
plt.tight_layout()
plt.savefig('/mnt/data/average_rank_change_stage.png')
plt.show()

# 4. Instrumental Variable (Rank Difference) vs. Outcome
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(x='rank_diff', y='team_rank', hue='team_win', data=df, ax=axs[0])
axs[0].set_title('Rank Difference vs. Team Rank')
axs[0].set_xlabel('Rank Difference')
axs[0].set_ylabel('Team Rank')
sns.scatterplot(x='rank_diff', y='next_team_points', hue='team_win', data=df, ax=axs[1])
axs[1].set_title('Rank Difference vs. Next Team Points')
axs[1].set_xlabel('Rank Difference')
axs[1].set_ylabel('Next Team Points')
fig.tight_layout()
plt.savefig('/mnt/data/scatter_plots_rank_diff.png')
plt.show()

# 5. Average Treatment Effects by Opponent Rank
df['rank_diff_bin'] = pd.cut(df['rank_diff'], bins=10)
average_metrics_binned = df.groupby('rank_diff_bin').agg({
    'next_team_points': ['mean'],
    'team_rank_diff': ['mean']
}).reset_index()
average_metrics_binned.columns = ['rank_diff_bin', 'avg_next_team_points', 'avg_team_rank_diff']

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].bar(average_metrics_binned['rank_diff_bin'].astype(str), average_metrics_binned['avg_next_team_points'])
axs[0].set_title('Average Next Team Points by Binned Opponent Rank Difference')
axs[0].set_xlabel('Opponent Rank Difference (Binned)')
axs[0].set_ylabel('Average Next Team Points')
axs[0].tick_params(axis='x', rotation=45)
axs[1].bar(average_metrics_binned['rank_diff_bin'].astype(str), average_metrics_binned['avg_team_rank_diff'])
axs[1].set_title('Average Rank Change by Binned Opponent Rank Difference')
axs[1].set_xlabel('Opponent Rank Difference (Binned)')
axs[1].set_ylabel('Average Rank Change')
axs[1].tick_params(axis='x', rotation=45)
fig.tight_layout()
plt.savefig('/mnt/data/average_metrics_binned_opponent_rank.png')
plt.show()

# 6. Team Performance Analysis by Round
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='stage', y='team_win', hue='team_better', data=df, ax=axs[0], ci=None)
axs[0].set_title('Team Win vs. Team Better per Round')
axs[0].set_xlabel('Cup Stage')
axs[0].set_ylabel('Win Rate')
sns.barplot(x='stage', y='team_win', hue='rank_diff', data=df, ax=axs[1], ci=None)
axs[1].set_title('Team Win vs. Rank Difference per Round')
axs[1].set_xlabel('Cup Stage')
axs[1].set_ylabel('Win Rate')
fig.tight_layout()
plt.savefig('/mnt/data/team_win_comparison.png')
plt.show()
