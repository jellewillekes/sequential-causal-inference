import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
np.random.seed(42)

# Number of rounds in the cup competition
N_rounds = 6

# Number of teams
N_teams = 2 ** N_rounds

# Number of years to simulate
N_years = 10

# Create an empty DataFrame to store all data across 10 years
all_years_data_wide = pd.DataFrame()

# Simulate the tournament for each year
for year in range(2012, 2012 + N_years):
    # Generate team identifiers and previous year's ranks (1 to 64 where 1 is the best)
    teams = np.arange(1, N_teams + 1)
    team_rank_prev = np.arange(1, N_teams + 1)

    # Assign team IDs
    team_ids = pd.DataFrame({'team_id': teams, 'team_rank_prev': team_rank_prev})

    # Determine current year's team rank based on the previous year's rank with a maximum change of 10 positions
    def calculate_team_rank(team_rank_prev):
        change = np.random.randint(-10, 11, size=len(team_rank_prev))
        team_rank = team_rank_prev + change
        team_rank = np.clip(team_rank, 1, np.inf)  # Ensure ranks do not go below 1
        team_rank = np.argsort(np.argsort(team_rank)) + 1  # Ensure ranks are unique
        return team_rank

    team_ids['team_rank'] = calculate_team_rank(team_ids['team_rank_prev'])

    # Function to simulate matches and outcomes for one round
    def simulate_round(teams, team_ranks, round_num):
        n_teams = len(teams)
        n_matches = n_teams // 2
        matches = np.random.permutation(teams).reshape(n_matches, 2)

        data = []

        for match in matches:
            team_a, team_b = match
            rank_a = team_ranks[team_a]
            rank_b = team_ranks[team_b]

            # Determine team_better based on previous year's ranks
            team_better_a = 1 if rank_a < rank_b else 0
            team_better_b = 1 - team_better_a

            # Simulate team_win
            team_win_a = np.random.choice([1, 0], p=[0.8, 0.2] if team_better_a == 1 else [0.2, 0.8])
            team_win_b = 1 - team_win_a

            data.append([team_a, team_b, round_num, team_better_a, team_win_a])
            data.append([team_b, team_a, round_num, team_better_b, team_win_b])

        df = pd.DataFrame(data, columns=['team_id', 'opponent', 'round', 'team_better', 'team_win'])
        winners = df[df['team_win'] == 1]['team_id'].values
        return df, winners

    # Initialize data frame for this year
    all_data = pd.DataFrame()

    # Simulate all rounds
    current_teams = teams
    team_ranks = team_ids.set_index('team_id')['team_rank'].to_dict()
    for r in range(1, N_rounds + 1):
        result, current_teams = simulate_round(current_teams, team_ranks, r)
        all_data = pd.concat([all_data, result])

    # Fill remaining rounds for eliminated teams with 0
    for team in teams:
        max_round = all_data[all_data['team_id'] == team]['round'].max()
        if max_round < N_rounds:
            for r in range(max_round + 1, N_rounds + 1):
                all_data = pd.concat([all_data, pd.DataFrame({
                    'team_id': [team],
                    'opponent': [np.nan],
                    'round': [r],
                    'team_better': [0],
                    'team_win': [0]
                })])

    # Merge team_ids to include team_rank_prev and team_rank
    all_data = all_data.merge(team_ids, on='team_id')

    # Convert to DataFrame for better display
    all_data = all_data.reset_index(drop=True)

    # Create wide format data for each round's variables
    all_data_wide = all_data.pivot_table(index=['team_id', 'team_rank_prev', 'team_rank'], columns='round',
                                         values=['team_better', 'team_win'], fill_value=0)
    all_data_wide.columns = [f'{col[0]}_round{col[1]}' for col in all_data_wide.columns]
    all_data_wide.reset_index(inplace=True)

    # Add a column for the year
    all_data_wide['year'] = year

    # Append this year's data to the combined dataset
    all_years_data_wide = pd.concat([all_years_data_wide, all_data_wide])

# Display the first few rows of the combined dataset
print(all_years_data_wide.head())

# Save the combined data to a CSV file
output_dir = "Germany"
os.makedirs(output_dir, exist_ok=True)
all_years_data_wide.to_csv(os.path.join(output_dir, f"cup_simulated_10_years_k{N_rounds}.csv"), index=False)
