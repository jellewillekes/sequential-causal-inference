# Load necessary libraries
library(MASS)  # for mvrnorm
library(Formula)
library(tibble)
library(dplyr)

# Set seed for reproducibility
set.seed(42)

# Load the dataset
data <- read.csv("Germany/cup_simulated.csv")

# Ensure the data is in the correct format
data <- data %>%
  rename(
    team_rank = team_rank,
    team_win_round1 = team_win_round1,
    team_win_round2 = team_win_round2,
    team_win_round3 = team_win_round3,
    team_win_round4 = team_win_round4,
    team_win_round5 = team_win_round5,
    team_win_round6 = team_win_round6,
    team_better_round1 = team_better_round1,
    team_better_round2 = team_better_round2,
    team_better_round3 = team_better_round3,
    team_better_round4 = team_better_round4,
    team_better_round5 = team_better_round5,
    team_better_round6 = team_better_round6
  )

# Define the formula for iv_factorial
formula <- as.formula("team_rank ~ team_win_round1 + team_win_round2 + team_win_round3 + team_win_round4 + team_win_round5 + team_win_round6 | team_better_round1 + team_better_round2 + team_better_round3 + team_better_round4 + team_better_round5 + team_better_round6")

# Load the iv_factorial function here if not already in the environment
# source("path_to_iv_factorial_function.R")

# Test the iv_factorial function
out <- iv_factorial(formula, data = data, method = "lm", level = 0.95)
summary(out)
