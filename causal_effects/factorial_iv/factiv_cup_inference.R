# Set the working directory
setwd("C:/Users/JelleWillekes/Documents/GitHub/causal-inference/causal_effects/factorial_iv")

# Check the working directory to ensure it's correctly set
print(getwd())

# Verify that the file exists in the specified location
file_path <- "Germany/DFB_Pokal_preprocessed.csv"
if (!file.exists(file_path)) {
  stop(paste("File does not exist:", file_path))
}

# Load necessary libraries
library(factiv)
library(here)
library(dplyr)

# Function to get the project root
get_project_root <- function() {
  return(here::here())
}

# Set the parameters
country <- 'Germany'
cup <- 'DFB_Pokal'
outcome_var <- 'team_rank'
treatment_var <- 'team_win'
instrument_var <- 'team_better'
round_specific_controls <- c('team_home')  # Control variables that change each round
season_specific_controls <- c('team_size')  # Control variables that stay the same for the season
num_rounds <- 6  # Adjust based on the number of rounds in the competition

# Get the project root
project_root <- get_project_root()

# Define the file path using the project root
file_path <- file.path(project_root, "Germany", paste0(cup, "_preprocessed.csv"))

# Load the preprocessed data
data <- read.csv(file_path)

# Check column names to ensure all necessary columns are present
print("Column names in the dataset:")
print(colnames(data))

# Generate the formula for the treatment and instrument variables
treatment_vars <- paste0(treatment_var, "_round", 1:num_rounds)
instrument_vars <- paste0(instrument_var, "_round", 1:num_rounds)

# Generate the formula for the round-specific control variables
round_control_vars <- unlist(sapply(round_specific_controls, function(ctrl) {
  paste0(ctrl, "_round", 1:num_rounds)
}))

# Combine all parts of the formula
all_vars <- c(treatment_vars, instrument_vars, round_control_vars, season_specific_controls)

# Ensure all required variables are present in the data
required_vars <- unique(c(outcome_var, treatment_vars, instrument_vars, round_control_vars, season_specific_controls))
missing_vars <- setdiff(required_vars, colnames(data))
if (length(missing_vars) > 0) {
  stop(paste("Missing variables in data:", paste(missing_vars, collapse = ", ")))
}

# Adjust the formula string dynamically based on available columns
available_treatment_vars <- intersect(treatment_vars, colnames(data))
available_instrument_vars <- intersect(instrument_vars, colnames(data))
available_round_control_vars <- intersect(round_control_vars, colnames(data))

# Create the formula string
formula_string <- paste(outcome_var, "~", paste(available_treatment_vars, collapse = " + "), "|", paste(available_instrument_vars, collapse = " + "))
formula <- as.formula(formula_string)

print("Formula for analysis:")
print(formula)

# Check for missing values in the relevant columns
print("Missing values in relevant columns:")
missing_values <- sapply(data[, all_vars, drop = FALSE], function(x) sum(is.na(x)))
print(missing_values)

# Remove rows with missing values in any of the relevant columns
data_clean <- data[complete.cases(data[, c(outcome_var, available_treatment_vars, available_instrument_vars, round_control_vars, season_specific_controls)]), ]

# Inspect the dimensions of the outcome, treatment, and instrument matrices after cleaning
print("Dimensions of outcome, treatment, and instrument matrices after cleaning:")
Y <- data_clean[[outcome_var]]
D <- data_clean[, available_treatment_vars, drop = FALSE]
Z <- data_clean[, available_instrument_vars, drop = FALSE]

print(paste("Y (outcome) dimensions:", length(Y)))
print(paste("D (treatment) dimensions:", dim(D)))
print(paste("Z (instrument) dimensions:", dim(Z)))

# Ensure there are no missing values in the outcome variable
if (any(is.na(Y))) {
  stop("Missing values in the outcome variable.")
}

# Ensure there are no missing values in the treatment and instrument variables
if (any(is.na(D))) {
  stop("Missing values in the treatment variables.")
}
if (any(is.na(Z))) {
  stop("Missing values in the instrument variables.")
}

# Ensure that the dimensions of D and Z match
if (nrow(D) != nrow(Z) || ncol(D) != ncol(Z)) {
  stop("The dimensions of the treatment (D) and instrument (Z) matrices do not match.")
}

# Perform the factorial IV analysis
result <- iv_factorial(formula, data = data_clean)

# Print the summary of the results
summary(result)

# Save the results to a file
result_path <- file.path(project_root, paste0(cup, "_factorial_iv_results.txt"))
sink(result_path)
print(summary(result))
sink()

# Add compliance profile analysis separately
cov_prof <- compliance_profile(
  ~ win_round1 + win_round2 + win_round3 + win_round4 + win_round5 + win_round6 | 
    rank_opponent_round1 + rank_opponent_round2 + rank_opponent_round3 + rank_opponent_round4 + rank_opponent_round5 + rank_opponent_round6 |
    team_home_round1 + team_home_round2 + team_home_round3 + team_home_round4 + team_home_round5 + team_home_round6 + team_size,
  data = data_clean
)

print(cov_prof)
