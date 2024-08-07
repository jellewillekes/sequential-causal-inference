# Sequential Causal Inference

## Project Overview

This project aims to contribute to existing models by developing and integrating methodologies for sequential causal inference, particularly focusing on sequential treatments and randomization while addressing both linear and nonlinear relationships between the treatment variable and the outcome variable. By combining the Factorial IV framework and Causal Random Forests, we aim to provide robust and flexible tools for sequential causal inference. As a practical application, we analyze the causal effects of participating in domestic cup competitions on a team's performance in the league and in subsequent fixtures.

## Analysis Components

### 1. Analyzing Causal Effect of Domestic Cup Participation

**Objective:** 
To determine the impact of playing in domestic cup competitions on both long-term league performance and short-term next match performance.

**Approach:**
- Treat each round of the domestic cup competition as a binary factor.
- Use the opponent's randomly assigned rank as the instrumental variable.
- Evaluate both immediate effects (next match) and accumulated effects (end-of-season league performance).

**Instrumental Variables:**
1. **Rank Difference (Z1):** The difference in rank between team \(i\) and opponent team \(j\) based on the previous year's rank.
2. **Binary Indicator (Z2):** A binary variable indicating if team \(i\) is better (1) or worse (0) than opponent \(j\).

**Treatment Variable:**
- **Treatment Uptake (D):** The actual match result (win or lose) in round \(r\) of the domestic cup.

**Outcome Variables:**
1. **Long-term Outcome (Y1):** The league standing of team \(i\) at the end of the season.
2. **Short-term Outcome (Y2):** The result of games played within 5 days after the domestic cup match (win/lose).

### 2. Factorial IV Framework for Sequential Treatment and Randomization

**Objective:**
To utilize the factorial IV approach for sequential treatments, accounting for noncompliance in a setup where the treatment is the outcome of fixtures in each round.

**Key Components:**
- **Treatment Assignment (Z1, Z2):** Rank difference and binary indicator of opponent faced in each round.
- **Treatment Uptake (D):** Actual match result (win or lose).
- **Outcome (Y1, Y2):** Team’s rank in the domestic league at the end of the season and winning games within 5 days after the cup match.

**Assumptions:**
  - **Random Assignment:** Ensures independence between the instrument and potential outcomes.
  - **Exclusion Restriction:** The opponent rank impacts league performance only through match outcomes.
  - **Monotonicity:** Facing a weaker opponent cannot decrease the probability of winning.

**Methodology:**
- Define treatment combinations and handle noncompliance.
- Calculate ITT effects for both individual and population levels.
- Estimate the causal effects using IV estimands.

**Note:** The IV estimation typically assumes a linear relationship between the treatment and the outcome, making it suitable for scenarios where this assumption holds.

### 3. Causal Random Forest Method

**Objective:**
To leverage Causal Random Forests for non-parametric estimation of heterogeneous treatment effects.

**Key Components:**
- **Local Moment Conditions:** Solve local moment equations to estimate parameters.
- **Adaptive Weights:** Use forest-derived weights to enhance precision.
- **Recursive Partitioning:** Grow trees to maximize heterogeneity in the parameter of interest.
- **Inference:** Construct confidence intervals using asymptotic normality results and bootstrapping.

**Note:** Causal Random Forests do not assume a linear relationship, making them capable of capturing complex, nonlinear interactions between treatment and outcome.

## Loading the Data

Getting started with loading and preprocessing the data for this analysis is straightforward. Follow the steps below to set up the data for evaluating the impact of cup games on league performance.

<div align="left">

### Countries and Their Cups

<a href="#" style="display: inline-block; padding: 8px 16px; margin: 4px; background-color: #4CAF50; color: white; border-radius: 4px; text-decoration: none;">England</a>
<a href="#" style="display: inline-block; padding: 8px 16px; margin: 4px; background-color: #FF5722; color: white; border-radius: 4px; text-decoration: none;">FA_Cup</a>
<a href="#" style="display: inline-block; padding: 8px 16px; margin: 4px; background-color: #795548; color: white; border-radius: 4px; text-decoration: none;">League_Cup</a>

<a href="#" style="display: inline-block; padding: 8px 16px; margin: 4px; background-color: #2196F3; color: white; border-radius: 4px; text-decoration: none;">Germany</a>
<a href="#" style="display: inline-block; padding: 8px 16px; margin: 4px; background-color: #9C27B0; color: white; border-radius: 4px; text-decoration: none;">DFB_Pokal</a>

<a href="#" style="display: inline-block; padding: 8px 16px; margin: 4px; background-color: #FFC107; color: white; border-radius: 4px; text-decoration: none;">Netherlands</a>
<a href="#" style="display: inline-block; padding: 8px 16px; margin: 4px; background-color: #607D8B; color: white; border-radius: 4px; text-decoration: none;">KNVB_Beker</a>

</div>

### Steps

1. **Fetch Raw Data**:
    ```bash
    python main.py run_request_raw_data <country>
    ```
    Replace `<country>` with the desired country name from the supported list (England, Germany, Netherlands) to download the initial raw data.

    **Example**:
    ```bash
    python main.py run_request_raw_data Germany
    ```

2. **Preprocess Data**:
    ```bash
    python main.py run_preprocess_data <country> <cup>
    ```
    Replace `<country>` with the country name and `<cup>` with the cup competition name from the supported list paired above to preprocess the data for analysis.

    **Example**:
    ```bash
    python main.py run_preprocess_data Germany DFB_Pokal
    ```

These commands will handle all necessary steps:
- **Fetch raw data**: Downloads the data specific to the country.
- **Analyze Cup Data**: Processes cup competition data.
- **Analyze League Data**: Processes league standings and fixtures data.
- **Load Financial Data**: Gathers financial information of the teams.
- **Calculate Distances**: Computes travel distances for fixtures.
- **Combine Data**: Merges all data sources into a dataset ready for analysis.


