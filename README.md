# Effect of Cup Football on League Performance

## Project Overview

This repository contains the analysis code for the paper *"Evaluating the Effect of Cup Participation on League Performance: Evidence from European Domestic Cups"* by Jelle Willekes. The study investigates the impact of domestic cup participation and winning on subsequent league performance, using random cup draws as an exogenous variation source. The analysis spans the FA Cup (England), DFB Pokal (Germany), KNVB Beker (Netherlands), and Taca de Portugal (Portugal).

## Key Objectives

1. **Causal Effect of Cup Participation**: Evaluate whether participation in cup competitions affects league performance, focusing on both immediate (within 5 days) and end-of-season outcomes.
2. **Impact of Cup Wins**: Determine how winning individual cup matches influences a team’s momentum and subsequent league results.
3. **Instrumental Variable Approach**: Use the randomness of opponent rank and division from cup draws as instruments to estimate causal effects.

## Methodology

### Instrumental Variables and Two-Stage Least Squares (2SLS)

- **Instrumental Variables (IV)**:
  - **Opponent Rank**: Rank of the opponent based on the prior season’s standing.
  - **Opponent Division**: Division level of the opponent team.
- **Assumptions**:
  - **Exogeneity**: Cup match draws are random, meaning opponent characteristics are unrelated to unobserved factors in league performance.
  - **Exclusion Restriction**: The effect of the opponent’s characteristics on league outcomes occurs solely through match participation, not through other pathways.
- **2SLS Analysis**:
  - **First Stage**: Model cup participation or win probabilities based on the opponent’s rank and division.
  - **Second Stage**: Estimate league performance impact using predicted cup participation from the first stage.

## Data

- Data covers cup and league fixtures from 2012 to 2023 across England, Germany, the Netherlands, and Portugal.
- Sources include API-Football for match details and Transfermarkt for team characteristics.
- **Data Filtering**: Only league fixtures within 5 days of a cup match are analyzed to isolate immediate effects, avoiding confounding factors from other matches.

### Key Variables
- **Outcome Variables**:
  - **League Points**: Points obtained in the league fixture following a cup match.
  - **End-of-Season Rank**: Team’s final league standing.
- **Treatment Variables**:
  - **Cup Participation**: Binary indicator of a team’s participation in a given cup round.
  - **Cup Win**: Binary outcome indicating a win in a cup round.
- **Control Variables**: Distance traveled, recovery days, team size, market value, and foreign player percentage.

## Analysis Components

1. **Estimating the Effect of Cup Participation**:
   - Compares league performance following a cup match to league performance without cup congestion.
   
2. **Effect of Winning a Cup Round**:
   - Evaluates the momentum boost effect on league performance, focusing on different cup rounds.

3. **Robustness Checks**:
   - **Distance as an IV**: Examines the effect of travel fatigue by adding distance as an instrumental variable.
   - **Heterogeneity Analysis**: Assesses how effects vary based on team characteristics like market value and squad size.

## Usage

1. **Fetch Raw Data**:
    ```bash
    python main.py run_request_raw_data <country>
    ```
   Replace `<country>` with one of the supported countries (England, Germany, Netherlands) to fetch raw data.

2. **Preprocess Data**:
    ```bash
    python main.py run_preprocess_data <country> <cup>
    ```
   Replace `<country>` with the country name and `<cup>` with the competition name (e.g., FA_Cup) for data preparation.

## Results Summary

Initial findings indicate that winning cup matches boosts league performance, particularly for lower-market-value teams. However, merely participating in cup fixtures shows no significant league performance impact, implying that fixture congestion alone does not substantially hinder league outcomes.
