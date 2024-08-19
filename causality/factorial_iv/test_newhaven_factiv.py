import pandas as pd
import numpy as np
from factiv import FactorialIV

# Load the newhaven dataset in Python
newhaven = pd.read_csv("newhaven.csv")

print(newhaven.info())
print(newhaven.head())

# Define the outcome, treatment, and instrument variables as numpy arrays
outcome = newhaven['turnout_98'].to_numpy()
treatment = newhaven[['inperson', 'phone']].to_numpy()
instrument = newhaven[['inperson_rand', 'phone_rand']].to_numpy()

# Initialize and fit the FactorialIV model
model = FactorialIV(outcome, treatment, instrument)
model.summary()

# Get a tidy summary with confidence intervals
tidy_results = model.tidy(conf_int=True)
print(tidy_results)

# # Calculate compliance profiles
# covariates = ['age', 'maj_party', 'turnout_96']
# cov_profile = compliance_profile(['inperson', 'phone'], ['inperson_rand', 'phone_rand'], covariates, newhaven)
# print(cov_profile)
