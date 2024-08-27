import os
import pandas as pd
from utils.load import project_root, load_csv
from plot import *

# Load data
country = 'Netherlands'
cup = 'KNVB_Beker'
data = load_csv(os.path.join(project_root(), 'data', 'process', country, f'{cup}_fixtures.csv'))


print(data.head())


data.groupby(['year', 'round'])['team_id'].count().unstack(level='round').to_csv('rankings_KNVB.csv')
