import os
import yaml
import pandas as pd


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key


def load_csv(file_path):
    return pd.read_csv(file_path)


def load_mappings_from_yaml(filename):
    file_path = os.path.join(project_root(), filename)
    with open(file_path, 'r') as file:
        mappings = yaml.safe_load(file)
    return mappings


def load_league_mappings(country):
    with open(os.path.join(project_root(), 'settings', f'mapping_{country.lower()}.yaml'), 'r') as file:
        league_mappings = yaml.safe_load(file)
    return league_mappings.get(country)


def load_stages(country):
    # Load the stage order from the YAML file
    with open(os.path.join(project_root(), 'settings', 'stages.yaml'), 'r') as file:
        stages_data = yaml.safe_load(file)
    return stages_data.get(country)
