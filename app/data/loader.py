import os
import pandas as pd

# Get the PROJECT ROOT (biomed-extractor/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory at top level
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_trials_csv(filename='example_trials.csv'):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records from {filename}")
    return df

def load_trials_json(filename='example_trials.json'):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_json(path)
    print(f"Loaded {len(df)} records from {filename}")
    return df

if __name__ == "__main__":
    df_csv = load_trials_csv()
    print(df_csv.head())

    df_json = load_trials_json()
    print(df_json.head())