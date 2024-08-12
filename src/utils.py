import pandas as pd

def read_data(data_path, features=None, target=None):
    if features and target:
        df = pd.read_csv(data_path, usecols=[*feature, target])
    else:
        df = pd.read_csv(data_path)
    return df
