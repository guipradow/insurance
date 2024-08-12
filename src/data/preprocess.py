import pandas as pd
from src.utils import read_data
from src import config


def process_data(raw_data_path, processed_data_path):

    df_raw = read_data(raw_data_path)
    df_processed = df_raw.copy()
    df_processed[config.CATEGORICAL_FEATURES] = df_processed[
        config.CATEGORICAL_FEATURES
    ].astype('category')
    df_processed['medical_history'] = (
        df_processed['medical_history']
        .cat
        .add_categories(['No history'])
    )
    df_processed['family_medical_history'] = (
        df_processed['family_medical_history']
        .cat
        .add_categories(['No history'])
    )
    df_processed.loc[:, 'medical_history'] = (
        df_processed['medical_history']
        .fillna('No history')
    )
    df_processed.loc[:, 'family_medical_history'] = (
        df_processed['family_medical_history']
        .fillna('No history')
    )
    df_processed.to_csv(processed_data_path)
