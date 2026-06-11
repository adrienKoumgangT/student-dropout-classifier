import os

import pandas as pd

raw_data_path = os.path.join(os.getcwd(), 'data', 'raw')
processed_data_path = os.path.join(os.getcwd(), 'data', 'processed')

def load_data_raw(filename: str = 'student_dropout_academic_success.csv'):
    df = pd.read_csv(os.path.join(raw_data_path, filename))
    return df

def load_data_processed(filename: str = 'student_dropout_academic_success.csv'):
    df = pd.read_csv(os.path.join(processed_data_path, filename))
    return df

def save_data_processed(df: pd.DataFrame, filename: str = 'student_dropout_academic_success.csv'):
    df.to_csv(os.path.join(processed_data_path, filename), index=False)

