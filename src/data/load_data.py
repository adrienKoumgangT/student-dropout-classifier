import os

import pandas as pd

raw_data_path = os.path.join(os.getcwd(), '..', 'data', 'raw')
processed_data_path = os.path.join(os.getcwd(), '..', 'data', 'processed')


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the raw CSV file

    Returns:
    --------
    pd.DataFrame : Raw dataframe
    """
    filepath = os.path.join(raw_data_path, filename)
    print(f"Loading data from: {filepath}")

    try:

        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. "
                                f"Please place your CSV file in the data/raw/ directory.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the raw CSV file

    Returns:
    --------
    pd.DataFrame : Raw dataframe
    """
    filepath = os.path.join(processed_data_path, filename)
    print(f"Loading data from: {filepath}")

    try:

        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. "
                                f"Please place your CSV file in the data/processed/ directory.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

