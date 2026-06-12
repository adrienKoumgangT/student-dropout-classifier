import os

import yaml


base_config_path = os.path.join(os.getcwd(), '..', 'config')


def load_config(config_path: str = "feature_config.yaml") -> dict:
    """
    Load feature configuration from YAML file.

    Parameters:
    -----------
    config_path : str
        Path to the feature configuration YAML file

    Returns:
    --------
    dict : Configuration dictionary
    """
    try:
        with open(os.path.join(base_config_path, config_path), 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using default configuration.")
        return get_default_config()


def get_default_config() -> dict:
    """
    Get default feature configuration if config file is not found.

    Returns:
    --------
    dict : Default configuration dictionary
    """
    return {
        'binary_features': [
            'Gender', 'Scholarship holder', 'Debtor',
            'Tuition fees up to date', 'Displaced',
            'Educational special needs', 'International',
            'Daytime/evening attendance'
        ],
        'ordinal_features': [
            "Mother's qualification", "Father's qualification"
        ],
        'nominal_features': [
            'Course', 'Application mode', 'Application order',
            "Mother's occupation", "Father's occupation",
            'Previous qualification', 'Marital status', 'Nacionality'
        ],
        'continuous_features': [
            'Age at enrollment',
            'Admission grade',
            'Previous qualification (grade)',
            'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
            'Curricular units 2nd sem (without evaluations)',
            'Unemployment rate',
            'Inflation rate',
            'GDP'
        ],
        'target': {
            'name': 'Target',
            'mapping': {'Dropout': 1, 'Graduate': 0}
        }
    }


