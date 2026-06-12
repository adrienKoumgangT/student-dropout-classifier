import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Tuple, Optional, Union, List, Dict, Any

from src.config.configuration import load_config
from src.data.load_data import load_raw_data

"""
from columns_description import (application_mode, application_order, course, fathers_occupation, fathers_qualification,
                                 marital_status, mothers_occupation, mothers_qualification, nacionality,
                                 previous_qualification, categorical_mappings, binary_features)


def substitute_data(df: pd.DataFrame):
    # Apply mapping and cast categorical features to 'object' type
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype('object')

    # Since the source data is already 0/1, we just enforce the integer type
    for col in binary_features:
        df[col] = df[col].astype(int)

    return df


def remove_target_value(df: pd.DataFrame, target_feature_name: str = 'Target', target_feature_value: str = 'Enrolled'):
    df = df[df[target_feature_name] != target_feature_value]
    return df


def encode_binary_target(df: pd.DataFrame, target_feature_name: str = 'Target', target_feature_value: str = 'Dropout'):
    df[target_feature_name] = (df[target_feature_name] == target_feature_value).astype(int)
    return df
"""

# ----------





def get_mappings_and_binaries(config: dict) -> tuple:
    """
    Extract categorical mappings and binary feature names from config.

    Returns:
    --------
    tuple: (categorical_mappings, binary_features)
    """
    categorical_mappings = {}
    binary_features = []

    # Process binary features
    for feature in config.get('binary_features', []):
        # print(f"- feature = {feature}")
        col = feature.get('original_column') or feature.get('name')
        mapping = feature.get('mapping')
        if col:
            binary_features.append(col)
            if mapping:
                # Convert keys to match data type (handle YAML string keys)
                categorical_mappings[col] = {
                    # int(k) if k.lstrip('-').isdigit() else k: v
                    int(k): v
                    for k, v in mapping.items()
                }

    # Process ordinal features
    for feature in config.get('ordinal_features', []):
        col = feature.get('original_column') or feature.get('name')
        mapping = feature.get('mapping')
        if col and mapping:
            categorical_mappings[col] = {
                # int(k) if k.lstrip('-').isdigit() else k: v
                int(k): v
                for k, v in mapping.items()
            }

    # Process nominal features
    for feature in config.get('nominal_features', []):
        col = feature.get('original_column') or feature.get('name')
        mapping = feature.get('mapping')
        if col and mapping:
            categorical_mappings[col] = {
                # int(k) if k.lstrip('-').isdigit() else k: v
                int(k): v
                for k, v in mapping.items()
            }

    return categorical_mappings, binary_features


def substitute_data(df: pd.DataFrame, config_path: str = "feature_config.yaml") -> pd.DataFrame:
    """
    Substitute coded values with descriptive labels from config.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with coded values
    config_path : str
        Path to YAML configuration file

    Returns:
    --------
    pd.DataFrame : Dataframe with substituted values
    """
    # Load configuration
    config = load_config(config_path)
    categorical_mappings, binary_features = get_mappings_and_binaries(config)

    # Apply mappings to categorical columns
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            # Handle type mismatches between mapping keys and column values
            if df[col].dtype.kind in 'if':  # integer or float
                mapping = {int(k) if isinstance(k, str) else k: v
                           for k, v in mapping.items()}
            df[col] = df[col].map(mapping).astype('object')

    # Enforce binary features as integers
    #for col in binary_features:
    #    if col in df.columns:
    #        df[col] = df[col].astype(int)

    return df


def enforce_data_type(df: pd.DataFrame, config_path: str = "feature_config.yaml") -> pd.DataFrame:
    """
    Enforce correct data types based on feature_config.yaml.

    - Binary (0/1) features -> int
    - Ordinal features -> int
    - Nominal features (categorical codes) -> category (object-like)
    - Continuous features -> float
    """
    config = load_config(config_path)

    def get_cols(category: str) -> List[str]:
        """Extract column names from config category."""
        features = config.get(category, [])
        cols = []

        if isinstance(features, list):
            for item in features:
                if isinstance(item, dict):
                    cols.append(item.get('original_column') or item.get('name', ''))
                elif isinstance(item, str):
                    cols.append(item)
        elif isinstance(features, dict):
            for subgroup in features.values():
                if isinstance(subgroup, list):
                    for item in subgroup:
                        if isinstance(item, dict):
                            cols.append(item.get('original_column') or item.get('name', ''))
                        elif isinstance(item, str):
                            cols.append(item)
        return [c for c in cols if c]

    binary_cols = get_cols('binary_features')
    ordinal_cols = get_cols('ordinal_features')
    nominal_cols = get_cols('nominal_features')
    continuous_cols = get_cols('continuous_features')

    print("Enforcing data types...")

    # Binary -> int
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).clip(0, 1)
            print(f"  {col} -> int (binary)")

    # Ordinal -> int
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            print(f"  {col} -> int (ordinal)")

    # Nominal -> category (object-like, not numeric)
    for col in nominal_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype('object')
            print(f"  {col} -> category (nominal)")

    # Continuous -> float
    for col in continuous_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            print(f"  {col} -> float (continuous)")

    # Target -> category
    target_col = config.get('target', {}).get('original_column') or config.get('target', {}).get('name', 'Target')
    if target_col in df.columns:
        df[target_col] = df[target_col].astype('object')
        print(f"  {target_col} -> category (target)")

    print(f"Done. Final dtypes:\n{df.dtypes.value_counts()}")
    return df


def get_basic_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 60)
    print("DATASET BASIC INFORMATION")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nData Types:")
    print(df.dtypes.value_counts())

    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100)
    missing_df = pd.DataFrame({
        'Missing Count': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    })
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    if 'Target' in df.columns:
        print(f"\nTarget Distribution:")
        print(df['Target'].value_counts())
        print(f"\nTarget Proportions:")
        print(df['Target'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))


def check_data_quality(df: pd.DataFrame) -> Dict[str, List[str] | int]:
    """
    Perform data quality checks and report issues.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict : Dictionary of quality issues found
    """
    issues = {
        'missing_values': [],
        'duplicates': 0,
        'outliers': [],
        'invalid_values': [],
        'warnings': []
    }

    # Check missing values
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            issues['missing_values'].append(f"{col}: {missing} missing values")

    # Check duplicates
    issues['duplicates'] = df.duplicated().sum()

    # Check for invalid age values
    if 'Age at enrollment' in df.columns:
        age_col = df['Age at enrollment']
        if pd.api.types.is_numeric_dtype(age_col):
            age_min = float(age_col.min())
            age_max = float(age_col.max())
            if age_min < 15:
                issues['warnings'].append(f"Age at enrollment minimum is {age_min:.0f}")
            if age_max > 100:
                issues['warnings'].append(f"Age at enrollment maximum is {age_max:.0f}")

    # Check for invalid grades
    grade_cols = [col for col in df.columns if 'grade' in col.lower()]
    for col in grade_cols:
        if col not in df.columns:
            continue

        # Convert to numeric for safe comparison
        try:
            numeric_values = pd.to_numeric(df[col], errors='coerce')

            if numeric_values.isna().all():
                continue  # Skip completely non-numeric columns

            min_val = numeric_values.min()
            max_val = numeric_values.max()

            if pd.isna(min_val) or pd.isna(max_val):
                continue

            min_val = float(min_val)
            max_val = float(max_val)

            if min_val < 0:
                issues['invalid_values'].append(
                    f"{col} has negative values (min: {min_val:.2f})"
                )
            if max_val > 20:
                issues['warnings'].append(
                    f"{col} maximum is {max_val:.2f} (exceeds typical 0-20 scale)"
                )
        except Exception as e:
            issues['warnings'].append(f"{col}: Could not validate grade range - {str(e)}")

    # Check for unexpected values in binary columns
    binary_cols = ['Gender', 'Scholarship holder', 'Debtor',
                   'Tuition fees up to date', 'Displaced',
                   'Educational special needs', 'International',
                   'Daytime/evening attendance']

    for col in binary_cols:
        if col in df.columns:
            try:
                unique_vals = df[col].dropna().unique()
                # Convert to Python native types for comparison
                unique_vals_list = [float(v) if pd.api.types.is_number(v) else v
                                    for v in unique_vals]
                unexpected = [v for v in unique_vals_list if v not in [0, 1, 0.0, 1.0]]
                if unexpected:
                    issues['invalid_values'].append(
                        f"{col} has unexpected values: {unexpected}"
                    )
            except Exception as e:
                issues['warnings'].append(f"{col}: Could not validate binary values - {str(e)}")

    return issues


def print_quality_report(issues: Dict[str, List[str] | int]) -> None:
    """
    Print data quality report in a readable format.

    Parameters:
    -----------
    issues : dict
        Dictionary of quality issues from check_data_quality()
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    total_issues = (len(issues['missing_values']) +
                    issues['duplicates'] +
                    len(issues['outliers']) +
                    len(issues['invalid_values']) +
                    len(issues['warnings']))

    if total_issues == 0:
        print("\n No data quality issues found!")
        return

    if issues['missing_values']:
        print(f"\n  Missing Values ({len(issues['missing_values'])} columns):")
        for issue in issues['missing_values']:
            print(f"   - {issue}")

    if issues['duplicates'] > 0:
        print(f"\n  Duplicate Rows: {issues['duplicates']}")

    if issues['invalid_values']:
        print(f"\n Invalid Values ({len(issues['invalid_values'])} issues):")
        for issue in issues['invalid_values']:
            print(f"   - {issue}")

    if issues['outliers']:
        print(f"\n  Potential Outliers ({len(issues['outliers'])} detected):")
        for issue in issues['outliers']:
            print(f"   - {issue}")

    if issues['warnings']:
        print(f"\n  Warnings ({len(issues['warnings'])} items):")
        for issue in issues['warnings']:
            print(f"   - {issue}")


def clean_data(df: pd.DataFrame, drop_enrolled: bool = True) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values, duplicates, and filtering.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    drop_enrolled : bool
        Whether to remove 'Enrolled' students from the dataset

    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    initial_shape = df.shape

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows...")
        df = df.drop_duplicates()

    # Remove 'Enrolled' students if specified
    if drop_enrolled and 'Target' in df.columns:
        enrolled_count = (df['Target'] == 'Enrolled').sum()
        if enrolled_count > 0:
            print(f"Removing {enrolled_count} 'Enrolled' students...")
            df = df[df['Target'] != 'Enrolled']

    # Handle missing values (if any)
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Handling {missing_count} missing values...")

        # Fill missing numerical values with median
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

    final_shape = df.shape
    print(f"Data cleaning complete: {initial_shape} -> {final_shape}")
    print(f"Removed {initial_shape[0] - final_shape[0]} rows")

    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for better model performance.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with original features

    Returns:
    --------
    pd.DataFrame : Dataframe with additional engineered features
    """
    print("\n" + "=" * 60)
    print("CREATING ENGINEERED FEATURES")
    print("=" * 60)

    new_features = []

    # 1. First semester approval rate
    if all(col in df.columns for col in ['Curricular units 1st sem (approved)',
                                         'Curricular units 1st sem (enrolled)']):
        df['approval_rate_sem1'] = np.where(
            df['Curricular units 1st sem (enrolled)'] > 0,
            (df['Curricular units 1st sem (approved)'] /
             df['Curricular units 1st sem (enrolled)']) * 100,
            0
        )
        new_features.append('approval_rate_sem1')
        print(
            f"Created: approval_rate_sem1 (range: {df['approval_rate_sem1'].min():.0f}% - {df['approval_rate_sem1'].max():.0f}%)")

    # 2. Second semester approval rate
    if all(col in df.columns for col in ['Curricular units 2nd sem (approved)',
                                         'Curricular units 2nd sem (enrolled)']):
        df['approval_rate_sem2'] = np.where(
            df['Curricular units 2nd sem (enrolled)'] > 0,
            (df['Curricular units 2nd sem (approved)'] /
             df['Curricular units 2nd sem (enrolled)']) * 100,
            0
        )
        new_features.append('approval_rate_sem2')
        print(
            f"Created: approval_rate_sem2 (range: {df['approval_rate_sem2'].min():.0f}% - {df['approval_rate_sem2'].max():.0f}%)")

    # 3. Grade momentum (change from admission to 1st semester)
    if all(col in df.columns for col in ['Curricular units 1st sem (grade)',
                                         'Admission grade']):
        df['grade_momentum'] = (df['Curricular units 1st sem (grade)'] -
                                df['Admission grade'])
        new_features.append('grade_momentum')
        print(f"Created: grade_momentum (range: {df['grade_momentum'].min():.1f} - {df['grade_momentum'].max():.1f})")

    # 4. Grade decline from 1st to 2nd semester
    if all(col in df.columns for col in ['Curricular units 1st sem (grade)',
                                         'Curricular units 2nd sem (grade)']):
        df['grade_decline'] = (df['Curricular units 1st sem (grade)'] -
                               df['Curricular units 2nd sem (grade)'])
        new_features.append('grade_decline')
        print(f"Created: grade_decline (range: {df['grade_decline'].min():.1f} - {df['grade_decline'].max():.1f})")

    # 5. Evaluation intensity (evaluations per enrolled unit - 1st sem)
    if all(col in df.columns for col in ['Curricular units 1st sem (evaluations)',
                                         'Curricular units 1st sem (enrolled)']):
        df['evaluation_intensity_sem1'] = np.where(
            df['Curricular units 1st sem (enrolled)'] > 0,
            df['Curricular units 1st sem (evaluations)'] /
            df['Curricular units 1st sem (enrolled)'],
            0
        )
        new_features.append('evaluation_intensity_sem1')
        print(
            f"Created: evaluation_intensity_sem1 (range: {df['evaluation_intensity_sem1'].min():.1f} - {df['evaluation_intensity_sem1'].max():.1f})")

    # 6. Financial stress indicator (combination of financial features)
    if all(col in df.columns for col in ['Debtor', 'Tuition fees up to date',
                                         'Scholarship holder']):
        # Higher score = more financial stress
        df['financial_stress'] = (df['Debtor'] +
                                  (1 - df['Tuition fees up to date']) +
                                  (1 - df['Scholarship holder']))
        new_features.append('financial_stress')
        print(
            f"Created: financial_stress (range: {df['financial_stress'].min():.0f} - {df['financial_stress'].max():.0f})")

    # 7. Parent education level (combined)
    if all(col in df.columns for col in ["Mother's qualification",
                                         "Father's qualification"]):
        df['parent_education'] = (df["Mother's qualification"] +
                                  df["Father's qualification"]) / 2
        new_features.append('parent_education')
        print(
            f"Created: parent_education (range: {df['parent_education'].min():.1f} - {df['parent_education'].max():.1f})")

    # 8. Credits utilization (credited vs enrolled - 1st sem)
    if all(col in df.columns for col in ['Curricular units 1st sem (credited)',
                                         'Curricular units 1st sem (enrolled)']):
        df['credit_utilization_sem1'] = np.where(
            df['Curricular units 1st sem (enrolled)'] > 0,
            df['Curricular units 1st sem (credited)'] /
            df['Curricular units 1st sem (enrolled)'],
            0
        )
        new_features.append('credit_utilization_sem1')

    # 9. Course load (total enrolled units across semesters)
    if all(col in df.columns for col in ['Curricular units 1st sem (enrolled)',
                                         'Curricular units 2nd sem (enrolled)']):
        df['total_course_load'] = (df['Curricular units 1st sem (enrolled)'] +
                                   df['Curricular units 2nd sem (enrolled)'])
        new_features.append('total_course_load')

    print(f"\nTotal new features created: {len(new_features)}")
    print(f"New features: {new_features}")

    return df


def prepare_data(filepath: str = "data/raw/dataset.csv",
                 drop_enrolled: bool = True,
                 create_features: bool = True,
                 save_processed: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Complete data preparation pipeline.

    Parameters:
    -----------
    filepath : str
        Path to the raw CSV file
    drop_enrolled : bool
        Whether to remove 'Enrolled' students
    create_features : bool
        Whether to create engineered features
    save_processed : bool
        Whether to save the processed dataframe

    Returns:
    --------
    tuple : (processed_dataframe, config_dictionary)
    """
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Step 1: Load configuration
    config = load_config()

    # Step 2: Load raw data
    df = load_raw_data(filepath)

    # Step 3: Get basic information
    get_basic_info(df)

    # Step 4: Check data quality
    issues = check_data_quality(df)
    print_quality_report(issues)

    # Step 5: Clean data
    df = clean_data(df, drop_enrolled=drop_enrolled)

    # Step 6: Create engineered features
    if create_features:
        df = create_engineered_features(df)

    # Step 7: Save processed data
    if save_processed:
        processed_path = Path("data/processed")
        processed_path.mkdir(parents=True, exist_ok=True)
        output_path = processed_path / "df_cleaned.csv"
        df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")

    print(f"\nData preparation complete!")
    print(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    return df, config


# Utility function for quick column inspection
def inspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame of all columns with their properties.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame : Column summary
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'Dtype': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Null %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Sample Values': [str(df[col].dropna().unique()[:5].tolist()) for col in df.columns]
    })

    return summary

