import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path

from src.config.configuration import load_config


def encode_target(df: pd.DataFrame, target_col: str = 'Target', mapping: Optional[Dict] = None) -> pd.DataFrame:
    """
    Encode the target variable to binary (1=Dropout, 0=Graduate).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    mapping : dict, optional
        Custom mapping for target values

    Returns:
    --------
    pd.DataFrame : Dataframe with encoded target
    """
    print("\n" + "=" * 60)
    print("ENCODING TARGET VARIABLE")
    print("=" * 60)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    if mapping is None:
        mapping = {'Dropout': 1, 'Graduate': 0}

    # Check for unexpected values
    unique_targets = df[target_col].unique()
    for val in unique_targets:
        if val not in mapping:
            print(f"Warning: '{val}' not in mapping. Available mappings: {list(mapping.keys())}")

    # Create encoded target
    df['Target_encoded'] = df[target_col].map(mapping)

    # Check for unmapped values
    if df['Target_encoded'].isnull().any():
        unmapped = df[target_col][df['Target_encoded'].isnull()].unique()
        print(f"Error: Some target values could not be mapped: {unmapped}")
        raise ValueError(f"Could not map target values: {unmapped}")

    print(f"Target encoding: {mapping}")
    print(f"Distribution after encoding:")
    print(f"  Dropout (1):  {df['Target_encoded'].sum()} ({df['Target_encoded'].mean() * 100:.1f}%)")
    print(f"  Graduate (0): {(1 - df['Target_encoded']).sum()} ({(1 - df['Target_encoded'].mean()) * 100:.1f}%)")

    return df


def identify_feature_types(df: pd.DataFrame,
                           binary_cols: List[str],
                           ordinal_cols: List[str],
                           nominal_cols: List[str],
                           exclude_cols: List[str] = None) -> Tuple[List, List, List, List]:
    """
    Identify and categorize feature types from the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    binary_cols : list
        List of binary column names
    ordinal_cols : list
        List of ordinal column names
    nominal_cols : list
        List of nominal column names
    exclude_cols : list, optional
        Columns to exclude from categorization

    Returns:
    --------
    tuple : (binary_features, ordinal_features, nominal_features, continuous_features)
    """
    if exclude_cols is None:
        exclude_cols = ['Target', 'Target_encoded']

    # Filter to only columns that exist in the dataframe
    binary_features = [col for col in binary_cols
                       if col in df.columns and col not in exclude_cols]

    ordinal_features = [col for col in ordinal_cols
                        if col in df.columns and col not in exclude_cols]

    nominal_features = [col for col in nominal_cols
                        if col in df.columns and col not in exclude_cols]

    # All other numerical columns are continuous
    identified = binary_features + ordinal_features + nominal_features + exclude_cols
    continuous_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns
                           if col not in identified]

    # Also check for any new engineered features that are continuous
    for col in df.columns:
        if col not in identified and col not in continuous_features:
            if df[col].dtype in ['int64', 'float64']:
                continuous_features.append(col)

    print(f"\nFeature type identification:")
    print(f"  Binary features:     {len(binary_features)}")
    print(f"  Ordinal features:    {len(ordinal_features)}")
    print(f"  Nominal features:    {len(nominal_features)}")
    print(f"  Continuous features: {len(continuous_features)}")

    return binary_features, ordinal_features, nominal_features, continuous_features


def create_preprocessing_pipeline(binary_features: List[str],
                                  ordinal_features: List[str],
                                  nominal_features: List[str],
                                  continuous_features: List[str]) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing different feature types.

    Parameters:
    -----------
    binary_features : list
        Binary feature column names
    ordinal_features : list
        Ordinal feature column names
    nominal_features : list
        Nominal feature column names
    continuous_features : list
        Continuous feature column names

    Returns:
    --------
    ColumnTransformer : Preprocessing pipeline
    """
    print("\n" + "=" * 60)
    print("CREATING PREPROCESSING PIPELINE")
    print("=" * 60)

    transformers = []

    # One-hot encode nominal features
    if nominal_features:
        print(f"Adding OneHotEncoder for {len(nominal_features)} nominal features...")
        nominal_transformer = OneHotEncoder(
            drop='first',  # Avoid multicollinearity
            sparse_output=False,
            handle_unknown='ignore'  # Ignore unknown categories during prediction
        )
        transformers.append(('nominal', nominal_transformer, nominal_features))

    # Standard scale continuous + binary + ordinal features
    scale_features = continuous_features + binary_features + ordinal_features
    if scale_features:
        print(f"Adding StandardScaler for {len(scale_features)} features (continuous + binary + ordinal)...")
        scaler = StandardScaler()
        transformers.append(('scaler', scaler, scale_features))

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not explicitly specified
        verbose_feature_names_out=False
    )

    print(f"Preprocessing pipeline created with {len(transformers)} transformers")

    return preprocessor


def encode_features_for_tree(df: pd.DataFrame,
                             nominal_features: List[str],
                             ordinal_features: List[str]) -> pd.DataFrame:
    """
    Alternative encoding for tree-based models that don't require scaling.
    Uses Label Encoding for nominal features (works well with tree splits).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    nominal_features : list
        Nominal feature column names
    ordinal_features : list
        Ordinal feature column names

    Returns:
    --------
    pd.DataFrame : Label-encoded dataframe
    """
    print("\n" + "=" * 60)
    print("ENCODING FEATURES FOR TREE-BASED MODELS")
    print("=" * 60)

    df_encoded = df.copy()

    # Label encode nominal features
    label_encoders = {}
    for col in nominal_features:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Label encoded: {col} ({le.classes_.__len__()} categories)")

    # Ordinal features are kept as-is (already numerical codes)
    for col in ordinal_features:
        if col in df.columns:
            print(f"  Ordinal kept as-is: {col}")

    return df_encoded


def prepare_features_for_modeling(df: pd.DataFrame,
                                  config: Optional[Dict] = None,
                                  model_type: str = 'linear',
                                  exclude_cols: Optional[List[str]] = None,
                                  target_col: str = 'Target_encoded') -> Tuple:
    """
    Complete feature preparation pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with engineered features
    config : dict, optional
        Feature configuration dictionary
    model_type : str
        Type of model: 'linear' (requires scaling) or 'tree' (no scaling)
    exclude_cols : list, optional
        Columns to exclude from features
    target_col : str
        Name of the target column

    Returns:
    --------
    tuple : (X, y, preprocessor_or_none, feature_names)
    """
    print("\n" + "=" * 60)
    print(f"PREPARING FEATURES FOR {model_type.upper()} MODEL")
    print("=" * 60)

    # Load config if not provided
    if config is None:
        config = load_config()

    # Define columns to exclude
    if exclude_cols is None:
        exclude_cols = ['Target', 'Target_encoded']
        # Also exclude 2nd semester features for early prediction models
        # Uncomment the next line if you want to use only 1st semester data
        # exclude_cols += [col for col in df.columns if '2nd sem' in col]

    # Encode target if not already done
    if target_col not in df.columns and 'Target' in df.columns:
        df = encode_target(df)

    # Identify feature types
    binary_features, ordinal_features, nominal_features, continuous_features = \
        identify_feature_types(
            df,
            config.get('binary_features', []),
            config.get('ordinal_features', []),
            config.get('nominal_features', []),
            exclude_cols=exclude_cols
        )

    # Add any new engineered features
    engineered_features = [col for col in df.columns
                           if col not in binary_features + ordinal_features +
                           nominal_features + continuous_features + exclude_cols + [target_col]
                           and df[col].dtype in ['int64', 'float64']]

    if engineered_features:
        print(f"\nAdding {len(engineered_features)} engineered features as continuous:")
        for feat in engineered_features:
            print(f"  - {feat}")
        continuous_features = continuous_features + engineered_features

    # Separate features and target
    feature_cols = binary_features + ordinal_features + nominal_features + continuous_features
    X = df[feature_cols]
    y = df[target_col]

    print(f"\nFinal feature matrix: {X.shape[1]} features")

    if model_type == 'linear':
        # Create preprocessing pipeline (scaling + encoding)
        preprocessor = create_preprocessing_pipeline(
            binary_features, ordinal_features, nominal_features, continuous_features
        )

        # Fit and transform
        print("Applying preprocessing pipeline...")
        X_processed = preprocessor.fit_transform(X)

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        print(f"Processed feature matrix: {X_processed.shape[1]} features (after encoding)")
        return X_processed, y, preprocessor, feature_names

    elif model_type == 'tree':
        # For tree models, use label encoding for nominals
        X_processed = encode_features_for_tree(X, nominal_features, ordinal_features)

        # Drop the original nominal columns and keep the label-encoded versions
        # (They're already replaced in place by encode_features_for_tree)

        return X_processed.values, y, None, list(X_processed.columns)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'linear' or 'tree'.")


def split_data(X: Union[pd.DataFrame, np.ndarray],
               y: Union[pd.Series, np.ndarray],
               test_size: float = 0.2,
               val_size: float = 0.25,
               random_state: int = 42,
               stratify: bool = True) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Parameters:
    -----------
    X : DataFrame or array
        Feature matrix
    y : Series or array
        Target variable
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of training data for validation set
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to use stratified splitting

    Returns:
    --------
    tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)

    stratify_param = y if stratify else None

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    # Second split: separate validation from training
    stratify_param_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_param_temp
    )

    print(f"Training set:   {X_train.shape[0]} samples ({X_train.shape[0] / len(y) * 100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / len(y) * 100:.1f}%)")
    print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0] / len(y) * 100:.1f}%)")

    # Check class distribution
    for name, y_set in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        if isinstance(y_set, pd.Series):
            dropout_pct = y_set.mean() * 100
        else:
            dropout_pct = np.mean(y_set) * 100
        print(f"  {name} dropout rate: {dropout_pct:.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_encoded_data(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      output_dir: str = "data/processed",
                      prefix: str = "encoded",
                      save_metadata: bool = True) -> None:
    """
    Save encoded datasets to CSV files.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation target
    output_dir : str
        Directory to save the files
    prefix : str
        Prefix for saved filenames
    save_metadata : bool
        Whether to save metadata about the saved data
    """
    print("\n" + "=" * 60)
    print("SAVING ENCODED DATA")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame if numpy arrays
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    if X_val is not None and isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)

    # Combine features and target for each split
    train_df = X_train.copy()
    train_df['Target_encoded'] = y_train.values if isinstance(y_train, pd.Series) else y_train

    test_df = X_test.copy()
    test_df['Target_encoded'] = y_test.values if isinstance(y_test, pd.Series) else y_test

    # Save training data
    train_path = output_path / f"{prefix}_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f" Training data saved: {train_path}")
    print(f"   Shape: {train_df.shape}")

    # Save test data
    test_path = output_path / f"{prefix}_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f" Test data saved: {test_path}")
    print(f"   Shape: {test_df.shape}")

    # Save validation data if provided
    if X_val is not None and y_val is not None:
        val_df = X_val.copy()
        val_df['Target_encoded'] = y_val.values if isinstance(y_val, pd.Series) else y_val

        val_path = output_path / f"{prefix}_val.csv"
        val_df.to_csv(val_path, index=False)
        print(f" Validation data saved: {val_path}")
        print(f"   Shape: {val_df.shape}")

    # Save metadata
    if save_metadata:
        metadata = {
            'created_at': pd.Timestamp.now().isoformat(),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'val_samples': len(val_df) if X_val is not None else 0,
            'total_features': len(X_train.columns) if isinstance(X_train, pd.DataFrame) else X_train.shape[1],
            'target_column': 'Target_encoded',
            'class_distribution': {
                'train': {
                    'dropout_count': int(y_train.sum()),
                    'dropout_rate': float(y_train.mean() * 100)
                },
                'test': {
                    'dropout_count': int(y_test.sum()),
                    'dropout_rate': float(y_test.mean() * 100)
                }
            }
        }

        if X_val is not None:
            metadata['class_distribution']['val'] = {
                'dropout_count': int(y_val.sum()),
                'dropout_rate': float(y_val.mean() * 100)
            }

        metadata_path = output_path / f"{prefix}_metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        print(f"Metadata saved: {metadata_path}")


def load_encoded_data(data_dir: str = "data/processed",
                      prefix: str = "encoded") -> tuple:
    """
    Load previously saved encoded datasets.

    Parameters:
    -----------
    data_dir : str
        Directory containing the saved data
    prefix : str
        Prefix used when saving the files

    Returns:
    --------
    tuple : (X_train, y_train, X_test, y_test, [X_val, y_val])
        Returns validation data only if available
    """
    print("\n" + "=" * 60)
    print("LOADING ENCODED DATA")
    print("=" * 60)

    data_path = Path(data_dir)

    # Load training data
    train_path = data_path / f"{prefix}_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    y_train = train_df['Target_encoded']
    X_train = train_df.drop('Target_encoded', axis=1)
    print(f"Training data loaded: {X_train.shape}")

    # Load test data
    test_path = data_path / f"{prefix}_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_df = pd.read_csv(test_path)
    y_test = test_df['Target_encoded']
    X_test = test_df.drop('Target_encoded', axis=1)
    print(f"Test data loaded: {X_test.shape}")

    # Load validation data (optional)
    val_path = data_path / f"{prefix}_val.csv"
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        y_val = val_df['Target_encoded']
        X_val = val_df.drop('Target_encoded', axis=1)
        print(f"Validation data loaded: {X_val.shape}")
        return X_train, y_train, X_test, y_test, X_val, y_val

    return X_train, y_train, X_test, y_test


def save_preprocessor(preprocessor,
                      filepath: str = "models/preprocessor.pkl",
                      feature_names: Optional[List[str]] = None) -> None:
    """
    Save the fitted preprocessor to disk for later use.

    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    filepath : str
        Path to save the preprocessor
    feature_names : list, optional
        List of feature names to save alongside preprocessor
    """
    import joblib

    print("\n" + "=" * 60)
    print("SAVING PREPROCESSOR")
    print("=" * 60)

    # Create directory if needed
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessor
    joblib.dump(preprocessor, filepath)
    print(f"Preprocessor saved: {filepath}")

    # Save feature names separately
    if feature_names is not None:
        names_path = file_path.parent / f"{file_path.stem}_feature_names.txt"
        with open(names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"Feature names saved: {names_path}")


def load_preprocessor(filepath: str = "models/preprocessor.pkl"):
    """
    Load a saved preprocessor.

    Parameters:
    -----------
    filepath : str
        Path to the saved preprocessor

    Returns:
    --------
    ColumnTransformer : Loaded preprocessor
    """
    import joblib

    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {filepath}")

    preprocessor = joblib.load(filepath)
    print(f"Preprocessor loaded from: {filepath}")

    return preprocessor


def create_feature_dataframe_summary(X: pd.DataFrame,
                                     y: pd.Series,
                                     feature_names: List[str]) -> pd.DataFrame:
    """
    Create a summary of the final feature matrix.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    feature_names : list
        Feature names

    Returns:
    --------
    pd.DataFrame : Feature summary
    """
    print("\n" + "=" * 60)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 60)

    summary = pd.DataFrame({
        'Feature': feature_names,
        'Type': [str(X[col].dtype) if col in X.columns else 'N/A'
                 for col in feature_names],
        'Mean': [X[col].mean() if col in X.columns else np.nan
                 for col in feature_names],
        'Std': [X[col].std() if col in X.columns else np.nan
                for col in feature_names],
        'Min': [X[col].min() if col in X.columns else np.nan
                for col in feature_names],
        'Max': [X[col].max() if col in X.columns else np.nan
                for col in feature_names],
        'Missing': [X[col].isnull().sum() if col in X.columns else 0
                    for col in feature_names]
    })

    # Add correlation with target for continuous features
    if isinstance(X, pd.DataFrame):
        correlations = []
        for col in feature_names:
            if col in X.columns and X[col].dtype in ['int64', 'float64']:
                corr = X[col].corr(y)
                correlations.append(corr)
            else:
                correlations.append(np.nan)
        summary['Target_Correlation'] = correlations

    print(f"Total features: {len(feature_names)}")
    print(f"Missing values: {summary['Missing'].sum()}")

    return summary


def validate_encoded_data(X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          feature_names: List[str]) -> bool:
    """
    Validate the encoded data for common issues.

    Parameters:
    -----------
    X_train, y_train, X_test, y_test : arrays
        Data splits
    feature_names : list
        Feature names

    Returns:
    --------
    bool : True if validation passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("VALIDATING ENCODED DATA")
    print("=" * 60)

    all_checks_passed = True

    # Check 1: Feature dimensions match
    if X_train.shape[1] != X_test.shape[1]:
        print(f"Feature dimension mismatch: Train={X_train.shape[1]}, Test={X_test.shape[1]}")
        all_checks_passed = False
    else:
        print(f"Feature dimensions match: {X_train.shape[1]} features")

    # Check 2: No NaN values
    if np.isnan(X_train).any():
        print(f"NaN values found in training features")
        all_checks_passed = False
    else:
        print("No NaN values in training features")

    if np.isnan(y_train).any():
        print("NaN values found in training target")
        all_checks_passed = False
    else:
        print("No NaN values in training target")

    # Check 3: Target is binary
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)

    if len(unique_train) != 2 or set(unique_train) != {0, 1}:
        print(f"Training target not binary: {unique_train}")
        all_checks_passed = False
    else:
        print("Training target is binary (0/1)")

    if len(unique_test) != 2 or set(unique_test) != {0, 1}:
        print(f"Test target not binary: {unique_test}")
        all_checks_passed = False
    else:
        print("Test target is binary (0/1)")

    # Check 4: No infinite values
    if np.isinf(X_train).any():
        print("Infinite values found in training features")
        all_checks_passed = False
    else:
        print("No infinite values in training features")

    # Check 5: Feature names count matches
    if len(feature_names) != X_train.shape[1]:
        print(f"Feature names count ({len(feature_names)}) != feature count ({X_train.shape[1]})")
        all_checks_passed = False
    else:
        print("Feature names count matches feature matrix")

    # Check 6: Class balance
    dropout_pct_train = y_train.mean() * 100
    dropout_pct_test = y_test.mean() * 100
    print(f"Training dropout rate: {dropout_pct_train:.1f}%")
    print(f"Test dropout rate: {dropout_pct_test:.1f}%")

    if abs(dropout_pct_train - dropout_pct_test) > 5:
        print(f"Warning: Dropout rate difference > 5% between train and test")

    # Final result
    if all_checks_passed:
        print("\nAll validation checks passed!")
    else:
        print("\nSome validation checks failed. Review warnings above.")

    return all_checks_passed


# ============================================================
# COMPLETE PIPELINE FUNCTION
# ============================================================

def full_encode_pipeline(df: pd.DataFrame,
                         config: Optional[dict] = None,
                         model_type: str = 'tree',
                         test_size: float = 0.2,
                         val_size: float = 0.25,
                         random_state: int = 42,
                         save_data: bool = True,
                         output_dir: str = "data/processed",
                         exclude_second_semester: bool = False) -> Tuple:
    """
    Complete pipeline: encode features, split data, validate, and save.

    Parameters:
    -----------
    df : pd.DataFrame
        Input cleaned dataframe
    config : dict, optional
        Feature configuration
    model_type : str
        'linear' or 'tree'
    test_size : float
        Test set proportion
    val_size : float
        Validation set proportion (from training)
    random_state : int
        Random seed
    save_data : bool
        Whether to save encoded data
    output_dir : str
        Output directory for saved files
    exclude_second_semester : bool
        Whether to exclude 2nd semester features

    Returns:
    --------
    tuple : (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    print("\n" + "=" * 70)
    print("COMPLETE ENCODING PIPELINE")
    print("=" * 70)

    # Load config
    if config is None:
        config = load_config()

    # Define columns to exclude
    exclude_cols = ['Target', 'Target_encoded']
    if exclude_second_semester:
        second_sem_cols = [col for col in df.columns if '2nd sem' in col]
        exclude_cols.extend(second_sem_cols)
        print(f"Excluding 2nd semester features: {len(second_sem_cols)} columns")

    # Prepare features
    X, y, preprocessor, feature_names = prepare_features_for_modeling(
        df,
        config=config,
        model_type=model_type,
        exclude_cols=exclude_cols
    )

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # Validate encoded data
    validate_encoded_data(X_train, y_train, X_test, y_test, feature_names)

    # Save data
    if save_data:
        prefix = f"encoded_{model_type}"
        if exclude_second_semester:
            prefix += "_early"

        save_encoded_data(
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            output_dir=output_dir,
            prefix=prefix
        )

        # Save preprocessor (for linear models)
        if preprocessor is not None:
            save_preprocessor(
                preprocessor,
                filepath=f"models/preprocessor_{model_type}.pkl",
                feature_names=feature_names
            )

    # Create and print feature summary
    if isinstance(X_train, pd.DataFrame):
        summary = create_feature_dataframe_summary(
            pd.DataFrame(X_train, columns=feature_names[:X_train.shape[1]]),
            pd.Series(y_train),
            feature_names[:X_train.shape[1]]
        )

    print("\nEncoding pipeline complete!")

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


