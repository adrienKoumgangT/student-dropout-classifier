import pandas as pd

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

