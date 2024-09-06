import pandas as pd
import os

def load_data(file_path, handle_missing='fill', fill_value=None, enforce_dtypes=None):
    """
    Load data from a file, handling different structures, missing values, and other common issues.
    """
    # Strip surrounding quotes from the file path if necessary
    file_path = file_path.strip('\"\'')
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None

    # Determine the file extension and load accordingly
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            print(f"Error: Unsupported file format: {file_extension}")
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Handle missing values
    if handle_missing == 'drop':
        df = df.dropna()
    elif handle_missing == 'fill' and fill_value is not None:
        df = df.fillna(fill_value)
    
    # Enforce specific data types if provided
    if enforce_dtypes:
        for column, dtype in enforce_dtypes.items():
            try:
                df[column] = df[column].astype(dtype)
            except Exception as e:
                print(f"Error converting column {column} to {dtype}: {e}")

    return df

def clean_and_preprocess_data(df, handle_missing='fill', outlier_method='zscore', 
                              outlier_threshold=None, scaling_method='standard', 
                              categorical_encoding='onehot'):
    """
    Clean and preprocess the data by handling missing values, outliers, and performing feature engineering.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from scipy import stats
    import numpy as np

    # Handle missing values
    if handle_missing == 'drop':
        df = df.dropna()
    elif handle_missing == 'fill':
        df = df.fillna(0)

    # Select only numeric columns for outlier detection and scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Automatically set default thresholds if not provided
    if outlier_method == 'zscore' and outlier_threshold is None:
        outlier_threshold = 3  # Default Z-score threshold is 3
    elif outlier_method == 'iqr' and outlier_threshold is None:
        outlier_threshold = 1.5  # Default IQR multiplier is 1.5

    # Outlier detection and treatment
    if outlier_method == 'zscore':
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z_scores < outlier_threshold).all(axis=1)]
    elif outlier_method == 'iqr':
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[numeric_cols] < (Q1 - outlier_threshold * IQR)) | 
                  (df[numeric_cols] > (Q3 + outlier_threshold * IQR))).any(axis=1)]
    
    # Feature engineering (encoding categorical variables)
    if categorical_encoding == 'onehot':
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif categorical_encoding == 'label':
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
    
    # Scaling numerical features
    if scaling_method == 'standard':
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
