import os
import pandas as pd
from typing import Optional, Dict
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import numpy as np

# Set up logging
logging.basicConfig(filename='logs/data_preprocessing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str, handle_missing: str = 'fill', fill_value: Optional[Any] = None, enforce_dtypes: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
    """Load data from a file, handling missing values, file types, and data type enforcement.

    Args:
        file_path (str): Path to the file containing data.
        handle_missing (str, optional): Method for handling missing values ('fill', 'drop'). Defaults to 'fill'.
        fill_value (Any, optional): Value used to fill missing data when 'fill' is selected. Defaults to None.
        enforce_dtypes (dict, optional): A dictionary mapping columns to data types to enforce. Defaults to None.

    Returns:
        pd.DataFrame: The loaded and processed dataframe, or None if loading fails.
    """
    file_path = file_path.strip('\"\'')  # Strip surrounding quotes from file path

    if not os.path.exists(file_path):
        logging.error(f"File '{file_path}' does not exist.")
        return None

    # Determine file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            logging.error(f"Unsupported file format: {file_extension}")
            return None
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

    # Handle missing values
    if handle_missing == 'drop':
        df = df.dropna()
    elif handle_missing == 'fill' and fill_value is not None:
        df = df.fillna(fill_value)
    
    # Enforce specific data types
    if enforce_dtypes:
        for column, dtype in enforce_dtypes.items():
            try:
                df[column] = df[column].astype(dtype)
            except Exception as e:
                logging.error(f"Error converting column {column} to {dtype}: {e}")

    return df


def clean_and_preprocess_data(df: pd.DataFrame, handle_missing: str = 'fill', outlier_method: str = 'zscore', 
                              outlier_threshold: Optional[float] = None, scaling_method: str = 'standard', 
                              categorical_encoding: str = 'onehot') -> pd.DataFrame:
    """Clean and preprocess the data by handling missing values, detecting outliers, scaling, and encoding.

    Args:
        df (pd.DataFrame): Input dataframe to be cleaned and preprocessed.
        handle_missing (str, optional): Method to handle missing values ('fill', 'drop'). Defaults to 'fill'.
        outlier_method (str, optional): Method for outlier detection ('zscore', 'iqr'). Defaults to 'zscore'.
        outlier_threshold (float, optional): Threshold for outlier detection. Defaults to None.
        scaling_method (str, optional): Scaling method for numerical data ('standard', 'minmax'). Defaults to 'standard'.
        categorical_encoding (str, optional): Encoding method for categorical variables ('onehot', 'label'). Defaults to 'onehot'.

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataframe.
    """
    # Handle missing values
    if handle_missing == 'drop':
        df = df.dropna()
    elif handle_missing == 'fill':
        df = df.fillna(0)

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Set default thresholds if not provided
    if outlier_method == 'zscore' and outlier_threshold is None:
        outlier_threshold = 3  # Default Z-score threshold is 3
    elif outlier_method == 'iqr' and outlier_threshold is None:
        outlier_threshold = 1.5  # Default IQR multiplier is 1.5

    # Outlier detection
    try:
        if outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(df[numeric_cols]))
            df = df[(z_scores < outlier_threshold).all(axis=1)]
        elif outlier_method == 'iqr':
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[numeric_cols] < (Q1 - outlier_threshold * IQR)) | 
                      (df[numeric_cols] > (Q3 + outlier_threshold * IQR))).any(axis=1)]
    except Exception as e:
        logging.error(f"Error during outlier detection: {e}")

    # Categorical encoding
    try:
        if categorical_encoding == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif categorical_encoding == 'label':
            for col in categorical_cols:
                df[col] = df[col].astype('category').cat.codes
    except Exception as e:
        logging.error(f"Error during categorical encoding: {e}")

    # Scaling numerical features
    try:
        if scaling_method == 'standard':
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    except Exception as e:
        logging.error(f"Error during scaling: {e}")

    return df
