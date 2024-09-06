import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Set up logging
logging.basicConfig(filename='logs/model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def encode_categorical_columns(df: pd.DataFrame, encoding_type: str) -> pd.DataFrame:
    """Automatically encode categorical columns based on the specified encoding type.

    Args:
        df (pd.DataFrame): The dataframe to encode.
        encoding_type (str): The type of encoding to apply ('onehot' or 'label').

    Returns:
        pd.DataFrame: The dataframe with encoded categorical columns.
    """
    try:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        if encoding_type == 'label':
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])

        elif encoding_type == 'onehot':
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        return df

    except Exception as e:
        logging.error(f"Error encoding categorical columns: {e}")
        raise


def train_model(df_cleaned: pd.DataFrame, model_type: str, model_params: Dict[str, Any], 
                encoding_type: str = 'label') -> Optional[Dict[str, Any]]:
    """Train a machine learning model with the specified parameters and data preprocessing steps.

    Args:
        df_cleaned (pd.DataFrame): The preprocessed dataframe.
        model_type (str): The type of model to train.
        model_params (Dict[str, Any]): Model parameters.
        encoding_type (str, optional): The type of encoding to apply to categorical variables ('onehot' or 'label'). Defaults to 'label'.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with evaluation metrics, or None if training fails.
    """
    try:
        logging.info(f"Training {model_type} model...")

        # Define the target column
        target_column = 'High_Performance' if 'High_Performance' in df_cleaned.columns else df_cleaned.columns[-1]

        # Encode categorical variables
        df_cleaned = encode_categorical_columns(df_cleaned, encoding_type)

        # Prepare features (X) and target (y)
        if 'Total' in df_cleaned.columns:
            X = df_cleaned.drop(columns=['Total', target_column])
        else:
            X = df_cleaned.drop(columns=[target_column])

        y = df_cleaned[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale the features for KNN and logistic regression models
        if model_type in ['knn', 'logistic_regression']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Initialize the model based on the specified type
        model, default_metrics = initialize_model(model_type, model_params)
        if model is None:
            logging.error(f"Unsupported model type '{model_type}'")
            return None

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model performance
        evaluation_results = evaluate_model(model_type, y_test, y_pred, default_metrics)

        logging.info(f"Model Evaluation Results: {evaluation_results}")
        return evaluation_results

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None


def initialize_model(model_type: str, model_params: Dict[str, Any]) -> Optional[tuple]:
    """Initialize the model and set default metrics based on the model type.

    Args:
        model_type (str): The type of model to initialize.
        model_params (Dict[str, Any]): Model parameters.

    Returns:
        tuple: The initialized model and a list of default metrics, or None if the model type is unsupported.
    """
    try:
        if model_type == 'random_forest_classifier':
            model = RandomForestClassifier(**model_params)
            default_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**model_params)
            default_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(**model_params)
            default_metrics = ['accuracy', 'precision', 'recall', 'f1']
        elif model_type == 'knn':
            model = KNeighborsClassifier(**model_params)
            default_metrics = ['accuracy', 'precision', 'recall', 'f1']
        elif model_type == 'linear_regression':
            model = LinearRegression(**model_params)
            default_metrics = ['mae', 'mse', 'rmse', 'r2']
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor(**model_params)
            default_metrics = ['mae', 'mse', 'rmse', 'r2']
        elif model_type == 'gradient_boosting_regressor':
            model = GradientBoostingRegressor(**model_params)
            default_metrics = ['mae', 'mse', 'rmse', 'r2']
        elif model_type == 'svr':
            model = SVR(**model_params)
            default_metrics = ['mae', 'mse', 'rmse', 'r2']
        else:
            return None, None

        return model, default_metrics

    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        return None, None


def evaluate_model(model_type: str, y_test: np.ndarray, y_pred: np.ndarray, default_metrics: list) -> Dict[str, Any]:
    """Evaluate the model's performance based on the specified metrics.

    Args:
        model_type (str): The type of model used.
        y_test (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
        default_metrics (list): List of metrics to calculate.

    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics.
    """
    try:
        evaluation_results = {}

        # Classification metrics
        if model_type in ['random_forest_classifier', 'logistic_regression', 'decision_tree', 'knn']:
            average_type = 'binary' if len(set(y_test)) == 2 else 'macro'

            if 'accuracy' in default_metrics:
                evaluation_results['accuracy'] = accuracy_score(y_test, y_pred)
            if 'precision' in default_metrics:
                evaluation_results['precision'] = precision_score(y_test, y_pred, average=average_type)
            if 'recall' in default_metrics:
                evaluation_results['recall'] = recall_score(y_test, y_pred, average=average_type)
            if 'f1' in default_metrics:
                evaluation_results['f1'] = f1_score(y_test, y_pred, average=average_type)
            if 'roc_auc' in default_metrics and average_type == 'binary':
                evaluation_results['roc_auc'] = roc_auc_score(y_test, y_pred)

        # Regression metrics
        else:
            if 'mae' in default_metrics:
                evaluation_results['mae'] = mean_absolute_error(y_test, y_pred)
            if 'mse' in default_metrics:
                evaluation_results['mse'] = mean_squared_error(y_test, y_pred)
            if 'rmse' in default_metrics:
                evaluation_results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            if 'r2' in default_metrics:
                evaluation_results['r2'] = r2_score(y_test, y_pred)

        return evaluation_results

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return {}


