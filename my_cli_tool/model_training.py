from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

def encode_categorical_columns(df, encoding_type):
    """
    Automatically encode categorical columns based on the specified encoding type (onehot or label).
    """
    # Identify categorical columns (dtype 'object' or 'category')
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    if encoding_type == 'label':
        # Apply label encoding
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

    elif encoding_type == 'onehot':
        # Apply one-hot encoding
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    return df

def train_model(df_cleaned, model_type, model_params, encoding_type='label'):
    print(f"Training {model_type} model...")

    # Check if 'High_Performance' exists, if not, use the last column as the target
    target_column = 'High_Performance' if 'High_Performance' in df_cleaned.columns else df_cleaned.columns[-1]

    # Automatically encode categorical variables based on user input
    df_cleaned = encode_categorical_columns(df_cleaned, encoding_type)

    # Check if 'Total' exists before dropping it
    if 'Total' in df_cleaned.columns:
        X = df_cleaned.drop(columns=['Total', target_column])
    else:
        X = df_cleaned.drop(columns=[target_column])

    y = df_cleaned[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features for KNN (and for logistic regression if needed)
    if model_type in ['knn', 'logistic_regression']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Initialize the model based on the specified type
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
        print(f"Error: Unsupported model type '{model_type}'")
        return None

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Initialize the evaluation results dictionary
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

    # Print and return evaluation results
    print(f"Model Evaluation Results: {evaluation_results}")
    return evaluation_results
