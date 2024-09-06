import argparse
import os
from colorama import Fore, Style, init
from my_cli_tool.data_processing import load_data, clean_and_preprocess_data
from my_cli_tool.model_training import train_model
from my_cli_tool.report_generation import generate_report
import logging

# Initialize colorama for cross-platform colored output
init(autoreset=True)

def log_and_print(message, level='info'):
    if level == 'info':
        print(Fore.GREEN + message)
        logging.info(message)
    elif level == 'warning':
        print(Fore.YELLOW + message)
        logging.warning(message)
    elif level == 'error':
        print(Fore.RED + message)
        logging.error(message)
    elif level == 'success':
        print(Fore.BLUE + message)
        logging.info(message)

def ensure_directories_exist():
    """Ensure required directories exist."""
    directories = ['data/processed', 'models', 'reports', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            log_and_print(f"Created directory: {directory}", 'info')

ensure_directories_exist()

# Set up logging
logging.basicConfig(filename='logs/cli.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def input_with_validation(prompt, valid_options=None, default=None):
    while True:
        user_input = input(prompt).strip().lower()
        if valid_options and user_input not in valid_options:
            print(Fore.YELLOW + f"Invalid input. Expected one of: {', '.join(valid_options)}")
            continue
        return user_input or default

def interactive_mode():
    ensure_directories_exist()
    log_and_print("Welcome to the Interactive Mode of the Data Analysis Pipeline!", 'info')
    file_path = input("Enter the path to the data file: ").strip('\"\'')
    df = load_data(file_path)
    if df is None:
        return

    run_analysis = input_with_validation("Would you like to run data analysis? (yes/no): ", ['yes', 'no'], 'no')
    if run_analysis == 'yes':
        handle_missing = input_with_validation("Choose a missing value handling method (fill, drop, ffill, bfill): ", ['fill', 'drop', 'ffill', 'bfill'])
        outlier_method = input_with_validation("Choose an outlier detection method (zscore, iqr): ", ['zscore', 'iqr'])
        scaling_method = input_with_validation("Choose a feature scaling method (standard, minmax): ", ['standard', 'minmax'])
        categorical_encoding = input_with_validation("Choose a categorical encoding method (onehot, label): ", ['onehot', 'label'])

        df_cleaned = clean_and_preprocess_data(df, handle_missing, outlier_method, scaling_method=scaling_method, categorical_encoding=categorical_encoding)
        log_and_print("Data cleaning and preprocessing completed!", 'info')
        if df_cleaned is None:
            return

    train_model_flag = input_with_validation("Would you like to train a model? (yes/no): ", ['yes', 'no'], 'no')
    if train_model_flag == 'yes':
        model_type = input_with_validation("Choose a model type (random_forest, logistic_regression, decision_tree, knn, linear_regression): ", ['random_forest', 'logistic_regression', 'decision_tree', 'knn', 'linear_regression'])
        model_params = input("Enter model parameters in key=value format (e.g., n_estimators=100,max_depth=5): ")
        params_dict = {}
        for param in model_params.split(','):
            key, value = param.split('=')
            params_dict[key] = eval(value)

        # Train the model and pass the chosen encoding type
        evaluation_results = train_model(df_cleaned, model_type, params_dict, encoding_type=categorical_encoding)

    generate_report_flag = input_with_validation("Would you like to generate a report? (yes/no): ", ['yes', 'no'], 'no')
    if generate_report_flag == 'yes':
        report_format = input_with_validation("Choose a report format (text, html, pdf): ", ['text', 'html', 'pdf'])
        generate_report(evaluation_results, report_format)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Data Analysis Pipeline CLI')
    parser.add_argument('-f', '--file_path', help='Path to the data file', type=str)
    parser.add_argument('-a', '--run_analysis', help='Run the data analysis', action='store_true')
    parser.add_argument('-t', '--train_model', help='Train the selected ML model', action='store_true')
    parser.add_argument('-g', '--generate_report', help='Generate a report', action='store_true')
    parser.add_argument('-i', '--interactive', help='Run the CLI in interactive mode', action='store_true')

    parser.add_argument('--handle_missing', help='Missing value handling method (fill, drop, ffill, bfill)', type=str, default='fill')
    parser.add_argument('--outlier_method', help='Outlier detection method (zscore, iqr)', type=str, default='zscore')
    parser.add_argument('--outlier_threshold', help='Outlier threshold value', type=float, default=3.0)
    parser.add_argument('--scaling_method', help='Feature scaling method (standard, minmax)', type=str, default='standard')
    parser.add_argument('--categorical_encoding', help='Categorical encoding method (onehot, label)', type=str, default='onehot')

    parser.add_argument('--model_type', help='Type of ML model to train (random_forest, logistic_regression, decision_tree, knn, linear_regression)', type=str, default='random_forest')
    parser.add_argument('--model_params', help='Model parameters in key=value format, separated by commas (e.g., n_estimators=100,max_depth=5)', type=str, default='')
    parser.add_argument('--report_format', help='Report format (text, html, pdf)', type=str, default='text')

    args = parser.parse_args()

    ensure_directories_exist()

    if args.interactive:
        interactive_mode()
        return

    # Load data
    if args.file_path:
        df = load_data(args.file_path)
        if df is None:
            return
    else:
        log_and_print("No file path provided. Exiting.", 'error')
        return

    # Run analysis
    if args.run_analysis:
        df_cleaned = clean_and_preprocess_data(
            df,
            handle_missing=args.handle_missing,
            outlier_method=args.outlier_method,
            outlier_threshold=args.outlier_threshold,
            scaling_method=args.scaling_method,
            categorical_encoding=args.categorical_encoding
        )
        if df_cleaned is None:
            return

    # Train the model
    if args.train_model:
        model_params = {}
        if args.model_params:
            for param in args.model_params.split(','):
                key, value = param.split('=')
                model_params[key] = eval(value)

        # Train model with auto-selected metrics and the correct encoding type
        evaluation_results = train_model(df_cleaned, args.model_type, model_params, encoding_type=args.categorical_encoding)

    # Generate report
    if args.generate_report:
        generate_report(evaluation_results, args.report_format)

if __name__ == '__main__':
    main()
