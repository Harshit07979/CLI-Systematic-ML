# Data Analysis Pipeline CLI

This is a command-line interface (CLI) tool for data analysis, machine learning model training, and report generation. It supports interactive mode and provides various options for data preprocessing, model selection, and report formatting.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Harshit07979/CLI-Systematic-ML
    cd project
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Run in Interactive Mode

To run the CLI in interactive mode, use:

```sh
python -m my_cli_tool.cli --interactive
```

## CLI INTERFACE
- Then it will ask for your different option and datset.
-  Please prvide dataset withouth comma or r prefix (like given below)
  ```
D:\Systematic Altruism\data\raw\olympics2024.csv
```
- Then choose your model and their parameters and get report in pdf or html or txt file.

## Further improvements.
- Making it more robust for data preprocessing and getting edge cases
- Edit the report format in pd.dataframe
- Deploying for link
- Adding more models to choose from
- Instead of writing what model,parameters and other arguements in future number will be provided for different parameters and only number you have to choose
