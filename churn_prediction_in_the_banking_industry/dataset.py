import pandas as pd
import os

def load_data(path='../data/raw/churn_dataset.csv'):
    """
    Loads a dataset from the specified path.
    
    Parameters:
    - path (str): The relative or absolute path to the CSV file.
    
    Returns:
    - pd.DataFrame: A pandas DataFrame containing the dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")
    print(f"Loading data from {path}")
    return pd.read_csv(path)
