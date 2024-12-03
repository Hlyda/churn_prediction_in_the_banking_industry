
import pandas as pd
from .config import DATA_PATH

def load_data(path=DATA_PATH):
    """Load the dataset from the given path."""
    return pd.read_csv(path)
