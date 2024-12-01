import pandas as pd
from pathlib import Path



def load_raw_data(file_name="churn_dataset.csv"):
    data_path = Path("data/raw") / file_name
    return pd.read_csv(data_path)

def save_processed_data(df, file_name="processed_data.csv"):
    processed_path = Path("data/processed") / file_name
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
