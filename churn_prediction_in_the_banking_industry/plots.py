from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from loguru import logger
from tqdm import tqdm

from churn_prediction_in_the_banking_industry.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def visualize_distribution (df): 
    # Visualize the class distribution (target variable)
    sns.countplot(x='Exited', data=df)
    plt.title("Churn Distribution")
    plt.show()
    
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
