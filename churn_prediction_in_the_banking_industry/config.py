
# Configuration and library imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATA_PATH = './data/raw/churn_dataset.csv'
MODEL_PATH = './models/random_forest_model.pkl'
