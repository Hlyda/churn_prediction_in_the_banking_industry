import pandas as pd
from churn_prediction_in_the_banking_industry.features import preprocess_data
from churn_prediction_in_the_banking_industry.modeling.train import train_model
from churn_prediction_in_the_banking_industry.modeling.predict import evaluate_model
from sklearn.model_selection import train_test_split

# Step 1: Load the data
path = '../data/raw/churn_dataset.csv'  # Update this with the actual path
df = pd.read_csv(path)

# Step 2: Preprocess the data
print("Preprocessing data...")
X, df_processed = preprocess_data(df)
y = df_processed['Exited']  # Assuming 'Exited' is the target column

# Step 3: Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
print("Training the model...")
model = train_model(X_train, y_train)

# Step 5: Evaluate the model
print("Evaluating the model...")
accuracy, report = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
