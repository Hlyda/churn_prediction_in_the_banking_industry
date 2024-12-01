## Project Overview
This project focuses on predicting customer churn using machine learning techniques. It utilizes the **Churn_Modelling.csv** dataset, which contains customer demographic, account, and service details. The goal is to analyze the data, engineer relevant features, and build various classification models to predict whether a customer is likely to churn.

---

## Dataset
- **File**: `Churn_Modelling.csv`
- **Description**: The dataset includes customer information such as age, gender, account balance, and activity status. The target variable is `Exited`, indicating whether a customer churned.
- **Size**: 
  - Rows: As indicated in the script.
  - Columns: As indicated in the script.

---

## Steps and Methodologies
1. **Data Loading**
   - The dataset is loaded using `pandas`.

2. **Exploration**
   - Data types, structure, and statistical summaries are reviewed.
   - Missing and duplicate values are identified and handled.

3. **Preprocessing**
   - Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
   - Encoded categorical variables using `LabelEncoder` and `get_dummies`.

4. **Visualization**
   - Churn distribution plotted using `matplotlib` and `seaborn`.

5. **Feature Engineering**
   - Added features:
     - `BalanceZero`: Binary indicator for zero account balance.
     - `BalanceToSalaryRatio`: Ratio of balance to estimated salary.
     - Interaction terms between gender and geography.
   - Grouped customers into age and tenure categories.

6. **Feature Scaling**
   - Applied standard scaling to numerical features for better model performance.

7. **Model Training and Evaluation**
   - Models used:
     - **Random Forest**
     - **Logistic Regression**
     - **Support Vector Classifier (SVC)**
     - **K-Nearest Neighbors (KNN)**
     - **Gradient Boosting Classifier**
   - Evaluation metrics:
     - Confusion Matrix
     - Classification Report
     - Accuracy Score

---

## Key Features
- **Visualizations**: Feature importance is visualized for better interpretability.
- **Machine Learning Models**: Multiple classifiers are implemented to compare performance.
- **Automated Preprocessing**: Includes cleaning, encoding, and feature scaling steps.

---

## Libraries Used
- **Data Processing**
  - `pandas`
  - `numpy`
- **Visualization**
  - `matplotlib`
  - `seaborn`
- **Machine Learning**
  - `scikit-learn`

---

## Instructions
1. **Environment Setup**
   - Install required libraries:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn
     ```
   - Ensure the dataset `Churn_Modelling.csv` is in the specified path.

2. **Run the Script**
   - Execute the Python script `churn.py` to preprocess data, train models, and display results.

3. **Evaluate Results**
   - Review the accuracy, confusion matrices, and classification reports in the terminal or output logs.

---
