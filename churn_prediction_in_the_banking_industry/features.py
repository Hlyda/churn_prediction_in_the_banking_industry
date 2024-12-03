
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from re import X
from churn_prediction_in_the_banking_industry.plots import class_report, gender_distribution, class_distribution, plot_feature_distribution, plot_confusion_matrix

def preprocess_data(df):
    """Fill missing values and encode categorical variables."""
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df= pd.get_dummies(df, columns=['Geography'], drop_first=True)
    return df

def create_features(df):
    features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Spain', 'Geography_Germany']

    X = df[features]
    y = df['Exited']
    return [X,y]
