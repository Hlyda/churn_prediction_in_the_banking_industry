import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder # Encoding categorical variables


def check_missing_value(df): 
    df.isnull().sum() # check missing values

def check_duplicated(df): 
    df[df.duplicated()]

def drop_unnecessary_columns(df): 
    # Drop unnecessary columns
    # 'RowNumber', 'CustomerId', and 'Surname' are irrelevant for prediction
    df =df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    return df

def encode_variables(df): 
        # Instantiate the LabelEncoder
        label_encoder = LabelEncoder()
        df['Gender'] = label_encoder.fit_transform(df['Gender'])
        df= pd.get_dummies(df, columns=['Geography'], drop_first=True)
        return df

def preprocess_data(df):
    # Exemple de traitement
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])  # Suppression des colonnes inutiles
    df = pd.get_dummies(df, drop_first=True)  # Encodage des variables catégoriques
    return df
