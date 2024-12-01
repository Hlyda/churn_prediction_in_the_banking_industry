from churn_prediction_in_the_banking_industry.dataset import load_raw_data, save_processed_data
from churn_prediction_in_the_banking_industry.features import preprocess_data
from churn_prediction_in_the_banking_industry.modeling.train import train_model
from churn_prediction_in_the_banking_industry.features import check_duplicated, check_missing_value, drop_unnecessary_columns, encode_variables
def main():
    # Charger les données
    print("Loading raw data...")
    df = load_raw_data()
    print(df.head())

    print('Data Exploration...')
    print(df.info())
    # Display basic statistics of numerical features
    print('Display basic statistics of numerical features...')
    print(df.describe())
    # Display the shape of the dataset (nb rows and columns)
    print('Display the shape of the dataset (nb rows and columns)...')
    print(df.shape)
    # Prétraiter les données
    print("Preprocessing data...")

    print('check missing values...')
    print(check_missing_value(df))

    print('check duplicate values')
    print(check_duplicated(df))

    df = drop_unnecessary_columns(df)
    df = encode_variables(df)
    print('preprocessed data')
    df.head()
    processed_data = preprocess_data(df)
    
    # Sauvegarder les données traitées
    print("Saving processed data...")
    save_processed_data(processed_data)
    
    # Entraîner le modèle
    print("Training model...")
    model = train_model(processed_data)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
