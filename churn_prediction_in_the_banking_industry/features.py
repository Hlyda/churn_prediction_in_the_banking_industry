from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']])
    return scaled_features, df
