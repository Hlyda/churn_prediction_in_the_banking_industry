
from churn_prediction_in_the_banking_industry.dataset import load_data
from churn_prediction_in_the_banking_industry.features import preprocess_data, create_features
from churn_prediction_in_the_banking_industry.modeling.train import train_model_gbc, train_model_random_forest,train_model_knn, train_model_svm, train_model_logistic_regression
from churn_prediction_in_the_banking_industry.modeling.predict import evaluate_model
from churn_prediction_in_the_banking_industry.plots import class_report, gender_distribution, class_distribution, plot_feature_distribution, plot_confusion_matrix

def main():
    print("Loading dataset...")
    df = load_data()
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(df.head())

    print("Preprocessing dataset...")
    df = preprocess_data(df)
    print(df.head())
    
    train_model_random_forest(df=df)
    train_model_logistic_regression(df=df)
    train_model_svm(df=df)
    train_model_knn(df=df)
    train_model_gbc(df=df)
    # print("\nClassification Report:")
    # print(class_report)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("\nAccuracy Score (Random Forest):", accuracy)
    #plots
    # class_distribution(df=df)
    # gender_distribution(df=df)
    # print("Creating new features...")
    # df = create_features(df)

    # print("Training the model...")
    # model, X_test, y_test = train_model(df)

    # print("Evaluating the model...")
    # y_pred = evaluate_model(X_test, y_test)

    # print("Plotting confusion matrix...")
    # plot_confusion_matrix(y_test, y_pred, labels=["Not Churned", "Churned"])

    # print("Execution complete.")

if __name__ == "__main__":
    main()
