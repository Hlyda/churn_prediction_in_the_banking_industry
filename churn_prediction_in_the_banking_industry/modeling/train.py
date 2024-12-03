
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from ..config import MODEL_PATH
from churn_prediction_in_the_banking_industry.features import create_features
from churn_prediction_in_the_banking_industry.plots import acc_score,class_report, gender_distribution, class_distribution, plot_feature_distribution, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


def prepare_test_data (df) :
    features = create_features(df=df)
    X_train, X_test, y_train, y_test = train_test_split(features[0], features[1], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_test, y_train

def train_model_random_forest(df, target_column='Exited'):
    """Train a Random Forest model and save it."""
    print ('-----Random Forest------')
    X_train, X_test, y_test, y_train= prepare_test_data(df=df)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    class_report(y_test, y_pred)

def train_model_logistic_regression(df):
    # Build and train the Logistic Regression model
    print ('-----Logistic Regression------')
    X_train, X_test, y_test, y_train= prepare_test_data(df=df)
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    # Make predictions
    y_pred_log_reg = log_reg.predict(X_test)
    # Evalute the model
    plot_confusion_matrix(y_test, y_pred_log_reg)
    class_report(y_test, y_pred_log_reg)
    acc_score(y_test, y_pred_log_reg)

def train_model_svm(df):
    # Build and train the Logistic Regression model
    print ('-----Support Vector Classifier (SVC)------')
    # Build and train the SVC model
    X_train, X_test, y_test, y_train= prepare_test_data(df=df)
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    # Make predictions
    y_pred_svm = svm_model.predict(X_test)
    # Evaluate the model
    plot_confusion_matrix(y_test, y_pred_svm)
    class_report(y_test, y_pred_svm)
    acc_score(y_test, y_pred_svm)

def train_model_knn(df):
    print ('-----KNN------')
    X_train, X_test, y_test, y_train= prepare_test_data(df=df)
   # Build and train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Make predictions
    y_pred_knn = knn_model.predict(X_test)

    plot_confusion_matrix(y_test, y_pred_knn)
    class_report(y_test, y_pred_knn)
    acc_score(y_test, y_pred_knn)
    
def train_model_gbc(df):
    print ('-----Gradient Boosting Classifier------')
    X_train, X_test, y_test, y_train= prepare_test_data(df=df)
   # Build and train the KNN model
    # Build and train the Gradient Boosting Classifier model
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_gb = gb_model.predict(X_test)

    plot_confusion_matrix(y_test, y_pred_gb)
    class_report(y_test, y_pred_gb)
    acc_score(y_test, y_pred_gb)    
