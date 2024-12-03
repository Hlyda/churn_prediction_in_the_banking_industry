
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from ..config import MODEL_PATH

def evaluate_model(X_test, y_test):
    """Load the model and evaluate its performance."""
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return y_pred
