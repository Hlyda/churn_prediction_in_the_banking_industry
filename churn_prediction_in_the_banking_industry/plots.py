
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def plot_feature_distribution(df, column):
    """Plot the distribution of a specific feature."""
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """Plot a confusion matrix using Seaborn."""
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

def class_distribution (df) :
    sns.countplot(x='Exited', data=df)
    plt.title("Churn Distribution")
    plt.show()
def class_report (y_test, y_pred):
    class_report = classification_report(y_test, y_pred)
    print('class report', class_report)

def acc_score (y_test, y_pred):
    accuracy_plot = accuracy_score(y_test, y_pred)
    print('accuracy score', accuracy_plot)

def gender_distribution(df) :
    g= sns.catplot(x = "Gender", y = "Exited", data = df, kind = "bar", height = 5)
    g.set_ylabels("Churn Probability")
    plt.show()