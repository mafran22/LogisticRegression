import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

mlruns_path = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

def load_data():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    return iris.frame, iris.target_names

def train_model():
    df, target_names = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, "logreg_model")

        # Save locally
        joblib.dump(model, "iris_model.pkl")
        mlflow.log_artifact("iris_model.pkl")

        print(f"Accuracy: {acc:.2f}")
        return acc

if __name__ == "__main__":
    train_model()
