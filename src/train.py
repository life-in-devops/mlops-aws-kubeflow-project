import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import load_data, clean_data


DATA_PATH = "data/churn.csv"


def train():

    # 🔷 MLflow setup
    mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local:5000")
    mlflow.set_experiment("churn-prediction")

    # 🔷 Load & clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # 🔷 Target variable (convert Yes/No → 1/0)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # 🔷 Features
    X = df.drop("Churn", axis=1)

    # 🔷 Split BEFORE encoding (critical to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔷 Encode separately
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # 🔷 Align columns (VERY IMPORTANT for inference later)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # 🔷 Start MLflow run
    with mlflow.start_run():

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Accuracy: {acc}")

        # 🔷 Log parameters & metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        # 🔷 Save model + feature columns (VERY IMPORTANT for Day 3)
        joblib.dump(model, "model.joblib")
        joblib.dump(X_train.columns.tolist(), "features.joblib")

        # 🔷 Log artifacts
        mlflow.log_artifact("model.joblib")
        mlflow.log_artifact("features.joblib")

    print("Training complete and logged to MLflow.")


if __name__ == "__main__":
    train()