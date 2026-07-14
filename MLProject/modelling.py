import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

if __name__ == "__main__":
    # Tracking URI is left to default (local mlruns/) for CI compatibility
    # mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    # mlflow.set_experiment("Latihan Credit Scoring")

    with mlflow.start_run():
        # Load preprocessed data
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "credit_scoring_preprocessing")

        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

        features = [c for c in train_df.columns if c != 'target']
        X_train, y_train = train_df[features], train_df['target']
        X_test, y_test = test_df[features], test_df['target']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        print(f"Accuracy: {acc:.4f}")

        # Log model with explicit conda_env to ensure MLflow generates
        # a Dockerfile using Python 3.12 instead of its hardcoded default (3.9)
        # which is no longer supported by pip's get-pip.py script.
        conda_env = {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.12.7",
                "pip",
                {
                    "pip": [
                        "pandas",
                        "scikit-learn",
                        "mlflow==2.19.0",
                    ]
                },
            ],
        }
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            conda_env=conda_env,
        )
