import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Latihan Credit Scoring")
    
    # Use autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="basic_model"):
        # Load preprocessed data
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "Eksperimen_SML_MasMohamadRafiNouvalFadil", "preprocessing", "credit_scoring_preprocessing")
        
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        
        features = [c for c in train_df.columns if c != 'target']
        X_train, y_train = train_df[features], train_df['target']
        X_test, y_test = test_df[features], test_df['target']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
