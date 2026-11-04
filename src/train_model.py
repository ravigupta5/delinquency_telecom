import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Set the Tracking URI to your local server
mlflow.set_tracking_uri("http://localhost:5000")
# 2. Name the experiment
mlflow.set_experiment("Telecom_Delinquency_Prediction")

def load_data(file_path):
    """Load the processed dataset."""
    return pd.read_csv(file_path)

def evaluate_model(y_true, y_pred, y_prob):
    """Calculate and return rigorous classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    return metrics

def train():
    PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"
    
    # Load and split data
    df = load_data(PROCESSED_DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define hyperparameters explicitly so we can log them
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced" # Crucial for delinquency/default datasets
    }
    
    print("Starting MLflow run...")
    with mlflow.start_run(run_name="Baseline_RandomForest"):
        # Log hyperparameters
        mlflow.log_params(rf_params)
        
        # Train model
        print("Training model...")
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_prob)
        print(f"Metrics: {metrics}")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log the model artifact securely
        # Log the model artifact securely
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model", 
            registered_model_name="Baseline_RF_Model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print("Run complete. Artifacts logged to MLflow.")

if __name__ == "__main__":
    train()