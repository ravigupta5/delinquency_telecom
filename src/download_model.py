import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

def download_production_model():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    print(os.getenv("MLFLOW_TRACKING_URI"))
    model_name = " GBM_V2_Filtered"
    stage = "Production"
    
    print(f"Connecting to MLflow to fetch {model_name} at stage: {stage}...")
    
    client = MlflowClient()
    
    # MLflow URI format for registered models: models:/<model_name>/<stage>
    model_uri = f"models:/{model_name}/{stage}"
    download_path = "production_model"
    
    os.makedirs(download_path, exist_ok=True)
    
    try:
        # Download the artifact directory locally
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_path)
        print(f"Production model successfully downloaded to: {local_path}")
    except Exception as e:
        print(f"Failed to download model: {e}")
        exit(1)

if __name__ == "__main__":
    download_production_model()