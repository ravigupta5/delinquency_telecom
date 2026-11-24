import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import os
import json
import logging
from datetime import datetime

# 1. Define the Data Contract (Pydantic Schema)
# This guarantees that the API only accepts the exact 32 features the model expects.
class TelecomFeatures(BaseModel):
    aon: float
    daily_decr30: float
    daily_decr90: float
    rental30: float
    rental90: float
    last_rech_date_ma: float
    last_rech_date_da: float
    last_rech_amt_ma: float
    cnt_ma_rech30: float
    fr_ma_rech30: float
    sumamnt_ma_rech30: float
    medianamnt_ma_rech30: float
    medianmarechprebal30: float
    cnt_ma_rech90: float
    fr_ma_rech90: float
    sumamnt_ma_rech90: float
    medianamnt_ma_rech90: float
    medianmarechprebal90: float
    cnt_da_rech30: float
    fr_da_rech30: float
    cnt_da_rech90: float
    fr_da_rech90: float
    cnt_loans30: float
    amnt_loans30: float
    maxamnt_loans30: float
    medianamnt_loans30: float
    cnt_loans90: float
    amnt_loans90: float
    maxamnt_loans90: float
    medianamnt_loans90: float
    payback30: float
    payback90: float

# Initialize Persistent Logging
log_dir = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("model_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(log_dir, "predictions.jsonl"))
logger.addHandler(file_handler)

# 2. Initialize the FastAPI app
app = FastAPI(title="Telecom Delinquency Scoring API")
model = None
model_version = "unknown"

# 3. Load the model on startup
@app.on_event("startup")
def load_model():
    global model
    global model_version
    
    # The CI pipeline injects the downloaded model exactly here before Docker builds
    # Use an environment variable, defaulting to the Docker path if not provided
    model_path = os.getenv("MODEL_PATH", "/app/model")
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Production model not found at {model_path}. CI pipeline failed to inject it.")
        
    print(f"Loading production model directly from file system: {model_path}")
    model = mlflow.pyfunc.load_model(model_path)
    
    # Extract the MLflow Run ID to securely version the predictions
    try:
        model_version = model.metadata.run_id
    except Exception:
        model_version = "latest_unversioned"
        
    print(f"Production Model loaded successfully! Version: {model_version}")

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_delinquency(features: TelecomFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert the validated JSON payload into a dictionary and then a DataFrame
        input_dict = features.dict()
        input_data = pd.DataFrame([input_dict])
        
        # Generate prediction
        prediction = model.predict(input_data)[0]
        
        # Map the binary output to business logic
        risk_status = "Delinquent" if prediction == 1 else "Non-Delinquent"
        
        # --- LOGGING EXECUTION ---
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "input_features": input_dict,
            "prediction_code": int(prediction),
            "risk_status": risk_status
        }
        logger.info(json.dumps(log_entry))
        # -------------------------
        
        return {
            "prediction_code": int(prediction),
            "risk_status": risk_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))