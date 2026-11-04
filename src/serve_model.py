import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import os

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

# 2. Initialize the FastAPI app
app = FastAPI(title="PhotonMatters Delinquency Scoring API")
model = None

# 3. Load the model on startup
@app.on_event("startup")
def load_model():
    global model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"Fetching the latest model from MLflow at {tracking_uri}...")
    # Get the latest run from our experiment
    experiment = mlflow.get_experiment_by_name("Telecom_Delinquency_Prediction")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    
    if runs.empty:
        raise RuntimeError("No MLflow runs found. Train the model first.")
        
    latest_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{latest_run_id}/random_forest_model"
    
    # Load the model using MLflow's pyfunc wrapper (handles skops securely behind the scenes)
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded successfully from Run ID: {latest_run_id}")

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_delinquency(features: TelecomFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert the validated JSON payload into a pandas DataFrame (1 row)
        input_data = pd.DataFrame([features.dict()])
        
        # Generate prediction
        prediction = model.predict(input_data)[0]
        
        # Map the binary output to business logic
        risk_status = "Delinquent" if prediction == 1 else "Non-Delinquent"
        
        return {
            "prediction_code": int(prediction),
            "risk_status": risk_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))