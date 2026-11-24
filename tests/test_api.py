import os
import pytest
from fastapi.testclient import TestClient

# 1. Point the API to the local model folder downloaded by your CI pipeline
# (Adjust "./model" if your download_model.py script saves it to a different folder name locally)
os.environ["MODEL_PATH"] = "./production_model"

# 2. Import the FastAPI app AFTER setting the environment variable
from src.serve_model import app

client = TestClient(app)

def test_predict_endpoint_success():
    """
    Test that the API successfully accepts a valid 32-feature payload,
    processes it through the model, and returns the expected JSON structure.
    """
    # A valid mock customer profile matching your Pydantic schema
    valid_payload = {
        "aon": 120.0,
        "daily_decr30": 10.5,
        "daily_decr90": 15.0,
        "rental30": 50.0,
        "rental90": 100.0,
        "last_rech_date_ma": 2.0,
        "last_rech_date_da": 0.0,
        "last_rech_amt_ma": 153.0,
        "cnt_ma_rech30": 2.0,
        "fr_ma_rech30": 1.0,
        "sumamnt_ma_rech30": 300.0,
        "medianamnt_ma_rech30": 150.0,
        "medianmarechprebal30": 20.0,
        "cnt_ma_rech90": 3.0,
        "fr_ma_rech90": 2.0,
        "sumamnt_ma_rech90": 450.0,
        "medianamnt_ma_rech90": 150.0,
        "medianmarechprebal90": 25.0,
        "cnt_da_rech30": 0.0,
        "fr_da_rech30": 0.0,
        "cnt_da_rech90": 0.0,
        "fr_da_rech90": 0.0,
        "cnt_loans30": 1.0,
        "amnt_loans30": 12.0,
        "maxamnt_loans30": 12.0,
        "medianamnt_loans30": 12.0,
        "cnt_loans90": 1.0,
        "amnt_loans90": 12.0,
        "maxamnt_loans90": 12.0,
        "medianamnt_loans90": 12.0,
        "payback30": 2.0,
        "payback90": 2.0
    }

    # 3. Simulate a POST request exactly as a customer would send it
    response = client.post("/predict", json=valid_payload)

    # 4. Assert the mathematical and structural truths
    assert response.status_code == 200, f"API failed with error: {response.text}"
    
    response_data = response.json()
    assert "prediction_code" in response_data
    assert "risk_status" in response_data
    assert response_data["prediction_code"] in [0, 1]
    assert response_data["risk_status"] in ["Delinquent", "Non-Delinquent"]