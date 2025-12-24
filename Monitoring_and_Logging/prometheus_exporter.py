
from prometheus_client import start_http_server, Gauge, Counter, Summary
import requests
from fastapi import FastAPI
import uvicorn
import random
import threading
import time

# Metrics
PREDICTION_VALUE = Gauge('retail_prediction_qty', 'Hasil Prediksi Qty')
REQUEST_COUNT = Counter('request_count', 'Total Request')
LATENCY = Summary('process_latency', 'Waktu proses')

app = FastAPI()

@app.get("/")
def root():
    return {"message": "MLOps Retail Monitoring Running"}

@app.post("/predict")
@LATENCY.time()
def predict(data: dict):
    REQUEST_COUNT.inc()
    
    # Forward to MLflow Model Server
    # Assumes MLflow Serve is running on port 5001
    mlflow_url = "http://localhost:5001/invocations"
    
    try:
        # Wrap data for MLflow dataframe_records format
        # If data is already in correct format, use it directly, else wrap it
        if "dataframe_records" in data or "dataframe_split" in data or "instances" in data:
            payload = data
        else:
            payload = {"dataframe_records": [data]}
            
        response = requests.post(mlflow_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Handle different return formats (list of predictions or dict)
            if "predictions" in result:
                pred = result["predictions"][0]
            elif isinstance(result, list):
                pred = result[0]
            else:
                pred = 0 # Fallback
                
            PREDICTION_VALUE.set(pred)
            return {"prediction": pred}
        else:
            # Fallback for demonstration if MLflow server is not running or input invalid
            print(f"MLflow Serve Error: {response.text}")
            pred = random.uniform(10, 50) # Fallback simulation so metrics still show up
            PREDICTION_VALUE.set(pred)
            return {"prediction": pred, "status": "fallback_simulation", "error": response.text}
            
    except Exception as e:
        print(f"Connection Error: {e}")
        pred = random.uniform(10, 50)
        PREDICTION_VALUE.set(pred)
        return {"prediction": pred, "status": "fallback_simulation", "error": str(e)}

def start_metrics():
    start_http_server(8000)

if __name__ == "__main__":
    t = threading.Thread(target=start_metrics)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=5000)
