
from prometheus_client import start_http_server, Gauge, Counter, Summary
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
    
    # Simulasi prediksi (Agar tidak ribet load model asli saat monitoring demo)
    # Di dunia nyata, Anda load model.pkl disini
    pred = random.uniform(10, 50) 
    
    PREDICTION_VALUE.set(pred)
    return {"prediction": pred}

def start_metrics():
    start_http_server(8000)

if __name__ == "__main__":
    t = threading.Thread(target=start_metrics)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=5000)
