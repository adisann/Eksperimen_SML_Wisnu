@echo off
echo ===================================================
echo MLOps Serving & Monitoring Launcher
echo ===================================================
echo Ensure you have trained the model at least once so it is registered in MLflow.
echo Model Name: Retail_Forecasting_Model
echo.

echo Starting MLflow Model Serving on port 5001...
echo (Check the new window for logs)
start "MLflow Serve" mlflow models serve -m models:/Retail_Forecasting_Model/latest -p 5001 --no-conda

echo.
echo Waiting 15 seconds for Model Server to initialize...
timeout /t 15

echo.
echo Starting Prometheus Exporter (Monitoring Proxy) on port 8000 (Metrics) / 5000 (API)...
echo Access Metrics at: http://localhost:8000
echo Send Predictions to: http://localhost:5000/predict
python Monitoring_and_Logging/prometheus_exporter.py
