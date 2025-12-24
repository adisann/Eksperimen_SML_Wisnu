import requests
import time

# URL endpoint FastAPI (pastikan port sesuai dengan prometheus_exporter.py)
# prometheus_exporter.py: uvicorn.run(app, host="0.0.0.0", port=5000)
URL = "http://localhost:5000/predict" 

print(f"Starting traffic generator sending requests to {URL}...")

while True:
    try:
        # Data dummy
        payload = {"data": 1}
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            print(f"Request sent successfully. Prediction: {response.json()}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("Connection error: Is the exporter running?")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    time.sleep(2)
