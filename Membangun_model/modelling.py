
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import os

# Init DagsHub Connection
# GitHub Actions environment variables
DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")
DAGSHUB_REPO_NAME = "Eksperimen_SML"

if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
    # Set Tracking URI manually to avoid dagshub.init() auth issues
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI set to: {tracking_uri}")
else:
    print("Warning: DAGSHUB_USERNAME or DAGSHUB_TOKEN not found. Tracking may fail.")

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def main():
    print("Mulai Training dengan Parameter Default...")
    # Load Data
    try:
        df = pd.read_csv('data/processed/train_processed.csv')
    except FileNotFoundError:
        print("Data processed tidak ditemukan. Pastikan preprocessing sudah jalan.")
        return

    # Ambil 1 SKU saja untuk demo cepat
    target_sku = 216418
    print(f"Filtering for SKU: {target_sku}")
    
    # Debug: Check available SKUs
    print(f"Available SKUs in data: {df['sku_id'].unique()}")
    
    df = df[df['sku_id'] == target_sku]
    
    if df.empty:
        print(f"Error: No data found for SKU {target_sku}!")
        return
    
    # Pilih Fitur
    features = [c for c in df.columns if 'lag_' in c or 'ma_' in c or 'store_encoded' in c or 'total_price' in c]
    X = df[features]
    y = df['units_sold']
    
    mlflow.set_experiment("Retail_Demand_Ops")
    
    # ENABLE AUTOLOG
    # Log models=True ensures the model is saved.
    # registered_model_name ensures it's registered in Model Registry.
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True, registered_model_name="Retail_Forecasting_Model")

    # --- TRAINING WITH DEFAULT PARAMETERS (NO TUNING) ---
    with mlflow.start_run(run_name="Default_Parameters_Model"):
        # Manual parameters - no hyperparameter tuning
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42
        }
        
        # Log manual parameters
        mlflow.log_params(params)
        
        model = RandomForestRegressor(**params)
        
        # Simple train-validation split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Validation
        preds_val = model.predict(X_val)
        preds_val = np.maximum(preds_val, 0)
        error = rmsle(y_val, preds_val)
        
        mlflow.log_metric("validation_rmsle", error)
        print(f"Validation RMSLE: {error:.4f}")
        
        # Retrain on full data
        model.fit(X, y)
        
        # Generate Artifacts
        # 1. Feature Importance
        plt.figure(figsize=(8, 5))
        pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(kind='barh')
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        # 2. Actual vs Predicted
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        plt.figure(figsize=(8, 5))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.tight_layout()
        plt.savefig("prediction_check.png")
        mlflow.log_artifact("prediction_check.png")
        
        # 3. Residual Plot
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig("residual_plot.png")
        mlflow.log_artifact("residual_plot.png")

        # 4. Error Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.tight_layout()
        plt.savefig("error_distribution.png")
        mlflow.log_artifact("error_distribution.png")
        
        # 5. Upload Requirements
        if os.path.exists("Membangun_model/requirements.txt"):
             mlflow.log_artifact("Membangun_model/requirements.txt")
        
        print("Model dengan parameter default berhasil di-log ke DagsHub!")

if __name__ == "__main__":
    main()
