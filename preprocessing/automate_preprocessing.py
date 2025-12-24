
import pandas as pd
import numpy as np
import joblib
from category_encoders import MEstimateEncoder
from config import SKU_SPECIFIC_LAGS, SKU_SPECIFIC_MAS
import os
import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")

def preprocess():
    print("Mulai Preprocessing...")
    # Load Data
    train = pd.read_csv('data/train.csv')
    
    # 1. Convert Date
    train['week'] = pd.to_datetime(train['week'], format='%d/%m/%y')
    
    # 2. Fix Leap Year (Sesuai analisis notebook Anda)
    train.loc[train['week'] >= '2012-03-06', 'week'] -= pd.Timedelta(days=1)
    
    # 3. Fill Missing Price
    train['total_price'] = train['total_price'].fillna(train['base_price'])
    
    # 4. Target Encoding
    print("Melakukan Target Encoding...")
    encoder = MEstimateEncoder(cols=['store_id'])
    encoder.fit(train[['store_id']], train['units_sold'])
    
    # Simpan Encoder (Penting untuk Inference)
    os.makedirs('model_artifacts', exist_ok=True)
    joblib.dump(encoder, 'model_artifacts/store_encoder.pkl')
    
    train['store_encoded'] = encoder.transform(train[['store_id']])
    
    # 5. Feature Engineering (Lags & MA)
    # Kita filter dataset agar tidak terlalu besar saat demo
    target_skus = list(SKU_SPECIFIC_LAGS.keys())
    train = train[train['sku_id'].isin(target_skus)].copy()
    
    print("Membuat Fitur Time Series...")
    df_list = []
    for sku_id in target_skus:
        df_sku = train[train['sku_id'] == sku_id].copy().sort_values('week')
        
        # Lag
        lags = SKU_SPECIFIC_LAGS.get(sku_id, [1])
        for lag in lags:
            df_sku[f'lag_{lag}'] = df_sku['units_sold'].shift(lag)
            
        # Moving Average
        mas = SKU_SPECIFIC_MAS.get(sku_id, [1])
        for window in mas:
            df_sku[f'ma_{window}'] = df_sku['units_sold'].rolling(window=window).mean().shift(1)
            
        df_list.append(df_sku)
        
    df_final = pd.concat(df_list)
    df_final = df_final.dropna()
    
    # 6. Simpan Hasil
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_csv('data/processed/train_processed.csv', index=False)
    print("Selesai! Data tersimpan di data/processed/train_processed.csv")

if __name__ == "__main__":
    preprocess()
