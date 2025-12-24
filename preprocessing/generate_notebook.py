
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

# 1. Metadata / Perkenalan Dataset
text_intro = """# **1. Perkenalan Dataset**

**Nama Dataset**: Retail Demand Forecasting (Example)
**Sumber**: Data Internal / Kaggle
**Deskripsi**: Dataset ini berisi data penjualan historis untuk produk ritel (SKU) di berbagai toko. Tujuan eksperimen ini adalah memprediksi `units_sold` berdasarkan fitur historis dan eksternal.

Informasi Dataset:
- `week`: Tanggal penjualan (mingguan)
- `sku_id`: ID Produk
- `store_id`: ID Toko
- `base_price`: Harga dasar
- `total_price`: Harga total (setelah diskon)
- `units_sold`: Target variabel (jumlah unit terjual)
"""

# 2. Import Library
text_import = """# **2. Import Library**
Pada tahap ini, library yang dibutuhkan untuk pemrosesan data, visualisasi, dan pemodelan diimpor.
"""
code_import = """import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import MEstimateEncoder

# Pastikan config.py ada di folder yang sama atau definisikan di sini
try:
    from config import SKU_SPECIFIC_LAGS, SKU_SPECIFIC_MAS
except ImportError:
    # Definisi fallback jika config.py tidak ditemukan
    SKU_SPECIFIC_LAGS = {216418: [1, 2, 3]}
    SKU_SPECIFIC_MAS = {216418: [2, 4]} 
"""

# 3. Memuat Dataset
text_load = """# **3. Memuat Dataset**
Memuat dataset `train.csv` ke dalam DataFrame pandas.
"""
code_load = """try:
    train = pd.read_csv('../data/train.csv') # Path relative menyesuaikan struktur folder
    print("Dataset loaded successfully.")
except FileNotFoundError:
    # Fallback path
    if os.path.exists('data/train.csv'):
        train = pd.read_csv('data/train.csv')
    else:
        print("Error: train.csv not found.")

train.head()
"""

# 4. EDA
text_eda = """# **4. Exploratory Data Analysis (EDA)**
Melihat struktur data, tipe data, missing values, dan statistik deskriptif.
"""
code_eda = """# Cek Info Data
print("Data Info:")
print(train.info())

# Cek Statistik Deskriptif
print("\\nStatistik Deskriptif:")
print(train.describe())

# Cek Missing Values
print("\\nMissing Values:")
print(train.isnull().sum())

# Cek Duplikasi
print(f"\\nJumlah Duplikat: {train.duplicated().sum()}")

# Visualisasi Outlier Sederhana (Boxplot units_sold)
plt.figure(figsize=(8, 4))
sns.boxplot(x=train['units_sold'])
plt.title('Boxplot Units Sold')
plt.show()
"""

# 5. Preprocessing
text_pre = """# **5. Data Preprocessing**
Tahapan pembersihan dan penyiapan data:
1. Konversi Tipe Data (Date)
2. Handling Missing Values
3. Handling Duplicates
4. Encoding (Target Encoding)
5. Feature Engineering (Lags & Moving Averages)
"""

code_pre = """print("Mulai Preprocessing...")

# 1. Konversi Date & Fix Leap Year
train['week'] = pd.to_datetime(train['week'], format='%d/%m/%y')
train.loc[train['week'] >= '2012-03-06', 'week'] -= pd.Timedelta(days=1)

# 2. Handling Missing Values (Total Price)
train['total_price'] = train['total_price'].fillna(train['base_price'])
print("Missing values in total_price filled.")

# 3. Handling Duplicates (Jika ada)
initial_rows = len(train)
train = train.drop_duplicates()
print(f"Duplicates dropped: {initial_rows - len(train)}")

# 4. Target Encoding
print("Melakukan Target Encoding...")
encoder = MEstimateEncoder(cols=['store_id'])
encoder.fit(train[['store_id']], train['units_sold'])

# Simpan Encoder
os.makedirs('model_artifacts', exist_ok=True)
joblib.dump(encoder, 'model_artifacts/store_encoder.pkl')
train['store_encoded'] = encoder.transform(train[['store_id']])

# 5. Feature Engineering (Lags & MA)
# Filter SKU sesuai eksperimen (untuk efisiensi)
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
print(f"Data final shape: {df_final.shape}")
"""

# 6. Simpan Hasil
text_save = """# **6. Simpan Hasil**
Menyimpan data yang sudah diproses ke `train_processed.csv`.
"""
code_save = """os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/train_processed.csv'
df_final.to_csv(output_path, index=False)
print(f"Data tersimpan di {output_path}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_markdown_cell(text_import),
    nbf.v4.new_code_cell(code_import),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda),
    nbf.v4.new_markdown_cell(text_pre),
    nbf.v4.new_code_cell(code_pre),
    nbf.v4.new_markdown_cell(text_save),
    nbf.v4.new_code_cell(code_save)
]

with open('preprocessing/Eksperimen_SML_IMadeWisnuAdiSanjaya.ipynb', 'w') as f:
    nbf.write(nb, f)
    
print("Notebook created successfully.")
