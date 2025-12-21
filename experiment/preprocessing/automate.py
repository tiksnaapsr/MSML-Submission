import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

# Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'raw_data/water_potability.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'water_potability_preprocessing')

def load_data(path):
    """
    Fungsi untuk memuat data dari file CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {path}")
    
    print(f"✅ Memuat data dari: {path}")
    return pd.read_csv(path)

def save_data(X, y, output_path):
    """
    Fungsi untuk menggabungkan fitur & target, lalu menyimpannya ke CSV.
    """
    # Gabungkan kembali Fitur (X) dan Target (y)
    data = pd.concat([X, y.reset_index(drop=True)], axis=1)
    
    # Simpan ke CSV
    data.to_csv(output_path, index=False)
    print(f"Data tersimpan di: {output_path}")

def main():
    print("--- MULAI PROSES OTOMATISASI PREPROCESSING ---")
    
    # Load Data
    df = load_data(INPUT_PATH)
    
    # Pisahkan Fitur dan Target
    if 'Potability' not in df.columns:
        raise ValueError("Kolom target 'Potability' hilang!")
        
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Split Data (Train & Test)
    print("🔄 Membagi data menjadi Train (80%) dan Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Buat Pipeline Preprocessing
    # Menggabungkan Imputer (isi null) dan Scaler (normalisasi)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Eksekusi Preprocessing (Fit & Transform)
    print("Melakukan pembersihan dan scaling...")
    
    # Belajar (Fit) hanya dari data Train, lalu Terapkan (Transform) ke Train & Test
    X_train_clean = pipeline.fit_transform(X_train)
    X_test_clean = pipeline.transform(X_test)
    
    # Kembalikan ke bentuk DataFrame (karena output scikit-learn adalah numpy array)
    X_train_df = pd.DataFrame(X_train_clean, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_clean, columns=X.columns)
    
    # Simpan Hasil
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Buat folder output jika belum ada
    
    save_data(X_train_df, y_train, os.path.join(OUTPUT_DIR, 'train_clean.csv'))
    save_data(X_test_df, y_test, os.path.join(OUTPUT_DIR, 'test_clean.csv'))
    
    print("--- PREPROCESSING SELESAI SUKSES ---")

if __name__ == "__main__":
    main()