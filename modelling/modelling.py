import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# KONFIGURASI PATH
# Mengambil data dari folder preprocessing yang sudah dibuat sebelumnya
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'water_potability_preprocessing_automated/train_clean_automated.csv')
TEST_PATH = os.path.join(BASE_DIR, 'water_potability_preprocessing_automated/test_clean_automated.csv')

def main():
    print("--- MEMULAI TRAINING MODEL ---")
    
    # Load Data
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        print("Data berhasil dimuat.")
    except FileNotFoundError:
        print("File data tidak ditemukan. Jalankan script preprocessing dulu!")
        return

    # Persiapan Data (Split X dan y)
    X_train = train_df.drop('Potability', axis=1)
    y_train = train_df['Potability']
    X_test = test_df.drop('Potability', axis=1)
    y_test = test_df['Potability']

    # Set Experiment Name
    mlflow.set_experiment("Water_Potability_Basic")

    # Aktifkan Autolog 
    # Ini akan otomatis mencatat parameter, metrik, dan model tanpa diketik manual
    mlflow.autolog()

    # Training Model
    print("Melatih model Random Forest...")
    with mlflow.start_run(run_name="Basic_Training_Autolog"):
        
        # Inisialisasi Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Training
        model.fit(X_train, y_train)
        
        # Evaluasi sederhana (Opsional, karena autolog sudah menghitungnya)
        # Print untuk konfirmasi di terminal
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc:.4f}")
        
    print("--- TRAINING SELESAI ---")
    print("Cek hasil di MLflow UI dengan menjalankan perintah: mlflow ui")

if __name__ == "__main__":
    main()