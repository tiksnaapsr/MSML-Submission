import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Konfigurasi Logging ke DagsHub MLflow
DAGSHUB_REPO_OWNER = "ayutksnaa"     
DAGSHUB_REPO_NAME = "msml-repo" 
# Ganti dengan Tracking URI dari tombol 'Remote' di DagsHub
MLFLOW_TRACKING_URI = "https://dagshub.com/ayutksnaa/msml-repo.mlflow" 

# Inisialisasi DagsHub MLflow Tracking
dagshub.init(repo_owner='ayutksnaa', repo_name='msml-repo', mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load Data dari preprocessing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Mengarah ke folder preprocessing
TRAIN_PATH = os.path.join(BASE_DIR, 'water_potability_preprocessing_automated/train_clean_automated.csv')
TEST_PATH = os.path.join(BASE_DIR, 'water_potability_preprocessing_automated/test_clean_automated.csv')

print("Memuat data...")
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
except FileNotFoundError:
    print("Error: File dataset tidak ditemukan. Pastikan preprocessing sudah dijalankan.")
    exit()

X_train = train_df.drop('Potability', axis=1)
y_train = train_df['Potability']
X_test = test_df.drop('Potability', axis=1)
y_test = test_df['Potability']

# Hyperparameter Tuning dengan RandomizedSearchCV
print("Memulai Hyperparameter Tuning...")

rf = RandomForestClassifier(random_state=42)

# Menentukan ruang pencarian parameter
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Menggunakan RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,       # Mencoba 10 kombinasi acak
    cv=3,            # 3-fold Cross Validation
    verbose=1,
    random_state=42,
    n_jobs=-1        # Gunakan semua core CPU
)

# Latih model untuk mencari parameter terbaik
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_params = random_search.best_params_
print(f"Tuning Selesai! Parameter Terbaik: {best_params}")

# Evaluasi Model Terbaik
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"📊 Metrics -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# MLflow Manual Logging ke DagsHub
mlflow.sklearn.autolog(disable=True)    # Nonaktifkan autologging otomatis

experiment_name = "Water_Potability_Advanced"
mlflow.set_experiment(experiment_name)

print("Mengirim log ke DagsHub...")
with mlflow.start_run(run_name="RandomForest_Tuned_Advanced"):
    
    # Manual Logging Parameter
    mlflow.log_params(best_params)
    
    # Manual Logging Metrics
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    
    # Manual Logging Artefak Tambahan
    
    # --- Artefak 1: Confusion Matrix ---
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_filename = "confusion_matrix.png"
    plt.savefig(cm_filename)
    mlflow.log_artifact(cm_filename) # Upload gambar
    plt.close()
    print("   -> Artefak 1 (Confusion Matrix) terupload.")

    # --- Artefak 2: Feature Importance ---
    plt.figure(figsize=(8,6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_train.columns
    
    sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    
    fi_filename = "feature_importance.png"
    plt.savefig(fi_filename)
    mlflow.log_artifact(fi_filename) # Upload gambar
    plt.close()
    print("   -> Artefak 2 (Feature Importance) terupload.")
    
    # Logging Model (Menyimpan file model .pkl)
    signature = infer_signature(X_train, y_pred)
    input_example = X_train.head(1)

    mlflow.sklearn.log_model(
        best_model,
        "model",
        signature=signature,
        input_example=input_example
    )
    
    # Bersihkan file gambar sementara di lokal
    os.remove(cm_filename)
    os.remove(fi_filename)

print("Selesai! Cek DagsHub Anda sekarang.")