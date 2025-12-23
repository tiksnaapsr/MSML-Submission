# MSML Submission – Water Potability Classification

Repositori ini berisi pipeline lengkap untuk submission "Membangun Sistem Machine Learning" dengan studi kasus klasifikasi kelayakan air minum (Water Potability). Seluruh tahapan mulai dari pembersihan data, training model dasar, tuning lanjutan, sampai pelacakan eksperimen disiapkan agar mudah direproduksi.

## Highlights
- Otomatisasi preprocessing: imputasi nilai hilang, standardisasi fitur, dan split stratifikasi tersedia di script tunggal.
- Dua alur pelatihan: baseline Random Forest dengan MLflow autolog dan versi lanjutan dengan RandomizedSearchCV.
- Tracking eksperimen ganda: lokal melalui `mlruns/` dan remote ke DagsHub/MLflow untuk kolaborasi.
- Struktur repo sederhana sehingga mudah diadaptasi ke eksperimen lain.

## Struktur Repositori
```
.
├── experiment/
│   └── preprocessing/
│       ├── automate.py                 # Pipeline preprocessing otomatis
│       ├── raw_data/water_potability.csv
│       └── water_potability_preprocessing/
├── modelling/
│   ├── modelling.py                    # Training baseline + MLflow autolog
│   ├── modelling_tuning.py             # RandomizedSearch + logging manual
│   └── water_potability_preprocessing_automated/
├── mlruns/                             # Artefak MLflow lokal
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset & Preprocessing
- Dataset: Water Potability (Kaggle) dengan fitur seperti pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, dan Turbidity, serta target `Potability` (0 = tidak layak, 1 = layak).
- File sumber berada di `experiment/preprocessing/raw_data/water_potability.csv`.
- Script `experiment/preprocessing/automate.py` melakukan:
  1. Pembagian data train/test 80/20 dengan stratify target.
  2. Imputasi rata-rata untuk nilai hilang.
  3. Standardisasi seluruh fitur numerik.
  4. Penyimpanan hasil ke `modelling/water_potability_preprocessing_automated/` agar bisa dikonsumsi modul training.

## Persiapan Lingkungan
1. Gunakan Python 3.10 atau lebih baru.
2. Buat virtual environment (opsional tapi direkomendasikan):
	```powershell
	python -m venv .venv
	.\.venv\Scripts\activate
	```
3. Pasang dependensi:
	```powershell
	pip install -r requirements.txt
	```

## Menjalankan Pipeline
1. **Preprocessing**
	```powershell
	python experiment/preprocessing/automate.py
	```
	Output: `train_clean_automated.csv` dan `test_clean_automated.csv` di folder `modelling/water_potability_preprocessing_automated/`.

2. **Training Baseline** (`RandomForestClassifier`, 100 trees):
	```powershell
	python modelling/modelling.py
	```
	- Mengaktifkan `mlflow.autolog()` sehingga parameter, metrik, dan model tersimpan otomatis di folder `mlruns/`.
	- Jalankan `mlflow ui` untuk melihat dashboard lokal (default di http://127.0.0.1:5000).

3. **Training Lanjutan + Hyperparameter Tuning**:
	```powershell
	python modelling/modelling_tuning.py
	```
	- Menggunakan `RandomizedSearchCV` (10 kombinasi, 3-fold) atas parameter `n_estimators`, `max_depth`, `min_samples_split`, dan `min_samples_leaf`.
	- Logging dilakukan manual ke experiment `Water_Potability_Advanced`.

## Tracking ke DagsHub / MLflow Remote
Script tuning sudah memanggil `dagshub.init()` dengan repo `ayutksnaa/msml-repo`. Agar logging berhasil:
1. Buat Personal Access Token di DagsHub.
2. Ekspor kredensial sebelum menjalankan script, misalnya di PowerShell:
	```powershell
	$env:DAGSHUB_USER="USERNAME_ANDA"
	$env:DAGSHUB_TOKEN="TOKEN_DAGSHUB"
	```
3. Jalankan `python modelling/modelling_tuning.py` seperti biasa. Semua artefak (confusion matrix, feature importance, model) akan muncul di tab MLflow pada repo DagsHub.

## Hasil Sementara
Metrik berikut berasal dari run `Basic_Training_Autolog` yang tersimpan lokal (dataset train). Nilai validasi/test akan muncul setelah menjalankan evaluasi tambahan di script tuning.

| Run                     | Split  | Accuracy | Precision | Recall | F1 |
|-------------------------|--------|----------|-----------|--------|----|
| Basic_Training_Autolog  | Train  | 1.00     | 1.00      | 1.00   | 1.00 |

Catatan: skor sempurna pada data train mengindikasikan potensi overfitting. Gunakan metrik test (mis. lewat `RandomizedSearchCV` atau evaluasi terpisah) untuk menilai generalisasi.


