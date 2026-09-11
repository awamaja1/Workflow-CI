# Model Training & CI/CD Pipeline

Repositori ini merupakan bagian dari sistem *End-to-End Machine Learning* untuk memprediksi risiko kredit (*Credit Scoring*). Fokus utama repositori ini adalah pada **Pelatihan Model (*Model Training*)**, **Pelacakan Eksperimen (*Experiment Tracking*)**, dan **Continuous Integration (CI/CD)**.

## 📌 Deskripsi

Sistem otomasi adalah jantung dari praktik MLOps yang baik. Repositori ini membungkus model ke dalam format yang terstandardisasi dan memanfaatkan *GitHub Actions* untuk menjalankan uji coba dan melatih ulang model setiap kali terdapat pembaruan pada kode atau skema data.

## 🛠️ Fitur Utama
1. **Model Training:** Menggunakan algoritma klasifikasi `RandomForestClassifier` dari *scikit-learn* yang tangguh dalam menangani data *tabular*.
2. **MLflow Tracking:** Implementasi pelacakan otomatis (`mlflow.sklearn.autolog()`) untuk mencatat hyperparameter, metrik evaluasi (akurasi, presisi, recall), dan artifak model (*pickle*, `conda.yaml`) tanpa intervensi manual.
3. **MLProject Packaging:** Standardisasi lingkungan eksekusi menggunakan file `MLProject` dan `conda.yaml` sehingga model dapat dijalankan di berbagai *environment* secara identik.
4. **GitHub Actions CI Pipeline:** *Workflow* CI yang secara otomatis akan:
   - Memeriksa pembaruan kode.
   - Mengatur lingkungan *Python*.
   - Menginisiasi eksekusi `mlflow run` untuk melatih model setiap kali ada perubahan pada *branch* `main`.

## 📂 Struktur Direktori
- `MLProject/`: Root directory untuk eksekusi standar MLflow.
  - `MLProject`: File definisi *entry point* dan *environment*.
  - `conda.yaml`: Definisi *dependencies* lingkungan model.
  - `modelling.py`: *Script* inti untuk proses pelatihan.
  - `credit_scoring_preprocessing/`: Sampel data latih yang dibutuhkan *script*.
- `.github/workflows/main.yml`: Konfigurasi otomasi *pipeline* GitHub Actions.

## 🚀 Penggunaan Lokal
Untuk melakukan *training* model secara manual di mesin lokal Anda:
1. Pastikan Anda telah menginstal MLflow: `pip install mlflow scikit-learn pandas`
2. Jalankan UI Server MLflow: 
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```
3. Buka terminal baru dan jalankan proses training:
   ```bash
   export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
   mlflow run MLProject/
   ```
4. Kunjungi `http://127.0.0.1:5000` di browser untuk memantau hasil *training*.
