# Ujian Akhir Semester (UAS) — Machine Learning End-to-End
**Mata Kuliah**: Machine Learning End-to-End  
**Institusi**: Institut Teknologi Tangerang Selatan (ITTS)  
**Kasus**: Kasus A — Klasifikasi: Prediksi Keterlambatan Pengiriman Paket (`POST /predict-keterlambatan`)

---

## 👤 Identitas Mahasiswa

- **Nama**: Akhmad Fatih Jalaluddin Rumi
- **NIM**: 1002240031
- **Kelas**: Eksekutif
- **Semester**: 4

---

## 📌 Deskripsi Masalah

Perusahaan e-commerce dan logistik mengalami kerugian operasional akibat keterlambatan pengiriman paket yang tidak terprediksi. Paket yang terlambat memicu keluhan pelanggan, denda kompensasi, dan penurunan kepuasan konsumen. 

Sistem Machine Learning End-to-End ini dibangun untuk **mengidentifikasi dan memprediksi paket yang berisiko terlambat** sebelum dikirimkan. Dengan mendeteksi paket berisiko tinggi secara dini, tim operasional logistik dapat memprioritaskan paket tersebut ke jalur pengiriman ekspres premium atau melakukan optimasi rute.

---

## 📊 Sumber Data & Lisensi

- **Dataset**: E-Commerce Shipping & Delivery Dataset
- **Sumber Data**: [Kaggle - E-Commerce Shipping Data](https://www.kaggle.com/datasets/prachi13/customer-analytics)
- **Lisensi**: CC0: Public Domain (Dapat digunakan kembali secara bebas tanpa batasan hak cipta)
- **Ukuran Dataset**: 10,999 baris data, 12 fitur (termasuk label target biner `Reached.on.Time_Y.N`).

---

## 🛠️ Lingkungan Pengkodian & Versi Dependensi

Proyek ini dibangun dan diuji menggunakan versi pustaka berikut:
- **Python**: `3.14.5`
- **pandas**: `3.0.5`
- **scikit-learn**: `1.9.0`
- **fastapi**: `0.139.2`
- **uvicorn**: `0.51.0`
- **pydantic**: `2.13.4`
- **joblib**: `1.5.3`
- **pytest**: `9.1.1`

---

## 📁 Struktur Repositori Project

```
.
├── src/
│   ├── load_data.py          # Memuat dataset mentah ke data/ & mencetak inspeksi data
│   ├── eda.py                # Menghasilkan 4 grafik analisis EDA ke reports/
│   ├── train.py              # Cross-validation 5-fold, evaluasi 3 algoritma & save model pipeline
│   └── evaluate.py           # Evaluasi test set single-pass & analisis 5 kesalahan terburuk
├── app/
│   └── main.py               # REST API FastAPI (lifespan model loader & validasi Pydantic)
├── tests/
│   └── test_api.py           # 6 test otomatis pytest (4 mekanis + 2 behavioral test)
├── data/                     # Dataset (diabaikan oleh .gitignore)
├── models/                   # Artefak model.joblib & metadata.json (diabaikan oleh .gitignore)
├── reports/                  # Grafik PNG EDA dan Evaluasi (dikomit ke git)
│   ├── target_distribution.png
│   ├── missing_values.png
│   ├── feature_correlations.png
│   ├── discount_vs_delay.png
│   ├── confusion_matrix.png
│   ├── roc_pr_curve.png
│   └── cost_threshold_curve.png
├── requirements.txt          # Dependensi lingkungan training
├── requirements-api.txt      # Dependensi versi ter-pin persis untuk lingkungan serving
├── .gitignore                # Aturan ignoransi file repositori git
└── README.md                 # Dokumentasi utama proyek
```

---

## 🚀 Langkah Menjalankan Proyek Dari Nol (Reproduksibilitas)

Ikuti langkah-langkah berikut untuk memproduksi ulang seluruh proyek dari clone awal hingga API server berjalan:

### 1. Clone Repositori & Navigasi ke Folder
```bash
git clone <URL_REPO_ANDA>
cd uas-ml-<nim>
```

### 2. Buat & Aktifkan Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependensi Training
```bash
pip install -r requirements.txt
```

### 4. Eksekusi Pipeline (Data -> EDA -> Training -> Evaluasi)
```bash
# Step A: Load & Inspect Data
python src/load_data.py

# Step B: Jalankan EDA & Hasilkan Grafik Reports
python src/eda.py

# Step C: Latih Model Pipeline & Simpan Artefak
python src/train.py

# Step D: Evaluasi Test Set Single-Pass
python src/evaluate.py
```

### 5. Jalankan Automated Tests (Pytest)
```bash
python -m pytest tests/ -v
```

### 6. Jalankan Server FastAPI (Serving)
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Buka browser Anda di `http://127.0.0.1:8000/docs` untuk mengakses OpenAPI Swagger UI interaktif.

---

## 🔌 Contoh Pemanggilan API (`curl`)

### A. Contoh Request Berhasil (HTTP 200 OK)

**Command `curl`**:
```bash
curl -X POST "http://127.0.0.1:8000/predict-keterlambatan" \
     -H "Content-Type: application/json" \
     -d '{
           "Warehouse_block": "F",
           "Mode_of_Shipment": "Flight",
           "Customer_care_calls": 4,
           "Customer_rating": 3,
           "Cost_of_the_Product": 210.0,
           "Prior_purchases": 3,
           "Product_importance": "high",
           "Gender": "F",
           "Discount_offered": 5.0,
           "Weight_in_gms": 4500.0
         }'
```

**Respons JSON Berhasil**:
```json
{
  "status": "success",
  "prediction_label": "terlambat",
  "is_delayed": true,
  "delay_probability": 0.6342,
  "risk_level": "MODERAT / RISIKO TERLAMBAT",
  "applied_threshold": 0.25,
  "recommendation": "Tandai paket untuk pemantauan rute dan optimasi penanganan gudang.",
  "model_version": "GradientBoosting-v1.0"
}
```

### B. Contoh Request Tidak Valid (HTTP 422 Unprocessable Entity)

**Command `curl` dengan enum & tipe data tidak valid**:
```bash
curl -X POST "http://127.0.0.1:8000/predict-keterlambatan" \
     -H "Content-Type: application/json" \
     -d '{
           "Warehouse_block": "Z",
           "Mode_of_Shipment": "Teleportation",
           "Customer_care_calls": -5
         }'
```

**Respons JSON Ditolak 422**:
```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "Warehouse_block"],
      "msg": "Input should be 'A', 'B', 'C', 'D' or 'F'"
    },
    {
      "type": "literal_error",
      "loc": ["body", "Mode_of_Shipment"],
      "msg": "Input should be 'Ship', 'Flight' or 'Road'"
    },
    {
      "type": "greater_than_equal",
      "loc": ["body", "Customer_care_calls"],
      "msg": "Input should be greater than or equal to 1"
    }
  ]
}
```

---

## ❓ Jawaban Pertanyaan Refleksi Modul

### Why are `data/` and `models/` in `.gitignore`?
Artefak `data/` dan `models/` berukuran besar dan dapat berubah setiap kali pelatihan diulang. Mengkomit file biner `.joblib` atau data mentah `.csv` mengotori histori git dan dapat menyebabkan *merge conflict*. Penguji tetap dapat memproduksi ulang seluruh isi `data/` dan `models/` dari nol cukup dengan menjalankan skrip deterministik `python src/load_data.py` dan `python src/train.py`.

### Why are dependencies in `requirements-api.txt` pinned while `requirements.txt` is not?
Lingkungan training (`requirements.txt`) fleksibel terhadap pembaruan pustaka minor untuk eksplorasi dan perbaikan bug scikit-learn. Sebaliknya, lingkungan produksi serving (`requirements-api.txt`) membutuhkan kestabilan mutlak di mana versi pustaka dikunci persis (`==`) untuk menjamin bahwasanya deserialisasi file `.joblib` dan perilakunya pada server production identik 100% dengan lingkungan pengujian tanpa adanya kejutan *breaking changes*.
