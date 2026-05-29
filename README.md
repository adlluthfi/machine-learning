# ML Model Prediction Web Application

Aplikasi Flask untuk prediksi kesehatan menggunakan Machine Learning Gaussian Naive Bayes.

## 📁 Struktur Folder

```
flask-app/
├── app.py                 # Flask main application
├── requirements.txt       # Python dependencies
├── README.md             # Dokumentasi
├── models/               # Folder untuk menyimpan model pickle
│   ├── diabetes_model.pkl
│   ├── diabetes_metadata.json
│   ├── blood_test_model.pkl
│   ├── blood_test_metadata.json
│   ├── sleep_disorder_model.pkl
│   └── sleep_disorder_metadata.json
├── templates/            # HTML templates
│   ├── index.html        # Dashboard halaman utama
│   ├── diabetes.html     # Halaman prediksi Diabetes
│   ├── blood_test.html   # Halaman prediksi Blood Test
│   ├── sleep_disorder.html # Halaman prediksi Sleep Disorder
│   ├── test_model.html   # Halaman untuk test model
│   ├── about.html        # Halaman tentang aplikasi
│   └── 404.html          # Halaman error
├── static/               # Static files
│   ├── css/
│   │   └── style.css     # Styling utama
│   └── js/               # JavaScript files (opsional)
├── tests/                # Unit tests
│   └── test_app.py       # Test file
└── data/                 # Sample data (opsional)
```

## 🎯 5 Halaman Utama

1. **Dashboard** (index.html) - Menampilkan semua model dan metriknya
2. **Prediksi Diabetes** (diabetes.html) - Form untuk prediksi diabetes
3. **Prediksi Blood Test** (blood_test.html) - Form untuk prediksi  memprediksi klasifikasi pasien) berdasarkan data pemeriksaan darah
4. **Prediksi Sleep Disorder** (sleep_disorder.html) - Form untuk deteksi gangguan tidur
5. **Test Model** (test_model.html) - Untuk testing akurasi model

Plus:
- **About Page** - Informasi tentang aplikasi
- **Error Page (404)** - Halaman error

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Salin Model dari Training

Sebelum menjalankan Flask app, salin model yang sudah dilatih:

```bash
# Jalankan tubes.py terlebih dahulu untuk generate model
python tubes.py

# Salin folder models dari output ke flask-app/models
cp -r tubes/outputs/models/* flask-app/models/
```

### 3. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000`

## 🧪 Menjalankan Tests

```bash
# Run semua tests
python -m pytest tests/

# Atau
python tests/test_app.py
```

## 📊 API Endpoints

### GET Endpoints

- `GET /` - Dashboard halaman utama
- `GET /about` - Halaman tentang
- `GET /diabetes` - Halaman prediksi diabetes
- `GET /blood-test` - Halaman prediksi blood test
- `GET /sleep-disorder` - Halaman prediksi sleep disorder
- `GET /test-model` - Halaman test model
- `GET /api/model-info/<model_name>` - Get info model

### POST Endpoints

- `POST /api/predict` - Prediksi menggunakan model
  ```json
  {
    "model_name": "diabetes",
    "features": [0, 137, 40, 35, 168, 43.1, 2.288, 33]
  }
  ```

- `POST /api/test-accuracy` - Test accuracy model
  ```json
  {
    "actual": [1, 0, 1, 0],
    "predicted": [1, 0, 1, 1]
  }
  ```

## 📋 Model yang Tersedia

### 1. Diabetes Predictor
- **Target:** Prediksi diabetes (0/1)
- **Features:** 8 variabel kesehatan
- **Input:** pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age

### 2. Blood Test Analyzer
- **Target:** Sumber tes darah (Normal/Patient)
- **Features:** 6 variabel tes darah
- **Input:** age, wbc, rbc, hgb, hct, plt

### 3. Sleep Disorder Detector
- **Target:** Jenis gangguan tidur (None/Insomnia/Sleep Apnea)
- **Features:** 7 variabel tidur
- **Input:** age, sleep_duration, quality_of_sleep, physical_activity, stress_level, heart_rate, daily_steps

## 📈 Fitur Aplikasi

✅ Dashboard dengan metrik model  
✅ Prediksi real-time  
✅ Test akurasi model  
✅ API endpoints  
✅ Responsive design  
✅ Unit tests  

## ⚙️ Konfigurasi

Edit file `app.py` untuk mengubah:
- Port: `port=5000`
- Host: `host='localhost'`
- Debug mode: `debug=True`

## 📝 Catatan

- Model disimpan dalam format pickle (.pkl)
- Metadata model disimpan dalam format JSON (.json)
- Aplikasi menggunakan Gaussian Naive Bayes
- Semua input akan di-scale sebelum prediksi

## ⚠️ Disclaimer

Aplikasi ini untuk tujuan edukatif. Selalu konsultasikan dengan profesional medis untuk diagnosis yang akurat.

## 👨‍💻 Developer

Created by: 
- ABdullah Luthfi (luthfikkc@gmail.com)
- Daniel Febrian Sijabat (danielsijaban@gmail.com)

Date: 2026

---

**Untuk pertanyaan atau bantuan, silakan hubungi developer.**
