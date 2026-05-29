# Setup Model XGBoost Patient Treatment untuk Blood Test

## 📋 Daftar Isi
1. [Overview](#overview)
2. [File yang Diperlukan](#file-yang-diperlukan)
3. [Struktur Model](#struktur-model)
4. [Cara Menggunakan](#cara-menggunakan)
5. [Testing dengan Script](#testing-dengan-script)
6. [API Endpoints](#api-endpoints)
7. [Frontend Integration](#frontend-integration)

---

## Overview

Model ini adalah XGBoost yang dilatih untuk memprediksi **SOURCE** (sumber/klasifikasi pasien) berdasarkan data pemeriksaan darah dari dataset `data-ori.csv`.

### Dataset
- **File**: `data-ori.csv`
- **Target**: `SOURCE` (klasifikasi pasien)
- **Total Fitur**: 11 kolom

### Fitur yang Digunakan
```
1. HAEMATOCRIT      - Persentase sel darah merah
2. ERYTHROCYTE      - Jumlah eritrosit
3. LEUCOCYTE        - Jumlah leukosit
4. THROMBOCYTE      - Jumlah trombosit
5. MCH              - Mean Corpuscular Hemoglobin
6. MCV              - Mean Corpuscular Volume
7. AGE              - Usia pasien
8. SEX              - Jenis kelamin (M/F)
```

### Fitur yang Dikecualikan
- `SOURCE` - Target variable (bukan fitur)
- `MCHC` - Tidak digunakan
- `HAEMOGLOBINS` - Tidak digunakan

---

## File yang Diperlukan

Untuk menjalankan model ini, Anda memerlukan 3 file yang harus ditempatkan di folder `models/`:

### 1. **model-xgboost-patient-treatment.pkl**
```
models/
└── model-xgboost-patient-treatment.pkl
```
File pickle yang berisi model XGBoost yang sudah dilatih.

### 2. **label-encoder-patient-treatment.pkl**
```
models/
└── label-encoder-patient-treatment.pkl
```
File pickle yang berisi LabelEncoder untuk mengubah prediksi angka menjadi label asli.

**Contoh encoding**:
```
0 -> 'out'      (atau label pertama dari training)
1 -> 'in'       (atau label kedua dari training)
```

### 3. **feature-columns-patient-treatment.pkl**
```
models/
└── feature-columns-patient-treatment.pkl
```
File pickle yang berisi urutan kolom fitur yang tepat sesuai training.

**Contoh list**:
```python
['HAEMATOCRIT', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCV', 'AGE', 'SEX']
```

---

## Struktur Model

### Di Backend (Flask)

#### 1. **app.py** - Modifikasi yang dilakukan:

**a. Tambahan Dictionary untuk Label Encoders dan Feature Columns**
```python
label_encoders = {}      # Menyimpan label encoder untuk setiap model
feature_columns = {}     # Menyimpan urutan fitur untuk setiap model
```

**b. Fungsi `load_models()` - Enhanced**
- Otomatis load label encoder jika file ada
- Otomatis load feature columns jika file ada

**c. Endpoint Baru: `/api/predict-blood-test` [POST]**
```python
@app.route('/api/predict-blood-test', methods=['POST'])
def predict_blood_test():
    """API untuk prediksi blood test dengan encoding dan decoding"""
```

Endpoint ini melakukan:
1. ✅ Encoding SEX dari 'M'/'F' ke angka
2. ✅ Reorder fitur sesuai feature_columns
3. ✅ Prediksi menggunakan model
4. ✅ Decoding hasil prediksi ke label asli
5. ✅ Return probabilitas untuk setiap kelas

---

## Cara Menggunakan

### 1. Persiapan File

Pastikan file berikut sudah ada di folder `models/`:

```bash
models/
├── model-xgboost-patient-treatment.pkl
├── label-encoder-patient-treatment.pkl
└── feature-columns-patient-treatment.pkl
```

### 2. Update Flask App

Pastikan Anda sudah melakukan modifikasi pada `app.py` seperti yang dijelaskan di atas.

### 3. Restart Flask

```bash
# Terminal
python app.py
```

Server akan berjalan di `http://localhost:5000`

### 4. Akses Halaman Blood Test

Buka browser dan navigasi ke:
```
http://localhost:5000/blood-test
```

---

## Testing dengan Script

### Cara Menjalankan

```bash
python test-blood-prediction.py
```

### Output Script

Script ini akan:
1. ✅ Load model dari file
2. ✅ Load label encoder
3. ✅ Load feature columns
4. ✅ Melakukan prediksi pada 4 sample data berbeda
5. ✅ Menampilkan hasil prediksi dengan probabilitas

### Contoh Output

```
================================================================================
🔬 TEST PREDIKSI BLOOD TEST - PATIENT TREATMENT MODEL
================================================================================

📦 Loading model dari: models/xgboost-patient-treatment_model.pkl
✅ Model loaded successfully

📦 Loading label encoder dari: models/label-encoder-patient-treatment.pkl
✅ Label encoder loaded successfully
   Classes: ['in' 'out']

📦 Loading feature columns dari: models/feature-columns-patient-treatment.pkl
✅ Feature columns loaded successfully
   Columns: ['HAEMATOCRIT', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCV', 'AGE', 'SEX']

[Sample 1] Sample 1 - Data Normal
────────────────────────────────────────────────────────────────────────────────
📥 Input Data:
   HAEMATOCRIT: 35.1
   ERYTHROCYTE: 4.65
   LEUCOCYTE: 6.3
   THROMBOCYTE: 310
   MCH: 25.4
   MCV: 75.5
   AGE: 1
   SEX: F

📊 Hasil Prediksi:
   Prediksi (angka): 0
   Prediksi (label): out
   Confidence: 85.43%

   Probabilitas detail:
      out: 85.43%
      in: 14.57%
```

### Contoh Input Dictionary

```python
test_data = {
    'HAEMATOCRIT': 35.1,      # nilai numerik (float/int)
    'ERYTHROCYTE': 4.65,      # nilai numerik
    'LEUCOCYTE': 6.3,         # nilai numerik
    'THROMBOCYTE': 310,       # nilai numerik
    'MCH': 25.4,              # nilai numerik
    'MCV': 75.5,              # nilai numerik
    'AGE': 1,                 # nilai numerik
    'SEX': 'F'                # string: 'M' atau 'F'
}
```

---

## API Endpoints

### Endpoint: `/api/predict-blood-test`

**Method**: POST

**Request Body**:
```json
{
  "input": {
    "HAEMATOCRIT": 35.1,
    "ERYTHROCYTE": 4.65,
    "LEUCOCYTE": 6.3,
    "THROMBOCYTE": 310,
    "MCH": 25.4,
    "MCV": 75.5,
    "AGE": 1,
    "SEX": "F"
  }
}
```

**Response (Success)**:
```json
{
  "success": true,
  "prediction": 0,
  "prediction_label": "out",
  "probability": [0.8543, 0.1457],
  "model_name": "XGBoost Patient Treatment"
}
```

**Response (Error)**:
```json
{
  "error": "Error message here",
  "success": false
}
```

### Contoh cURL Request

```bash
curl -X POST http://localhost:5000/api/predict-blood-test \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "HAEMATOCRIT": 35.1,
      "ERYTHROCYTE": 4.65,
      "LEUCOCYTE": 6.3,
      "THROMBOCYTE": 310,
      "MCH": 25.4,
      "MCV": 75.5,
      "AGE": 1,
      "SEX": "F"
    }
  }'
```

---

## Frontend Integration

### Halaman: `/blood-test`

Halaman ini telah diupdate dengan:

#### 1. **Form Input**
- Semua 8 fitur input dalam bahasa Indonesia
- Input validation (required, min, step)
- Select dropdown untuk SEX (M/F)

#### 2. **Model Information**
- Menampilkan informasi tentang model yang digunakan
- Dataset yang digunakan
- Target prediksi

#### 3. **Hasil Prediksi**
- Prediksi dengan label asli
- Confidence score
- Detail probabilitas untuk setiap kelas

#### 4. **Error Handling**
- Menampilkan pesan error jika ada
- UI feedback yang jelas

### Struktur Form

```html
<form id="bloodTestForm">
  <!-- Input fields untuk 8 fitur -->
  <input name="AGE" type="number">
  <select name="SEX">
    <option value="M">Laki-laki</option>
    <option value="F">Perempuan</option>
  </select>
  <input name="HAEMATOCRIT" type="number" step="0.1">
  <input name="ERYTHROCYTE" type="number" step="0.01">
  <input name="LEUCOCYTE" type="number" step="0.1">
  <input name="THROMBOCYTE" type="number" step="1">
  <input name="MCH" type="number" step="0.1">
  <input name="MCV" type="number" step="0.1">
</form>
```

---

## Checklist Setup

Pastikan semua item berikut sudah selesai:

- [ ] File `model-xgboost-patient-treatment.pkl` ada di `models/`
- [ ] File `label-encoder-patient-treatment.pkl` ada di `models/`
- [ ] File `feature-columns-patient-treatment.pkl` ada di `models/`
- [ ] `app.py` sudah diupdate dengan code baru
- [ ] `templates/blood_test.html` sudah diupdate
- [ ] Script `test-blood-prediction.py` sudah ada
- [ ] Flask app sudah direstart
- [ ] Akses halaman `/blood-test` berhasil
- [ ] Test prediksi dengan script berhasil
- [ ] Form input bekerja dan dapat melakukan prediksi

---

## Troubleshooting

### Error: Model tidak ditemukan

**Solusi**:
- Pastikan file `model-xgboost-patient-treatment.pkl` ada di folder `models/`
- Restart Flask app

### Error: Model patient-treatment tidak ditemukan

**Solusi**:
- Pastikan nama file model tepat: `model-xgboost-patient-treatment.pkl`
- Penamaan harus sesuai dengan pattern `*_model.pkl`

### Error saat encoding SEX

**Solusi**:
- Pastikan file `label-encoder-patient-treatment.pkl` ada
- Pastikan nilai SEX adalah 'M' atau 'F' (case-sensitive)
- Check bahwa LabelEncoder di file sesuai dengan training

### Prediksi tidak konsisten

**Solusi**:
- Pastikan file `feature-columns-patient-treatment.pkl` ada
- Urutan fitur harus tepat sesuai dengan training
- Check nilai input apakah dalam range yang benar

---

## Catatan Penting

1. **Urutan Fitur**: Sangat penting bahwa urutan fitur sesuai dengan training
2. **Encoding SEX**: Harus menggunakan label encoder yang sama dari training
3. **Decoding**: Hasil prediksi harus di-decode ke label asli
4. **Probabilitas**: Diambil untuk semua kelas, bukan hanya prediksi terbaik

---

## Referensi

- Dataset: `data-ori.csv`
- Script training: (jika ada)
- Model type: XGBoost Classifier
- Framework: Flask, scikit-learn

---

**Last Updated**: 2024
**Status**: ✅ Ready for Production
