# Setup Guide - ML Model Prediction Web Application

Panduan lengkap untuk setup aplikasi Flask dengan ML Models.

## 📋 Prerequisite

- Python 3.8+
- pip (Python Package Manager)
- Git (opsional)

## 🔧 Step-by-Step Setup

### Step 1: Persiapan Environment

```bash
# Navigasi ke folder project
cd /home/luthfi/www/mlearning

# Buat virtual environment
python -m venv venv

# Activate virtual environment
# Untuk Linux/Mac:
source venv/bin/activate

# Untuk Windows:
# venv\Scripts\activate
```

### Step 2: Training Models

```bash
# Pastikan sudah ada clean datasets di folder tubes/clean/
# - diabetes_clean.csv
# - data-ori_clean.csv
# - Sleep_clean.csv

# Jalankan training script
python tubes/tubes.py

# Output akan tersimpan di tubes/outputs/
```

### Step 3: Copy Models ke Flask App

```bash
# Dari folder project root
cp -r tubes/outputs/models flask-app/

# Verify models sudah tercopy
ls flask-app/models/
# Output harusnya:
# diabetes_metadata.json
# diabetes_model.pkl
# blood_test_metadata.json
# blood_test_model.pkl
# sleep_disorder_metadata.json
# sleep_disorder_model.pkl
```

### Step 4: Install Dependencies

```bash
# Navigasi ke flask-app folder
cd flask-app

# Install requirements
pip install -r requirements.txt
```

### Step 5: Jalankan Aplikasi

```bash
# Jalankan Flask app
python app.py

# Output harusnya:
# 🚀 Starting Flask App...
# 📦 Models loaded: 3
#    ✅ Diabetes
#    ✅ Blood Test
#    ✅ Sleep Disorder
# * Running on http://localhost:5000
```

### Step 6: Akses Aplikasi

Buka browser dan kunjungi:
- Dashboard: http://localhost:5000/
- Diabetes: http://localhost:5000/diabetes
- Blood Test: http://localhost:5000/blood-test
- Sleep Disorder: http://localhost:5000/sleep-disorder
- Test Model: http://localhost:5000/test-model
- About: http://localhost:5000/about

## 🧪 Running Tests

```bash
# Dari folder flask-app
python -m pytest tests/ -v

# Atau
python tests/test_app.py
```

## 📝 Troubleshooting

### Models tidak terdeteksi

**Problem:** "Models loaded: 0"

**Solution:**
1. Pastikan models folder ada di `flask-app/models/`
2. Pastikan file `.pkl` dan `.json` ada dengan nama yang benar
3. Run tubes.py untuk generate models baru

```bash
python tubes/tubes.py
cp -r tubes/outputs/models/* flask-app/models/
```

### Import Error

**Problem:** "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
# Pastikan virtual environment sudah aktif
source venv/bin/activate

# Install requirements lagi
pip install -r requirements.txt
```

### Port 5000 sudah digunakan

**Problem:** "Address already in use"

**Solution:**
1. Ubah port di `app.py`:
```python
app.run(debug=True, host='localhost', port=5001)  # Ubah 5000 ke 5001
```

2. Atau kill process yang menggunakan port:
```bash
# Linux/Mac
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Template tidak ditemukan

**Problem:** "jinja2.exceptions.TemplateNotFound"

**Solution:**
```bash
# Pastikan templates folder ada di:
# flask-app/templates/
# dan file HTML sudah ada
ls flask-app/templates/
```

## 📁 Project Structure Check

```
flask-app/
├── ✅ app.py
├── ✅ config.py
├── ✅ requirements.txt
├── ✅ README.md
├── ✅ SETUP.md
├── ✅ .gitignore
├── models/
│   ├── ✅ diabetes_model.pkl
│   ├── ✅ diabetes_metadata.json
│   ├── ✅ blood_test_model.pkl
│   ├── ✅ blood_test_metadata.json
│   ├── ✅ sleep_disorder_model.pkl
│   └── ✅ sleep_disorder_metadata.json
├── templates/
│   ├── ✅ index.html
│   ├── ✅ diabetes.html
│   ├── ✅ blood_test.html
│   ├── ✅ sleep_disorder.html
│   ├── ✅ test_model.html
│   ├── ✅ about.html
│   └── ✅ 404.html
├── static/
│   ├── css/
│   │   └── ✅ style.css
│   └── js/
├── tests/
│   ├── ✅ __init__.py
│   └── ✅ test_app.py
└── data/
    └── ✅ sample_test_data.json
```

## 🚀 Production Deployment

Untuk deployment ke production:

1. **Update config:**
```python
# Di app.py
app.config.from_object('config.ProductionConfig')
```

2. **Gunakan WSGI server (gunicorn):**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Setup reverse proxy (nginx/Apache)**

4. **Enable HTTPS**

## 📚 Dokumentasi Tambahan

- Flask Documentation: https://flask.palletsprojects.com/
- scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/

## ✅ Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created dan activated
- [ ] Models trained dan di-copy ke flask-app/models
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Flask app running (`python app.py`)
- [ ] Akses http://localhost:5000
- [ ] Test semua 5 halaman utama
- [ ] Run tests (`python tests/test_app.py`)

## 📞 Support

Jika ada masalah, cek:
1. Console output untuk error messages
2. Browser console (F12) untuk JavaScript errors
3. File logs jika ada

---

**Selamat! Aplikasi sudah siap digunakan.** 🎉
