#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk testing prediksi Blood Test (Patient Treatment Model)
Menggunakan input manual dengan dictionary untuk demonstrasi
"""

import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys

# Set UTF-8 encoding untuk output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Konfigurasi path
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'

# Nama model
MODEL_NAME = 'xgboost-patient-treatment'

# Load model
MODEL_FILE = MODELS_DIR / f'{MODEL_NAME}_model.pkl'
LABEL_ENCODER_FILE = MODELS_DIR / f'label-encoder-{MODEL_NAME}.pkl'
FEATURE_COLUMNS_FILE = MODELS_DIR / f'feature-columns-{MODEL_NAME}.pkl'

# Check alternate naming (model-*.pkl)
if not MODEL_FILE.exists():
    alt_model = MODELS_DIR / f'model-{MODEL_NAME}.pkl'
    if alt_model.exists():
        MODEL_FILE = alt_model

print("=" * 80)
print("[TEST] PREDIKSI BLOOD TEST - PATIENT TREATMENT MODEL")
print("=" * 80)

# Check if files exist
if not MODEL_FILE.exists():
    print(f"[ERROR] Model file tidak ditemukan: {MODEL_FILE}")
    exit(1)

# Load model
print(f"\n[LOAD] Loading model dari: {MODEL_FILE}")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print("[OK] Model loaded successfully")

# Load label encoder
label_encoder = None
if LABEL_ENCODER_FILE.exists():
    print(f"[LOAD] Loading label encoder dari: {LABEL_ENCODER_FILE}")
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    print("[OK] Label encoder loaded successfully")
    print(f"   Classes: {label_encoder.classes_}")
else:
    print(f"[WARN] Label encoder file tidak ditemukan: {LABEL_ENCODER_FILE}")

# Load feature columns
feature_columns = None
if FEATURE_COLUMNS_FILE.exists():
    print(f"[LOAD] Loading feature columns dari: {FEATURE_COLUMNS_FILE}")
    with open(FEATURE_COLUMNS_FILE, 'rb') as f:
        feature_columns = pickle.load(f)
    print("[OK] Feature columns loaded successfully")
    print(f"   Columns: {feature_columns}")
else:
    print(f"[WARN] Feature columns file tidak ditemukan: {FEATURE_COLUMNS_FILE}")

# Contoh data untuk testing
print("\n" + "=" * 80)
print("[DATA] CONTOH DATA TESTING")
print("=" * 80)

test_samples = [
    {
        'name': 'Sample 1 - Data Normal',
        'data': {
            'HAEMATOCRIT': 35.1,
            'ERYTHROCYTE': 4.65,
            'LEUCOCYTE': 6.3,
            'THROMBOCYTE': 310,
            'MCH': 25.4,
            'MCV': 75.5,
            'AGE': 1,
            'SEX': 'F'
        }
    },
    {
        'name': 'Sample 2 - Data dengan Variasi',
        'data': {
            'HAEMATOCRIT': 43.5,
            'ERYTHROCYTE': 5.39,
            'LEUCOCYTE': 12.7,
            'THROMBOCYTE': 334,
            'MCH': 27.5,
            'MCV': 80.7,
            'AGE': 1,
            'SEX': 'F'
        }
    },
    {
        'name': 'Sample 3 - Data Laki-laki',
        'data': {
            'HAEMATOCRIT': 39.1,
            'ERYTHROCYTE': 4.98,
            'LEUCOCYTE': 10.5,
            'THROMBOCYTE': 366,
            'MCH': 27.5,
            'MCV': 78.5,
            'AGE': 25,
            'SEX': 'M'
        }
    },
    {
        'name': 'Sample 4 - Data Variasi Usia',
        'data': {
            'HAEMATOCRIT': 33.5,
            'ERYTHROCYTE': 4.74,
            'LEUCOCYTE': 13.2,
            'THROMBOCYTE': 305,
            'MCH': 23.8,
            'MCV': 70.7,
            'AGE': 45,
            'SEX': 'M'
        }
    },
]

def predict_sample(input_dict):
    """
    Fungsi untuk melakukan prediksi pada sample data
    
    Parameters:
    -----------
    input_dict : dict
        Dictionary berisi fitur-fitur untuk prediksi
        
    Returns:
    --------
    dict : Hasil prediksi
    """
    try:
        # Copy data untuk tidak mengubah original
        data = input_dict.copy()
        
        # Encode SEX jika diperlukan
        if 'SEX' in data and label_encoder is not None:
            try:
                sex_encoded = label_encoder.transform([data['SEX']])[0]
                data['SEX'] = sex_encoded
            except Exception as e:
        print(f"   [ERROR] Error encoding SEX: {e}")
        
        # Reorder features sesuai feature_columns jika ada
        if feature_columns is not None:
            ordered_features = [data.get(col, 0) for col in feature_columns]
        else:
            # Jika tidak ada feature_columns, gunakan urutan default
            default_order = ['HAEMATOCRIT', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCV', 'AGE', 'SEX']
            ordered_features = [data.get(col, 0) for col in default_order]
        
        # Convert ke numpy array
        X = np.array([ordered_features], dtype=float)
        
        # Prediksi
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Decode prediction jika ada label encoder
        prediction_label = prediction
        if label_encoder is not None:
            try:
                prediction_label = label_encoder.inverse_transform([int(prediction)])[0]
            except Exception as e:
                prediction_label = str(prediction)
        
        return {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': str(prediction_label),
            'probabilities': probabilities,
            'max_probability': float(np.max(probabilities))
        }
    
    except Exception as e:
        print(f\"   [ERROR] Error: {e}\")\n        return None
        return None

# Jalankan prediksi untuk setiap sample
print("\n" + "=" * 80)
print("[RESULT] HASIL PREDIKSI")
print("=" * 80)

for idx, sample in enumerate(test_samples, 1):
    print(f"\n[Sample {idx}] {sample['name']}")
    print("-" * 80)
    
    # Tampilkan input data
    print("[INPUT] Input Data:")
    for key, value in sample['data'].items():
        print(f"   {key}: {value}")
    
    # Lakukan prediksi
    result = predict_sample(sample['data'])
    
    if result and result['success']:
        print("\n[PREDICTION] Hasil Prediksi:")
        print(f"   Prediksi (angka): {result['prediction']}")
        print(f"   Prediksi (label): {result['prediction_label']}")
        print(f"   Confidence: {result['max_probability'] * 100:.2f}%")
        print(f"\n   Probabilitas detail:")
        for class_idx, prob in enumerate(result['probabilities']):
            # Coba dekode class label jika ada label encoder
            if label_encoder is not None:
                try:
                    class_label = label_encoder.inverse_transform([class_idx])[0]
                except:
                    class_label = f"Class {class_idx}"
            else:
                class_label = f"Class {class_idx}"
            print(f"      {class_label}: {prob * 100:.2f}%")
    else:
        print("\n[FAILED] Prediksi gagal")

print("\n" + "=" * 80)
print("[DONE] Testing selesai")
print("=" * 80)

# Informasi Model
print("\n[INFO] INFORMASI MODEL")
print("-" * 80)
print(f"Model Name: {MODEL_NAME}")
print(f"Model Type: {type(model).__name__}")
print(f"Feature Columns: {feature_columns if feature_columns else 'Not found'}")
print(f"Label Encoder Classes: {label_encoder.classes_.tolist() if label_encoder else 'Not found'}")
print("\n")
