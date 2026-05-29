from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load models dan metadata
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'

models_data = {}
label_encoders = {}
feature_columns = {}

def load_models():
    """Load all models dan metadata"""
    # Pattern 1: *_model.pkl (e.g., blood_test_model.pkl)
    for model_file in MODELS_DIR.glob('*_model.pkl'):
        model_name = model_file.stem.replace('_model', '')
        _load_model_with_metadata(model_name, model_file)
    
    # Pattern 2: model-*.pkl (e.g., model-xgboost-patient-treatment.pkl)
    for model_file in MODELS_DIR.glob('model-*.pkl'):
        model_name = model_file.stem.replace('model-', '')
        _load_model_with_metadata(model_name, model_file)

def _load_model_with_metadata(model_name, model_file):
    """Helper function to load model with metadata, encoder, and feature columns"""
    try:
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata (try both naming patterns)
        metadata_file = MODELS_DIR / f'{model_name}_metadata.json'
        if not metadata_file.exists():
            # Try alternative pattern
            metadata_file = MODELS_DIR / f'{model_name}-metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            models_data[model_name] = {
                'model': model,
                'metadata': metadata,
                'name': metadata.get('model_name', model_name)
            }
        else:
            # If no metadata, still load model with empty metadata
            models_data[model_name] = {
                'model': model,
                'metadata': {'model_name': model_name},
                'name': model_name
            }
        
        # Load label encoder if available (try both naming patterns)
        label_encoder_file = MODELS_DIR / f'label-encoder-{model_name}.pkl'
        if label_encoder_file.exists():
            with open(label_encoder_file, 'rb') as f:
                label_encoders[model_name] = pickle.load(f)
        
        # Load feature columns if available (try both naming patterns)
        feature_cols_file = MODELS_DIR / f'feature-columns-{model_name}.pkl'
        if feature_cols_file.exists():
            with open(feature_cols_file, 'rb') as f:
                feature_columns[model_name] = pickle.load(f)
    
    except Exception as e:
        print(f"⚠️  Error loading model {model_name}: {e}")

# Load models at startup
load_models()


@app.route('/')
def index():
    """Halaman utama - Dashboard"""
    models_info = []
    for key, data in models_data.items():
        metrics = data['metadata'].get('metrics', {})
        models_info.append({
            'id': key,
            'name': data['name'],
            'accuracy': f"{metrics.get('accuracy', 0)*100:.2f}%",
            'precision': f"{metrics.get('precision', 0)*100:.2f}%",
            'recall': f"{metrics.get('recall', 0)*100:.2f}%",
            'f1_score': f"{metrics.get('f1_score', 0)*100:.2f}%"
        })
    return render_template('index.html', models=models_info)


@app.route('/about')
def about():
    """Halaman tentang aplikasi"""
    return render_template('about.html')


@app.route('/diabetes')
def diabetes():
    """Halaman prediksi Diabetes"""
    if 'diabetes' not in models_data:
        return "Model Diabetes tidak ditemukan", 404
    return render_template('diabetes.html', model_name='Diabetes')


@app.route('/blood-test')
def blood_test():
    """Halaman prediksi Blood Test"""
    if 'blood_test' not in models_data:
        return "Model Blood Test tidak ditemukan", 404
    return render_template('blood_test.html', model_name='Blood Test')


@app.route('/sleep-disorder')
def sleep_disorder():
    """Halaman prediksi Sleep Disorder"""
    if 'sleep_disorder' not in models_data:
        return "Model Sleep Disorder tidak ditemukan", 404
    return render_template('sleep_disorder.html', model_name='Sleep Disorder')


@app.route('/test-model')
def test_model_page():
    """Halaman untuk test model"""
    return render_template('test_model.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API untuk prediksi"""
    try:
        data = request.json
        model_name = data.get('model_name')
        features = data.get('features', [])
        
        if model_name not in models_data:
            return jsonify({'error': 'Model tidak ditemukan'}), 404
        
        model = models_data[model_name]['model']
        feature_names = models_data[model_name]['metadata']['feature_names']
        
        # Konversi features ke numpy array
        X = np.array([features], dtype=float)
        
        # Prediksi
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': probability.tolist(),
            'model_name': models_data[model_name]['name']
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400


@app.route('/api/predict-blood-test', methods=['POST'])
def predict_blood_test():
    """API untuk prediksi blood test dengan encoding dan decoding"""
    try:
        data = request.json
        model_name = 'xgboost-patient-treatment'
        
        if model_name not in models_data:
            return jsonify({'error': f'Model {model_name} tidak ditemukan. Available models: {list(models_data.keys())}'}), 404
        
        # Ambil data input
        input_data = data.get('input', {})
        
        # Encode SEX jika ada
        if 'SEX' in input_data and model_name in label_encoders:
            try:
                sex_encoded = label_encoders[model_name].transform([input_data['SEX']])[0]
                input_data['SEX'] = sex_encoded
            except Exception as e:
                print(f"Note: Could not encode SEX: {e}")
                # Fallback: encode manual jika gagal
                sex_map = {'M': 1, 'F': 0}
                input_data['SEX'] = sex_map.get(input_data['SEX'], 0)
        elif 'SEX' in input_data:
            # Jika tidak ada label encoder, gunakan fallback manual encoding
            sex_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0}
            input_data['SEX'] = sex_map.get(input_data['SEX'], 0)
        
        # Reorder features sesuai feature_columns jika ada
        if model_name in feature_columns:
            feature_cols = feature_columns[model_name]
            ordered_features = [input_data.get(col, 0) for col in feature_cols]
        else:
            # Fallback: gunakan order default
            default_order = ['HAEMATOCRIT', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCV', 'AGE', 'SEX']
            ordered_features = [input_data.get(col, 0) for col in default_order]
        
        # Convert to numpy array
        X = np.array([ordered_features], dtype=float)
        
        # Prediksi
        model = models_data[model_name]['model']
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Decode prediction jika ada label encoder
        prediction_label = prediction
        if model_name in label_encoders:
            try:
                prediction_label = label_encoders[model_name].inverse_transform([int(prediction)])[0]
            except Exception as e:
                print(f"Note: Could not decode prediction: {e}")
                prediction_label = str(prediction)
        else:
            # Fallback: gunakan mapping manual jika tidak ada encoder
            # (Asumsi binary classification dengan 0 dan 1)
            label_map = {0: 'out', 1: 'in'}  # Adjust sesuai dengan actual labels
            prediction_label = label_map.get(int(prediction), str(prediction))
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'prediction_label': str(prediction_label),
            'probability': probability.tolist(),
            'model_name': models_data[model_name]['name']
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg, 'success': False}), 400


@app.route('/api/model-info/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """Get informasi model"""
    if model_name not in models_data:
        return jsonify({'error': 'Model tidak ditemukan'}), 404
    
    data = models_data[model_name]
    metrics = data['metadata'].get('metrics', {})
    feature_names = data['metadata'].get('feature_names', [])
    
    return jsonify({
        'success': True,
        'name': data['name'],
        'metrics': metrics,
        'features_count': len(feature_names),
        'feature_names': feature_names
    })


@app.route('/api/test-accuracy', methods=['POST'])
def test_accuracy():
    """Test model accuracy"""
    try:
        data = request.json
        model_name = data.get('model_name')
        actual = data.get('actual')
        predicted = data.get('predicted')
        
        if not actual or not predicted or len(actual) != len(predicted):
            return jsonify({'error': 'Data tidak valid'}), 400
        
        # Calculate accuracy
        correct = sum(1 for a, p in zip(actual, predicted) if a == p)
        accuracy = correct / len(actual) * 100
        
        return jsonify({
            'success': True,
            'total': len(actual),
            'correct': correct,
            'accuracy': f"{accuracy:.2f}%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    print("🚀 Starting Flask App...")
    print(f"📦 Models loaded: {len(models_data)}")
    for key, data in models_data.items():
        print(f"   ✅ {data['name']}")
    app.run(debug=True, host='localhost', port=5000)
