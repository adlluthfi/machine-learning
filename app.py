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

def load_models():
    """Load all models dan metadata"""
    for model_file in MODELS_DIR.glob('*_model.pkl'):
        model_name = model_file.stem.replace('_model', '')
        
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_file = MODELS_DIR / f'{model_name}_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            models_data[model_name] = {
                'model': model,
                'metadata': metadata,
                'name': metadata.get('model_name', model_name)
            }

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
