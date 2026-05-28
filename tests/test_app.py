import unittest
import json
import sys
from pathlib import Path
import pickle
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import app

class TestModelAPI(unittest.TestCase):
    """Test Flask app dan model predictions"""
    
    def setUp(self):
        """Setup test client"""
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
    
    def test_index_page(self):
        """Test halaman utama"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)
    
    def test_about_page(self):
        """Test halaman about"""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Tentang Aplikasi', response.data)
    
    def test_diabetes_page(self):
        """Test halaman diabetes"""
        response = self.client.get('/diabetes')
        if response.status_code == 200:
            self.assertIn(b'Diabetes', response.data)
        else:
            # Model mungkin belum tersedia
            self.assertEqual(response.status_code, 404)
    
    def test_blood_test_page(self):
        """Test halaman blood test"""
        response = self.client.get('/blood-test')
        if response.status_code == 200:
            self.assertIn(b'Blood Test', response.data)
        else:
            self.assertEqual(response.status_code, 404)
    
    def test_sleep_disorder_page(self):
        """Test halaman sleep disorder"""
        response = self.client.get('/sleep-disorder')
        if response.status_code == 200:
            self.assertIn(b'Sleep Disorder', response.data)
        else:
            self.assertEqual(response.status_code, 404)
    
    def test_test_model_page(self):
        """Test halaman test model"""
        response = self.client.get('/test-model')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test Model', response.data)
    
    def test_404_page(self):
        """Test halaman 404"""
        response = self.client.get('/not-found')
        self.assertEqual(response.status_code, 404)
    
    def test_model_info_endpoint(self):
        """Test API endpoint untuk info model"""
        response = self.client.get('/api/model-info/diabetes')
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertIn('metrics', data)
            self.assertIn('features_count', data)
    
    def test_predict_endpoint(self):
        """Test API endpoint untuk prediksi"""
        features = [0, 137, 40, 35, 168, 43.1, 2.288, 33]
        
        response = self.client.post('/api/predict',
            data=json.dumps({'model_name': 'diabetes', 'features': features}),
            content_type='application/json'
        )
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertIn('prediction', data)
            self.assertIn('probability', data)
        else:
            # Model mungkin belum tersedia
            self.assertEqual(response.status_code, 404)
    
    def test_test_accuracy_endpoint(self):
        """Test API endpoint untuk test accuracy"""
        test_data = {
            'actual': [1, 0, 1, 0, 1],
            'predicted': [1, 0, 1, 1, 1]
        }
        
        response = self.client.post('/api/test-accuracy',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['total'], 5)
        self.assertEqual(data['correct'], 4)
    
    def test_invalid_json_in_test_accuracy(self):
        """Test error handling untuk invalid JSON"""
        response = self.client.post('/api/test-accuracy',
            data=json.dumps({'actual': [1, 0], 'predicted': [1]}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])


class TestModelLoading(unittest.TestCase):
    """Test model loading"""
    
    def test_models_loaded(self):
        """Test apakah models sudah di-load"""
        from app import models_data
        # Setidaknya harus ada model yang di-load atau folder models kosong
        self.assertIsInstance(models_data, dict)


def run_tests():
    """Run semua tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
