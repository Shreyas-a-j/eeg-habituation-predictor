"""
Unit Tests for EEG Habituation Predictor
Run with: pytest tests/test_suite.py -v
"""

import sys
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, 'src')

from ehp.feature_extractor import EEGFeatureExtractor
from ehp.classifier import HabitationPredictor
from ehp.evaluator import EvaluationMetrics

# ============================================================================
# FIXTURES (Reusable test data)
# ============================================================================

@pytest.fixture
def synthetic_sine_signal():
    """10 Hz sine wave (alpha band)"""
    fs = 256
    duration = 60
    t = np.linspace(0, duration, fs * duration)
    signal = np.sin(2 * np.pi * 10 * t)
    return signal

@pytest.fixture
def white_noise_signal():
    """Random white noise signal"""
    fs = 256
    duration = 60
    n_samples = fs * duration
    signal = np.random.randn(n_samples)
    return signal

@pytest.fixture
def sample_features():
    """Sample feature matrix and labels"""
    np.random.seed(42)
    n_samples = 50
    n_features = 7
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.3, n_samples)
    return X, y, ['theta', 'alpha', 'beta', 'theta_alpha_ratio', 'entropy', 'std', 'complexity']

# ============================================================================
# FEATURE EXTRACTOR TESTS
# ============================================================================

class TestFeatureExtractor:
    """Test suite for EEGFeatureExtractor"""
    
    def test_extract_features_sine_wave_alpha_dominance(self, synthetic_sine_signal):
        """Test that 10 Hz sine wave has high alpha power"""
        extractor = EEGFeatureExtractor(sampling_rate=256)
        features = extractor.extract_features(synthetic_sine_signal)
        
        # Alpha should dominate for 10 Hz signal
        assert features['alpha_power'] > features['theta_power'], \
            "10 Hz signal should have higher alpha than theta"
        assert features['alpha_power'] > features['beta_power'], \
            "10 Hz signal should have higher alpha than beta"
    
    def test_extract_features_no_nan_values(self, white_noise_signal):
        """Test that no NaN or inf values are produced"""
        extractor = EEGFeatureExtractor(sampling_rate=256)
        features = extractor.extract_features(white_noise_signal)
        
        for key, val in features.items():
            assert np.isfinite(val), f"{key} = {val} is not finite"
            assert not np.isnan(val), f"{key} is NaN"
    
    def test_extract_features_output_shape(self, white_noise_signal):
        """Test that output dict has correct number of features"""
        extractor = EEGFeatureExtractor(sampling_rate=256)
        features = extractor.extract_features(white_noise_signal)
        
        assert len(features) == 7, f"Expected 7 features, got {len(features)}"
        
        expected_keys = {'theta_power', 'alpha_power', 'beta_power', 'theta_alpha_ratio',
                        'sample_entropy', 'signal_std', 'hjorth_complexity'}
        assert set(features.keys()) == expected_keys
    
    def test_batch_extract_features_shape(self, white_noise_signal):
        """Test batch extraction produces correct shape"""
        extractor = EEGFeatureExtractor(sampling_rate=256)
        signals = [white_noise_signal for _ in range(5)]
        
        df = extractor.batch_extract_features(signals)
        
        assert df.shape[0] == 5, f"Expected 5 rows, got {df.shape[0]}"
        assert df.shape[1] == 7, f"Expected 7 columns, got {df.shape[1]}"
    
    def test_batch_extract_returns_dataframe(self, white_noise_signal):
        """Test that batch extraction returns DataFrame"""
        extractor = EEGFeatureExtractor(sampling_rate=256)
        signals = [white_noise_signal for _ in range(3)]
        
        result = extractor.batch_extract_features(signals)
        
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

# ============================================================================
# CLASSIFIER TESTS
# ============================================================================

class TestHabitationPredictor:
    """Test suite for HabitationPredictor"""
    
    def test_init_creates_four_classifiers(self):
        """Test that predictor initializes with 4 classifiers"""
        predictor = HabitationPredictor()
        
        assert len(predictor.classifiers) == 4
        assert 'Random Forest' in predictor.classifiers
        assert 'Gradient Boosting' in predictor.classifiers
        assert 'SVM' in predictor.classifiers
        assert 'Logistic Regression' in predictor.classifiers
    
    def test_train_without_error(self, sample_features):
        """Test that training completes without error"""
        X, y, _ = sample_features
        predictor = HabitationPredictor()
        
        results = predictor.train(X, y, test_size=0.2)
        
        assert results is not None
        assert len(results) == 4  # 4 classifiers
        assert predictor.best_clf is not None
    
    def test_predict_output_shape(self, sample_features):
        """Test that predictions have correct shape"""
        X, y, _ = sample_features
        predictor = HabitationPredictor()
        predictor.train(X, y, test_size=0.2)
        
        predictions = predictor.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_cross_validate_returns_dataframe(self, sample_features):
        """Test that cross-validation returns DataFrame"""
        X, y, _ = sample_features
        predictor = HabitationPredictor()
        
        cv_results = predictor.cross_validate(X, y, cv=3)
        
        assert isinstance(cv_results, pd.DataFrame)
        assert cv_results.shape[0] == 4  # 4 classifiers
        assert 'F1_mean' in cv_results.columns
        assert 'AUC_mean' in cv_results.columns
    
    def test_get_best_classifier(self, sample_features):
        """Test getting best classifier"""
        X, y, _ = sample_features
        predictor = HabitationPredictor()
        predictor.train(X, y)
        
        name, clf = predictor.get_best_classifier()
        
        assert name is not None
        assert clf is not None
    
    def test_predict_with_confidence(self, sample_features):
        """Test predictions with confidence scores"""
        X, y, _ = sample_features
        predictor = HabitationPredictor()
        predictor.train(X, y)
        
        preds, probs, high_conf = predictor.predict_with_confidence(X[:10], threshold=0.7)
        
        assert len(preds) == 10
        assert len(probs) == 10
        assert len(high_conf) == 10
        assert all(p >= 0 and p <= 1 for p in probs)

# ============================================================================
# EVALUATION TESTS
# ============================================================================

class TestEvaluationMetrics:
    """Test suite for EvaluationMetrics"""
    
    def test_compute_metrics_structure(self):
        """Test that metrics dict has all required keys"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        
        evaluator = EvaluationMetrics(y_true, y_pred)
        metrics = evaluator.compute_metrics()
        
        required_keys = {'accuracy', 'precision', 'recall', 'f1', 'specificity', 'sensitivity'}
        assert set(metrics.keys()) >= required_keys
    
    def test_perfect_predictions_all_ones(self):
        """Test that perfect predictions yield metric = 1.0"""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = y_true.copy()
        
        evaluator = EvaluationMetrics(y_true, y_pred)
        metrics = evaluator.compute_metrics()
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
    
    def test_random_predictions_around_chance(self):
        """Test that random predictions are near chance level"""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.binomial(1, 0.5, 100)
        
        evaluator = EvaluationMetrics(y_true, y_pred)
        metrics = evaluator.compute_metrics()
        
        # Random predictions should be close to 0.5
        assert 0.3 < metrics['accuracy'] < 0.7

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for end-to-end pipeline"""
    
    def test_feature_extraction_to_prediction(self, white_noise_signal):
        """Test full pipeline: extract features -> train -> predict"""
        
        # Generate multiple signals
        extractor = EEGFeatureExtractor(sampling_rate=256)
        signals = [white_noise_signal + 0.1 * np.random.randn(*white_noise_signal.shape) 
                   for _ in range(20)]
        
        # Extract features
        features_df = extractor.batch_extract_features(signals)
        assert features_df.shape == (20, 7)
        
        # Create labels and train
        X = features_df.values
        y = np.random.binomial(1, 0.3, 20)
        
        predictor = HabitationPredictor()
        predictor.train(X, y)
        
        # Make predictions
        predictions = predictor.predict(X[:5])
        assert len(predictions) == 5

# ============================================================================
# PYTEST EXECUTION
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])