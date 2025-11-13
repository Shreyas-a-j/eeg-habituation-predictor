"""
EEG Habituation Predictor Package
ML-based prediction of EEG habituation to binaural beats in Parkinson's Disease
"""

__version__ = "0.1.0"
__author__ = "Shreyas Jadhav"
__email__ = "airjadhav111@gmail.com"

from .feature_extractor import EEGFeatureExtractor
from .data_loader import load_eeg_dataset, preprocess_signal
from .classifier import HabitationPredictor
from .explainer import ExplainabilityAnalyzer
from .evaluator import EvaluationMetrics

__all__ = [
    "EEGFeatureExtractor",
    "load_eeg_dataset",
    "preprocess_signal",
    "HabitationPredictor",
    "ExplainabilityAnalyzer",
    "EvaluationMetrics",
]