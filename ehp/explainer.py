"""
SHAP Explainability Module
Generates explanations for model predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class ExplainabilityAnalyzer:
    """SHAP-based explainability for model predictions"""
    
    def __init__(self, trained_classifier, feature_names: list):
        """
        Initialize explainer
        
        Parameters:
        -----------
        trained_classifier : sklearn classifier
            Trained classifier with predict_proba method
        feature_names : list
            Names of features
        """
        self.clf = trained_classifier
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        logger.info("Initialized ExplainabilityAnalyzer")
    
    def explain_predictions(self, X_test: np.ndarray) -> Tuple:
        """
        Generate SHAP values for test set
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test feature matrix
            
        Returns:
        --------
        explainer, shap_values : SHAP objects
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP required. Install: pip install shap")
        
        # Create explainer based on classifier type
        if hasattr(self.clf, 'feature_importances_'):
            # Tree-based model
            self.explainer = shap.TreeExplainer(self.clf)
        else:
            # Linear model
            self.explainer = shap.LinearExplainer(self.clf, X_test[:100])
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_test)
        
        logger.info(f"Generated SHAP values for {X_test.shape[0]} samples")
        return self.explainer, self.shap_values
    
    def plot_summary(self, shap_values, X_test: np.ndarray, save_path: str = None):
        """Create SHAP summary plot"""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP required")
        
        plt.figure(figsize=(10, 6))
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]  # Use class 1 (habituators)
        else:
            shap_values_to_plot = shap_values
        
        shap.summary_plot(shap_values_to_plot, X_test, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot: {save_path}")
        
        plt.close()
    
    def plot_dependence(self, shap_values, X_test: np.ndarray, feature_name: str, save_path: str = None):
        """Create SHAP dependence plot for a specific feature"""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP required")
        
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature_idx = self.feature_names.index(feature_name)
        
        plt.figure(figsize=(8, 6))
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
        
        shap.dependence_plot(feature_idx, shap_values_to_plot, X_test, 
                            feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved SHAP dependence plot: {save_path}")
        
        plt.close()
    
    def get_feature_importance(self, shap_values) -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values
        
        Parameters:
        -----------
        shap_values : np.ndarray
            SHAP values matrix
            
        Returns:
        --------
        pd.DataFrame : Feature importance scores
        """
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': feature_importance,
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        df['Importance_Rank'] = range(1, len(df) + 1)
        
        return df
    
    def interpret_prediction(self, sample_idx: int, shap_values, X_test: np.ndarray, 
                           prediction: int, probability: float) -> str:
        """
        Generate English interpretation of prediction
        
        Parameters:
        -----------
        sample_idx : int
            Index of sample
        shap_values : np.ndarray
            SHAP values
        X_test : np.ndarray
            Test feature matrix
        prediction : int
            Predicted class (0 or 1)
        probability : float
            Prediction probability
            
        Returns:
        --------
        str : Interpretation text
        """
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_vals = shap_values[prediction]
        else:
            shap_vals = shap_values
        
        # Get top contributing features
        sample_shap = np.abs(shap_vals[sample_idx])
        top_indices = np.argsort(sample_shap)[-3:][::-1]
        
        prediction_text = "Habituator" if prediction == 1 else "Non-Habituator"
        
        interpretation = f"Prediction: {prediction_text} (confidence: {probability:.2%})\n"
        interpretation += "Top contributing features:\n"
        
        for rank, idx in enumerate(top_indices, 1):
            feature_name = self.feature_names[idx]
            feature_value = X_test[sample_idx, idx]
            shap_value = shap_vals[sample_idx, idx]
            
            direction = "increases" if shap_value > 0 else "decreases"
            interpretation += f"  {rank}. {feature_name}={feature_value:.3f} ({direction} prediction)\n"
        
        return interpretation