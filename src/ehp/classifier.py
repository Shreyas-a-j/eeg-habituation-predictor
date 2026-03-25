"""
Machine Learning Classifier Module
Trains and evaluates models for habituation prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class HabitationPredictor:
    """Multi-classifier ensemble for predicting EEG habituation"""
    
    def __init__(self):
        """Initialize with 4 different classifiers"""
        self.classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        }
        self.best_clf = None
        self.best_name = None
        self.scaler = StandardScaler()
        logger.info("Initialized HabitationPredictor with 4 classifiers")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train all classifiers and identify best performer
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Labels (n_samples,)
        test_size : float
            Fraction for test set
            
        Returns:
        --------
        dict : Training results for each classifier
        """
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifiers
        results = {}
        best_score = -1
        
        for name, clf in self.classifiers.items():
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = clf.score(X_train_scaled, y_train)
            test_score = clf.score(X_test_scaled, y_test)
            
            results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'model': clf,
            }
            
            logger.info(f"{name}: Train={train_score:.3f}, Test={test_score:.3f}")
            
            if test_score > best_score:
                best_score = test_score
                self.best_clf = clf
                self.best_name = name
        
        logger.info(f"Best classifier: {self.best_name} (Accuracy={best_score:.3f})")
        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> pd.DataFrame:
        """
        5-fold cross-validation for all classifiers
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        cv : int
            Number of folds
            
        Returns:
        --------
        pd.DataFrame : Cross-validation results
        """
        
        X_scaled = self.scaler.fit_transform(X)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_results = []
        
        for name, clf in self.classifiers.items():
            f1_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='f1_weighted')
            auc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='roc_auc_ovr')
            acc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring='accuracy')
            
            cv_results.append({
                'Classifier': name,
                'F1_mean': f1_scores.mean(),
                'F1_std': f1_scores.std(),
                'AUC_mean': auc_scores.mean(),
                'AUC_std': auc_scores.std(),
                'Accuracy_mean': acc_scores.mean(),
                'Accuracy_std': acc_scores.std(),
            })
            
            logger.info(f"{name}: F1={f1_scores.mean():.3f}±{f1_scores.std():.3f}")
        
        return pd.DataFrame(cv_results)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using best classifier
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : Predictions (0 or 1)
        """
        if self.best_clf is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.best_clf.predict(X_scaled)
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray, threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        threshold : float
            Confidence threshold (0-1)
            
        Returns:
        --------
        predictions : np.ndarray
        probabilities : np.ndarray
        high_confidence_mask : np.ndarray (boolean)
        """
        if self.best_clf is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.best_clf.predict(X_scaled)
        probabilities = self.best_clf.predict_proba(X_scaled).max(axis=1)
        
        high_confidence_mask = probabilities >= threshold
        
        return predictions, probabilities, high_confidence_mask
    
    def get_best_classifier(self) -> Tuple[str, object]:
        """Return name and instance of best classifier"""
        if self.best_clf is None:
            raise ValueError("No classifier trained yet")
        return self.best_name, self.best_clf
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance from best classifier (if tree-based)
        
        Parameters:
        -----------
        feature_names : list
            Names of features
            
        Returns:
        --------
        pd.DataFrame : Feature importance scores
        """
        if not hasattr(self.best_clf, 'feature_importances_'):
            raise ValueError(f"{self.best_name} does not support feature importance")
        
        importances = self.best_clf.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
        }).sort_values('Importance', ascending=False)
        
        return df