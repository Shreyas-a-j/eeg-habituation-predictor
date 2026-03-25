"""
Machine Learning Classifier Module
Trains and evaluates models for EEG habituation prediction
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
logging.basicConfig(level=logging.INFO)


class HabitationPredictor:
    """Multi-classifier ensemble for predicting EEG habituation"""

    def __init__(self):
        """Initialize 4 classifiers and a scaler"""
        self.classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.best_clf = None
        self.best_name = None
        self.scaler = StandardScaler()
        logger.info("Initialized HabitationPredictor with 4 classifiers")

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train classifiers and select the best performer.
        For very small datasets, it will train on the full dataset without splitting.
        """
        results = {}
        best_score = -1

        # Check for minimal class counts
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2 or min(counts) < 2:
            logger.warning("Too few samples per class. Training on full dataset without split.")
            X_scaled = self.scaler.fit_transform(X)
            for name, clf in self.classifiers.items():
                try:
                    clf.fit(X_scaled, y)
                    score = clf.score(X_scaled, y)
                    results[name] = {'accuracy': score, 'model': clf}
                    logger.info(f"{name} trained on full data, Accuracy={score:.3f}")
                    if score > best_score:
                        best_score = score
                        self.best_clf = clf
                        self.best_name = name
                except Exception as e:
                    logger.warning(f"{name} failed: {e}")
            return results

        # Normal train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        for name, clf in self.classifiers.items():
            clf.fit(X_train_scaled, y_train)
            train_score = clf.score(X_train_scaled, y_train)
            test_score = clf.score(X_test_scaled, y_test)
            results[name] = {'train_accuracy': train_score, 'test_accuracy': test_score, 'model': clf}
            logger.info(f"{name}: Train={train_score:.3f}, Test={test_score:.3f}")
            if test_score > best_score:
                best_score = test_score
                self.best_clf = clf
                self.best_name = name

        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        logger.info(f"Best classifier: {self.best_name} (Test Accuracy={best_score:.3f})")
        return results

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> pd.DataFrame:
        """
        Cross-validation for all classifiers.
        Safely handles small datasets by reducing cv folds.
        """
        X_scaled = self.scaler.fit_transform(X)
        cv = min(cv, len(y))
        if len(np.unique(y)) < 2:
            logger.warning("Not enough classes for cross-validation. Returning empty DataFrame.")
            return pd.DataFrame()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_results = []

        for name, clf in self.classifiers.items():
            try:
                f1 = cross_val_score(clf, X_scaled, y, cv=skf, scoring='f1_weighted')
                acc = cross_val_score(clf, X_scaled, y, cv=skf, scoring='accuracy')
                cv_results.append({
                    'Classifier': name,
                    'F1_mean': f1.mean(),
                    'F1_std': f1.std(),
                    'Accuracy_mean': acc.mean(),
                    'Accuracy_std': acc.std()
                })
                logger.info(f"{name}: F1={f1.mean():.3f}±{f1.std():.3f}, Acc={acc.mean():.3f}±{acc.std():.3f}")
            except Exception as e:
                logger.warning(f"{name} cross-validation failed: {e}")

        return pd.DataFrame(cv_results)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best trained classifier"""
        if self.best_clf is None:
            raise ValueError("Classifier not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.best_clf.predict(X_scaled)

    def predict_with_confidence(self, X: np.ndarray, threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with probability scores and high-confidence mask"""
        if self.best_clf is None:
            raise ValueError("Classifier not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        predictions = self.best_clf.predict(X_scaled)
        probabilities = self.best_clf.predict_proba(X_scaled).max(axis=1)
        high_conf_mask = probabilities >= threshold
        return predictions, probabilities, high_conf_mask

    def get_best_classifier(self) -> Tuple[str, object]:
        """Return the name and instance of the best classifier"""
        if self.best_clf is None:
            raise ValueError("No classifier trained yet.")
        return self.best_name, self.best_clf

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """Return feature importance if the best classifier supports it (tree-based)"""
        if not hasattr(self.best_clf, 'feature_importances_'):
            raise ValueError(f"{self.best_name} does not support feature importance.")
        importances = self.best_clf.feature_importances_
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)