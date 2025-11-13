"""
Evaluation and Visualization Module
Comprehensive metrics and plots for model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Compute and visualize evaluation metrics"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None):
        """
        Initialize evaluation
        
        Parameters:
        -----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray
            Prediction probabilities (for ROC)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        logger.info(f"Initialized evaluation for {len(y_true)} samples")
    
    def compute_metrics(self) -> dict:
        """
        Compute classification metrics
        
        Returns:
        --------
        dict : Metrics dictionary
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1': f1_score(self.y_true, self.y_pred, zero_division=0),
            'specificity': self._specificity(),
            'sensitivity': recall_score(self.y_true, self.y_pred, zero_division=0),
        }
        
        if self.y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
            metrics['auc'] = auc(fpr, tpr)
        
        return metrics
    
    def _specificity(self) -> float:
        """Calculate specificity (true negative rate)"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        if cm.shape[0] == 2:
            tn, fp = cm[0, 0], cm[0, 1]
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0
    
    def plot_confusion_matrix(self, save_path: str = None, normalize: bool = True):
        """Plot confusion matrix heatmap"""
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2%' if normalize else '.0f', 
                   cmap='Blues', cbar=True, square=True)
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, save_path: str = None):
        """Plot ROC curve"""
        
        if self.y_pred_proba is None:
            logger.warning("y_pred_proba required for ROC curve")
            return
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved ROC curve: {save_path}")
        
        plt.close()
    
    def plot_precision_recall(self, save_path: str = None):
        """Plot Precision-Recall curve"""
        
        if self.y_pred_proba is None:
            logger.warning("y_pred_proba required for PR curve")
            return
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved PR curve: {save_path}")
        
        plt.close()
    
    def generate_report(self) -> str:
        """Generate text report of metrics"""
        
        metrics = self.compute_metrics()
        
        report = "=" * 60 + "\n"
        report += "EVALUATION METRICS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Accuracy:    {metrics['accuracy']:.4f}\n"
        report += f"Precision:   {metrics['precision']:.4f}\n"
        report += f"Recall:      {metrics['recall']:.4f}\n"
        report += f"F1 Score:    {metrics['f1']:.4f}\n"
        report += f"Sensitivity: {metrics['sensitivity']:.4f}\n"
        report += f"Specificity: {metrics['specificity']:.4f}\n"
        
        if 'auc' in metrics:
            report += f"AUC-ROC:     {metrics['auc']:.4f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "INTERPRETATION\n"
        report += "=" * 60 + "\n"
        
        if metrics['f1'] > 0.75:
            report += "✓ Strong classification performance (F1 > 0.75)\n"
        elif metrics['f1'] > 0.60:
            report += "✓ Moderate classification performance (F1 > 0.60)\n"
        else:
            report += "✗ Weak classification performance (F1 < 0.60)\n"
        
        if metrics['precision'] > metrics['recall']:
            report += "✓ Low false positive rate (precision > recall)\n"
        else:
            report += "✓ Better recall than precision (more true positives captured)\n"
        
        return report


def plot_cv_comparison(cv_results_dict: dict, metric: str = 'f1', save_path: str = None):
    """
    Plot cross-validation comparison across classifiers
    
    Parameters:
    -----------
    cv_results_dict : dict
        Dictionary with classifier names as keys and metric arrays as values
    metric : str
        Metric to plot (f1, auc, accuracy)
    save_path : str
        Path to save figure
    """
    
    plt.figure(figsize=(10, 6))
    
    classifiers = list(cv_results_dict.keys())
    means = [np.mean(cv_results_dict[clf]) for clf in classifiers]
    stds = [np.std(cv_results_dict[clf]) for clf in classifiers]
    
    x = np.arange(len(classifiers))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    
    plt.xlabel('Classifier')
    plt.ylabel(f'{metric.upper()} Score')
    plt.title(f'Cross-Validation {metric.upper()} Comparison')
    plt.xticks(x, classifiers, rotation=45, ha='right')
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved CV comparison plot: {save_path}")
    
    plt.close()