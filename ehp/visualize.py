"""
EEG Wave Visualization by Class
Plots delta, theta, alpha, beta powers as waveforms with mean ± std shading per class.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def visualize_eeg_waves_by_class(features_csv, output_dir='results', label_column='label'):
    """
    Visualize EEG wave features (delta, theta, alpha, beta) per class with mean ± std shading.

    Parameters
    ----------
    features_csv : str
        Path to features.csv from pipeline.
    output_dir : str
        Directory to save plots.
    label_column : str
        Column name for labels (0/1).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load features
    features_df = pd.read_csv(features_csv)
    logger.debug(f"Loaded features shape: {features_df.shape}")

    # Check for wave columns
    wave_columns = [col for col in features_df.columns if col.lower() in ['delta', 'theta', 'alpha', 'beta']]
    if not wave_columns:
        raise ValueError("No EEG wave columns found in features.csv! Columns should include: delta, theta, alpha, beta")

    # Check for label column
    if label_column not in features_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in features.csv!")

    # Plot per wave
    plt.figure(figsize=(14, 7))
    for col in wave_columns:
        # Compute mean and std per class
        class0 = features_df[features_df[label_column] == 0][col].values
        class1 = features_df[features_df[label_column] == 1][col].values

        if len(class0) > 0:
            mean0 = np.mean(class0)
            std0 = np.std(class0)
            plt.plot(class0, label=f"{col} - class 0", color='blue', alpha=0.5)
            plt.fill_between(range(len(class0)), class0 - std0, class0 + std0, color='blue', alpha=0.2)

        if len(class1) > 0:
            mean1 = np.mean(class1)
            std1 = np.std(class1)
            plt.plot(class1, label=f"{col} - class 1", color='red', alpha=0.5)
            plt.fill_between(range(len(class1)), class1 - std1, class1 + std1, color='red', alpha=0.2)

    plt.title("EEG Wave Power per Class (Mean ± Std Shading)")
    plt.xlabel("Signal Index")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)

    # Save figure
    plot_path = os.path.join(output_dir, 'eeg_waves_by_class.png')
    plt.savefig(plot_path)
    plt.show()
    logger.info(f"EEG wave plot by class saved to {plot_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ehp.visualize <features_csv> [output_dir]")
        exit(1)

    features_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results'
    visualize_eeg_waves_by_class(features_csv, output_dir)