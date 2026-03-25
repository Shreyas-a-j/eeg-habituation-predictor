import os
from pathlib import Path
import numpy as np
import pandas as pd

from .data_loader import load_eeg_dataset, preprocess_signal
from .feature_extractor import EEGFeatureExtractor
from .classifier import HabitationPredictor

def run_full_analysis(dataset_dir, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = Path(dataset_dir)

    eeg_files = list(dataset_path.glob('*.edf')) + list(dataset_path.glob('*.EDF'))
    print(f"[DEBUG] Found {len(eeg_files)} EEG files in {dataset_dir}")
    if len(eeg_files) == 0:
        print("[ERROR] No EEG files found - check dataset directory and file extensions")
        return

    signals = []
    for i, file in enumerate(eeg_files):
        print(f"[DEBUG] Loading EEG file {i+1}/{len(eeg_files)}: {file.name}")
        try:
            signal, meta = load_eeg_dataset(str(file))
            print(f"[DEBUG] Loaded signal length: {len(signal)}, dtype: {type(signal)}")
            processed_signal = preprocess_signal(signal, fs=meta['sampling_rate'])
            print(f"[DEBUG] Processed signal length: {len(processed_signal)}")
            signals.append(processed_signal)
        except Exception as e:
            print(f"[ERROR] Failed to load or preprocess {file.name}: {e}")

    print(f"[DEBUG] Total usable signals: {len(signals)}")
    if len(signals) == 0:
        print("[ERROR] No usable signals loaded, exiting")
        return

    extractor = EEGFeatureExtractor()
    print("[DEBUG] Starting feature extraction...")
    features_df = extractor.batch_extract_features(signals)
    print(f"[DEBUG] Features extracted: {features_df.shape}")
    print(features_df.head())

    if features_df.empty:
        print("[ERROR] Feature extraction returned empty DataFrame, exiting")
        return

    np.random.seed(42)
    labels = np.random.binomial(1, 0.3, len(features_df))
    print(f"[DEBUG] Synthetic labels created: {sum(labels)} positives, {len(labels) - sum(labels)} negatives")

    clf = HabitationPredictor()

    try:
        print("[DEBUG] Running cross-validation...")
        cv_results = clf.cross_validate(features_df.values, labels, cv=3)
        print(cv_results)
    except Exception as e:
        print(f"[ERROR] Cross-validation failed: {e}")
        return

    try:
        print("[DEBUG] Training classifiers on full data...")
        clf.train(features_df.values, labels)
        print("[DEBUG] Training complete")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return

    features_df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    print(f"[INFO] Features saved: {os.path.join(output_dir, 'features.csv')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ehp.pipeline <dataset_dir> [output_dir]")
        exit(1)
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results'
    run_full_analysis(dataset_dir, output_dir)
