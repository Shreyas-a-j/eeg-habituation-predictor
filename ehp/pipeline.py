import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from .data_loader import load_eeg_dataset, preprocess_signal
from .feature_extractor import EEGFeatureExtractor
from .classifier import HabitationPredictor

from sklearn.model_selection import train_test_split, cross_val_score

def generate_labels_from_filenames(eeg_files):
    labels = []
    for file in eeg_files:
        name = file.name.lower()
        # Example rules (edit based on your dataset)
        if "r01" in name or "rest" in name:
            labels.append(0)  # non-habituated / baseline
        else:
            labels.append(1)  # habituated / task
    return labels

def run_full_analysis(dataset_dir, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = Path(dataset_dir)

    eeg_files = list(dataset_path.glob('*.edf')) + list(dataset_path.glob('*.EDF'))
    print(f"[DEBUG] Found {len(eeg_files)} EEG files in {dataset_dir}")

    if len(eeg_files) == 0:
        print("[ERROR] No EEG files found - check dataset directory")
        return

    signals = []
    for i, file in enumerate(eeg_files):
        print(f"[DEBUG] Loading EEG file {i+1}/{len(eeg_files)}: {file.name}")
        try:
            signal, meta = load_eeg_dataset(str(file))
            processed_signal = preprocess_signal(signal, fs=meta['sampling_rate'])
            signals.append(processed_signal)
        except Exception as e:
            print(f"[ERROR] Failed to load {file.name}: {e}")

    print(f"[DEBUG] Total usable signals: {len(signals)}")
    if len(signals) == 0:
        print("[ERROR] No usable signals loaded")
        return

    # Feature extraction
    extractor = EEGFeatureExtractor()
    print("[DEBUG] Starting feature extraction...")
    features_df = extractor.batch_extract_features(signals)
    print(f"[DEBUG] Features extracted: {features_df.shape}")

    # Generate labels and attach to features
    labels = generate_labels_from_filenames(eeg_files)
    labels = np.array(labels[:len(features_df)])
    print(f"[DEBUG] Labels created: {sum(labels)} positives, {len(labels) - sum(labels)} negatives")
    features_df['label'] = labels  # <-- important fix

    # Save features with labels
    features_csv_path = os.path.join(output_dir, 'features.csv')
    features_df.to_csv(features_csv_path, index=False)
    print(f"[INFO] Features with labels saved to {features_csv_path}")

    # Initialize classifier
    clf = HabitationPredictor()

    try:
        print("[DEBUG] Training classifier on full dataset...")
        clf.train(features_df.drop(columns='label').values, features_df['label'].values)
        print("[DEBUG] Training complete")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return

    # Run cross-validation
    try:
        print("[DEBUG] Running cross-validation...")
        cv_results = clf.cross_validate(features_df.drop(columns='label').values, features_df['label'].values, cv=3)
        print("[DEBUG] Cross-validation results:")
        print(cv_results)
    except Exception as e:
        print(f"[ERROR] Cross-validation failed: {e}")

    # Save trained model
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(clf, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Clean feature DataFrame
    features_df = features_df.dropna(axis=1).fillna(0)
    print("[DEBUG] Cleaned features shape:", features_df.shape)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ehp.pipeline <dataset_dir> [output_dir]")
        exit(1)
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results'
    run_full_analysis(dataset_dir, output_dir)