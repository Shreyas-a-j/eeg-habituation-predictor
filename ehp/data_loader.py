import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_eeg_dataset(filepath, channel_index=0, duration_seconds=60):
    from mne.io import read_raw_edf
    filepath = Path(filepath)

    raw = read_raw_edf(str(filepath), preload=True, verbose=False)
    sampling_rate = raw.info['sfreq']
    n_samples = int(sampling_rate * duration_seconds)
    signal = raw.get_data(picks=channel_index, start=0, stop=n_samples).flatten()

    logger.info(f"Loaded {filepath.name}: {len(signal)} samples at {sampling_rate} Hz")
    return signal, {'filename': filepath.name, 'sampling_rate': sampling_rate}

def preprocess_signal(signal, fs=256, lowcut=1.0, highcut=50.0, order=4):
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    norm_signal = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    return norm_signal
