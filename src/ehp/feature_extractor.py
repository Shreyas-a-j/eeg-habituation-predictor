import numpy as np
from scipy.signal import welch
import pandas as pd

class EEGFeatureExtractor:
    def __init__(self, sampling_rate=256):
        self.fs = sampling_rate

    def extract_features(self, signal):
        signal = signal.flatten()
        freqs, pxx = welch(signal, self.fs, nperseg=self.fs*4)
        theta = np.mean(pxx[(freqs >= 4) & (freqs <= 8)])
        alpha = np.mean(pxx[(freqs >= 8) & (freqs <= 12)])
        beta = np.mean(pxx[(freqs >= 12) & (freqs <= 30)])
        theta_alpha_ratio = theta / (alpha + 1e-8)
        sample_entropy = self._approximate_entropy(signal)
        std = np.std(signal)
        complexity = self._hjorth_complexity(signal)
        return {'theta_power': theta, 'alpha_power': alpha, 'beta_power': beta,
                'theta_alpha_ratio': theta_alpha_ratio, 'sample_entropy': sample_entropy,
                'signal_std': std, 'hjorth_complexity': complexity}

    def batch_extract_features(self, signals):
        features = []
        for i, signal in enumerate(signals):
            print(f"[DEBUG] Extracting features from signal {i+1}/{len(signals)} (length={len(signal)})")
            feats = self.extract_features(signal)
            features.append(feats)
        return pd.DataFrame(features)

    @staticmethod
    def _approximate_entropy(signal, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(signal)
        # Placeholder for approximate entropy, simplified version or use external packages
        return 2.5

    @staticmethod
    def _hjorth_complexity(signal):
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        var0 = np.var(signal)
        var1 = np.var(diff1)
        var2 = np.var(diff2)
        if var1 == 0 or var0 == 0:
            return 0
        mobility = np.sqrt(var1 / var0)
        return np.sqrt(var2 / var1) / mobility if mobility != 0 else 0
