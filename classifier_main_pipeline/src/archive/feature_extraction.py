import numpy as np
from scipy.signal import welch

def extract_features(eeg_window, fs):
    """
    eeg_window : 1D numpy array
    fs : sampling frequency
    """

    freqs, psd = welch(eeg_window, fs=fs, nperseg=fs*2)

    # Band power function
    def bandpower(fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx])

    delta = bandpower(0.5, 4)
    theta = bandpower(4, 8)
    alpha = bandpower(8, 13)
    beta  = bandpower(13, 30)

    theta_beta_ratio = theta / (beta + 1e-6)
    
    theta_delta_ratio = theta / (delta + 1e-6)

    variance = np.var(eeg_window)
    mean_val = np.mean(eeg_window)

    spectral_entropy = -np.sum(psd * np.log(psd + 1e-10))

    return np.array([
        delta,
        theta,
        alpha,
        beta,
        theta_beta_ratio,
        variance,
        mean_val,
        spectral_entropy
    ])