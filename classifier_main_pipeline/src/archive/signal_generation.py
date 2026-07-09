import numpy as np

def generate_signal(freq, fs, duration, noise_std=0.3):
    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, noise_std, len(t))
    return signal + noise

