from scipy.signal import welch

def compute_psd(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    return freqs, psd
