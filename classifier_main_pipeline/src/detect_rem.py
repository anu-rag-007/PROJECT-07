import joblib
import numpy as np
from archive.feature_extraction import extract_features

model = joblib.load("rem_model.pkl")

def predict_rem(eeg_window, fs):
    features = extract_features(eeg_window, fs)
    features = features.reshape(1, -1)

    prob = model.predict_proba(features)[0][1]

    if prob > 0.6:
        return 1, prob  # REM
    else:
        return 0, prob  # Non-REM