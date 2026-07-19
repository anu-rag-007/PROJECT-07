# In detect_rem.py — replace old model loading with:
import sys
import os
sys.path.append(os.path.dirname(__file__))
from lstm_model import SleepLSTM   # keep for reference
# Import new model
import torch
import torch.nn as nn

# Add CNNLSTMSleepClassifier definition here
# or import from a new file cnn_lstm_model.py

# Load best model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..',
    'models', 'cnn_lstm_hybrid_best.pth'
)
# model = CNNLSTMSleepClassifier(...)
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()