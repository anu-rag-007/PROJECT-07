import mne
import numpy as np
import os

import os
import numpy as np

# Current file directory (src folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level → project root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Define data/processed path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Create folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------
# SETTINGS (CHANGE THESE)
# -----------------------------
PSG_PATH = ".vscode/PROJECT 07/classifier_main_pipeline/data/raw/SC4001E0-PSG.edf"
HYPNO_PATH = ".vscode/PROJECT 07/classifier_main_pipeline/data/raw/SC4001EC-Hypnogram.edf"
EEG_CHANNEL = "EEG Fpz-Cz"   # change if your channel name differs
OUTPUT_DIR = "processed"

EPOCH_LENGTH = 30  # seconds
LOW_FREQ = 0.3
HIGH_FREQ = 35

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD PSG SIGNAL
# -----------------------------
print("Loading PSG file...")
raw = mne.io.read_raw_edf(PSG_PATH, preload=True)

print("Available channels:")
print(raw.ch_names)

# Pick EEG channel
raw.pick_channels([EEG_CHANNEL])

# Bandpass filter
raw.filter(LOW_FREQ, HIGH_FREQ)

sfreq = raw.info['sfreq']
samples_per_epoch = int(EPOCH_LENGTH * sfreq)

signal, _ = raw[:]
signal = signal[0]

# -----------------------------
# LOAD HYPNOGRAM
# -----------------------------
print("Loading Hypnogram...")
annotations = mne.read_annotations(HYPNO_PATH)
raw.set_annotations(annotations)

# Extract labels
events, event_dict = mne.events_from_annotations(raw)

print("Event dictionary:")
print(event_dict)

# -----------------------------
# LABEL MAPPING (Sleep-EDF)
# -----------------------------
stage_mapping = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # merge stage 3 and 4 into N3
    'Sleep stage R': 4,
}

labels = []

for annotation in raw.annotations:
    stage = annotation['description']
    duration = annotation['duration']

    if stage in stage_mapping:
        stage_id = stage_mapping[stage]
        num_epochs = int(duration / EPOCH_LENGTH)
        labels.extend([stage_id] * num_epochs)

labels = np.array(labels)

# -----------------------------
# CREATE SIGNAL EPOCHS
# -----------------------------
num_epochs_signal = len(signal) // samples_per_epoch
signal = signal[:num_epochs_signal * samples_per_epoch]

epochs = signal.reshape(num_epochs_signal, samples_per_epoch)

# -----------------------------
# ALIGN SIGNAL & LABELS
# -----------------------------
min_epochs = min(len(epochs), len(labels))

epochs = epochs[:min_epochs]
labels = labels[:min_epochs]

print("Final Epochs Shape:", epochs.shape)
print("Final Labels Shape:", labels.shape)

# -----------------------------
# SAVE DATA
# -----------------------------
epochs_path = os.path.join(PROCESSED_DIR, "epochs.npy")
labels_path = os.path.join(PROCESSED_DIR, "labels.npy")

np.save(epochs_path, epochs)
np.save(labels_path, labels)

print("Saved epochs at:", epochs_path)
print("Saved labels at:", labels_path)