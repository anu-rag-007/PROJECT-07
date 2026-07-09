import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SleepSequenceDataset(Dataset):
    def __init__(self, epochs_path, labels_path, seq_len=5):

        self.seq_len = seq_len
        self.half_window = seq_len // 2

        # Load data
        self.epochs = np.load(epochs_path)
        self.labels = np.load(labels_path)

        # Normalize per epoch
        self.epochs = (self.epochs - np.mean(self.epochs, axis=1, keepdims=True)) / \
                      np.std(self.epochs, axis=1, keepdims=True)

        self.epochs = torch.tensor(self.epochs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # Reduce usable length due to windowing
        self.valid_indices = list(range(self.half_window,
                                        len(self.epochs) - self.half_window))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):

        center_idx = self.valid_indices[idx]

        start = center_idx - self.half_window
        end = center_idx + self.half_window + 1

        sequence = self.epochs[start:end]  # shape: (5, 3000)
        label = self.labels[center_idx]    # center label

        # Add channel dimension for CNN
        sequence = sequence.unsqueeze(1)   # (5, 1, 3000)

        return sequence, label