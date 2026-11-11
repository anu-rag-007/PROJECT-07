import numpy as np
import torch
import torch.nn as nn

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


class CNNLSTMSleepClassifier(nn.Module):
    """Each 30 s epoch is split into windows; a shared CNN embeds each window;
    a bidirectional LSTM reads the window sequence; the output at the LAST
    window feeds a small MLP classifier."""

    def __init__(self, n_windows=10, window_size=300, cnn_out_dim=128,
                 lstm_hidden=128, lstm_layers=2, n_classes=5, dropout=0.3):
        super().__init__()
        self.n_windows = n_windows
        self.window_size = window_size

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(cnn_out_dim // 128),
        )
        cnn_flat_dim = 128 * (cnn_out_dim // 128)

        self.lstm = nn.LSTM(input_size=cnn_flat_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):                       # x: (batch, 3000)
        batch = x.shape[0]
        x = x.reshape(batch, self.n_windows, self.window_size)
        feats = []
        for t in range(self.n_windows):
            w = x[:, t, :].unsqueeze(1)         # (B, 1, 300)
            feats.append(self.cnn(w).reshape(batch, -1))
        seq = torch.stack(feats, dim=1)         # (B, 10, 128)
        out, _ = self.lstm(seq)
        return self.classifier(out[:, -1, :])   # last time step


def normalise_epochs(epochs):
    """Per-epoch z-score, exactly as RawEEGDataset did during training.
    The model was trained on normalised input, so inference MUST do this too."""
    mean = epochs.mean(axis=1, keepdims=True)
    std = epochs.std(axis=1, keepdims=True) + 1e-8
    return ((epochs - mean) / std).astype(np.float32)


def load_model(weights_path, device="cpu"):
    model = CNNLSTMSleepClassifier()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_proba(model, x_norm, batch=512, device="cpu"):
    out = []
    for i in range(0, len(x_norm), batch):
        xb = torch.as_tensor(x_norm[i:i + batch], dtype=torch.float32, device=device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)