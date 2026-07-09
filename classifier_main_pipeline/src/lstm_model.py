# lstm_model.py

import torch
import torch.nn as nn

class SleepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SleepLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,      # 2 LSTM layers
            batch_first=True,
            dropout=0.3,    # Dropout between layers
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]    # take last timestep
        out = self.fc(out)
        return out
