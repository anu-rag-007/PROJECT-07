import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2. PATH SETUP (Project Root)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# 3. LOAD DATA
# =========================
epochs = np.load(os.path.join(DATA_DIR, "epochs.npy"))
labels = np.load(os.path.join(DATA_DIR, "labels.npy"))

# Convert to Binary (Class 4 = REM)
labels = np.where(labels == 4, 1, 0)

print("\nBinary Class Distribution:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# Convert to tensors
X = torch.tensor(epochs, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

# Ensure 3D for LSTM
if len(X.shape) == 2:
    X = X.unsqueeze(-1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

# =========================
# 4. MODEL (REM Binary)
# =========================
class REMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(REMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


input_size = X.shape[2]
hidden_size = 128

model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# =========================
# 5. LOSS (Weighted for REM)
# =========================
rem_count = np.sum(labels == 1)
nonrem_count = np.sum(labels == 0)

ratio = nonrem_count / rem_count
pos_weight = torch.tensor([ratio], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("Feature variance:", np.var(X, axis=0)[:10])
  
# =========================
# 6. TRAINING
# =========================
epochs_num = 30

for epoch in range(epochs_num):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs_num}], Loss: {total_loss:.4f}")

# =========================
# 7. EVALUATION
# =========================
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)

        outputs = model(batch_X).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.4).int()

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch_y.numpy())

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=["Non-REM", "REM"]))
print("\nProbability stats:")
print("Min:", np.min(probs.cpu().numpy()))
print("Max:", np.max(probs.cpu().numpy()))
print("Mean:", np.mean(probs.cpu().numpy()))

# =========================
# 8. REM Timeline Plot
# =========================
plt.figure(figsize=(12, 4))
plt.plot(all_true[:300], label="True REM", linewidth=2)
plt.plot(all_preds[:300], linestyle='--', label="Predicted REM")

plt.yticks([0, 1], ["Non-REM", "REM"])
plt.xlabel("Epoch Index")
plt.ylabel("State")
plt.title("REM Detection Timeline")
plt.legend()

# =========================
# 7. SAVE HYPNOGRAM
# =========================
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plot_path = os.path.join(PLOTS_DIR, "rem_timeline.png")
plt.savefig(plot_path)
plt.show()

print(f"\nREM Timeline saved at: {plot_path}")