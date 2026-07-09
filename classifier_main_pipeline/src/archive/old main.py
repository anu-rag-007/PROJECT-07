import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1. DEVICE CONFIG
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2. LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

epochs = np.load(os.path.join(DATA_DIR, "epochs.npy"))
labels = np.load(os.path.join(DATA_DIR, "labels.npy"))

labels = np.where(labels == 4, 1, 0)

print("Binary Class Distribution:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# Convert to tensors
X = torch.tensor(epochs, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)
if len(X.shape) == 2:
    X = X.unsqueeze(-1)

print("Final X shape:", X.shape)
 
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
# If data is 2D (samples, features), convert to 3D for LSTM
if len(X.shape) == 2:
    X = X.unsqueeze(-1)   # adds feature dimension
    
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# =========================
# 3. ATTENTION MODEL
# =========================
class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        output = self.classifier(context)
        return output


input_size = X.shape[2]
hidden_size = 128
num_classes = 2

model = AttentionModel(input_size, hidden_size, num_classes).to(device)

# ===== CLASS WEIGHTS FOR IMBALANCE =====
# Compute REM weight
rem_count = np.sum(labels == 1)
nonrem_count = np.sum(labels == 0)

pos_weight = torch.tensor([nonrem_count / rem_count], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# =========================
# 4. TRAINING LOOP
# =========================
epochs_num = 15

for epoch in range(epochs_num):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs_num}], Loss: {total_loss:.4f}")

# =========================
# 5. EVALUATION
# =========================
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).int()

        all_preds.extend(preds.numpy())
        all_true.extend(batch_y.numpy())

accuracy = accuracy_score(all_true, all_preds)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(confusion_matrix(all_true, all_preds))
print(classification_report(all_true, all_preds, target_names=["Non-REM", "REM"]))

# =========================
# 6. HYPNOGRAM PLOT
# =========================
plt.figure(figsize=(12, 4))
plt.plot(all_true[:300], label="True", linewidth=2)
plt.plot(all_preds[:300], label="Predicted", linestyle='--')
plt.yticks(range(num_classes))
plt.xlabel("Epoch Index")
plt.ylabel("Sleep Stage")
plt.title("Hypnogram")
plt.legend()

# =========================
# 7. SAVE HYPNOGRAM
# =========================
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plot_path = os.path.join(PLOTS_DIR, "hypnogram.png")
plt.savefig(plot_path)
plt.show()

print(f"Hypnogram saved at: {plot_path}")