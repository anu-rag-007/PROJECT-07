import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from lstm_model import SleepLSTM

# ── Load ───────────────────────────────────────────────────
import os
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "plots")

sys.path.append(SRC_DIR)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data exists:  {os.path.exists(PROCESSED_DIR)}")
print(f"Files:        {os.listdir(PROCESSED_DIR)}")

epochs_all = np.load(os.path.join(PROCESSED_DIR,"epochs_all.npy"))
labels_all = np.load(os.path.join(PROCESSED_DIR,"labels_all.npy"))

print(f"Dataset: {epochs_all.shape}, {labels_all.shape}")

# ── Reshape for LSTM (samples, seq_len, features) ─────────
SEQ_LEN      = 100
FEAT_PER_STEP = 30    # 3000 / 100

X = epochs_all.reshape(-1, SEQ_LEN, FEAT_PER_STEP).astype(np.float32)
y = labels_all.astype(np.int64)

# Normalise — zero mean, unit std per feature
mean = X.mean(axis=(0,1), keepdims=True)
std  = X.std(axis=(0,1), keepdims=True) + 1e-8
X    = (X - mean) / std

print(f"X: {X.shape}, y: {y.shape}")

# ── Split — stratified ─────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

# ── Class weights ──────────────────────────────────────────
weights = compute_class_weight(
    'balanced', classes=np.unique(y_tr), y=y_tr
)
weights_tensor = torch.FloatTensor(weights)
print(f"\nClass weights: {weights.round(3)}")

# ── DataLoader ─────────────────────────────────────────────
train_ds = TensorDataset(
    torch.FloatTensor(X_tr),
    torch.LongTensor(y_tr)
)
train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True
)

# ── Model ──────────────────────────────────────────────────
model = SleepLSTM(
    input_dim=FEAT_PER_STEP,
    hidden_dim=128,
    output_dim=5
)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ── Evaluate function ──────────────────────────────────────
X_te_t = torch.FloatTensor(X_te)
y_te_t = torch.LongTensor(y_te)

def evaluate(model):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(X_te_t), 128):
            batch = X_te_t[start:start+128]
            preds = model(batch).argmax(1)
            all_preds.extend(preds.numpy())
    return np.array(all_preds)

# ── Training loop ──────────────────────────────────────────
stage_names = ['Wake','N1','N2','N3','REM']
losses, accs = [], []
best_acc = 0

print("\nTraining SleepLSTM...\n")

for epoch in range(60):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()

        # Gradient clipping — important for LSTMs
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    preds    = evaluate(model)
    acc      = (preds == y_te).mean()
    avg_loss = total_loss / len(train_loader)

    losses.append(avg_loss)
    accs.append(acc)
    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(
            model.state_dict(),
            os.path.join(PROJECT_ROOT, 'models',
                         'best_sleep_lstm.pth')
        )
        flag = " ← best"
    else:
        flag = ""

    if epoch % 5 == 0 or flag:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {acc:.4f} | "
              f"LR: {lr:.6f}{flag}")

# ── Final evaluation ───────────────────────────────────────
# Load best model
model.load_state_dict(
    torch.load(os.path.join(PROJECT_ROOT,
               'models', 'best_sleep_lstm.pth'))
)
final_preds = evaluate(model)

print(f"\n{'='*50}")
print(f"Best accuracy: {best_acc:.4f}")
print(f"\n=== Classification Report ===\n")
print(classification_report(
    y_te, final_preds,
    target_names=stage_names
))

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(losses, color='#185FA5', linewidth=2)
axes[0].set_title('Training loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-entropy loss')
axes[0].grid(alpha=0.3)

# Accuracy curve
axes[1].plot(accs, color='#1D9E75', linewidth=2)
axes[1].axhline(y=best_acc, color='#E24B4A',
                linestyle='--', alpha=0.7,
                label=f'Best: {best_acc:.4f}')
axes[1].set_title('Test accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Confusion matrix
cm = confusion_matrix(y_te, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=stage_names,
            yticklabels=stage_names,
            ax=axes[2])
axes[2].set_title('Confusion matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')

plt.suptitle('SleepLSTM — Training Results',
             fontsize=13)
plt.tight_layout()
plt.savefig(
    os.path.join(PROJECT_ROOT, 'plots',
                 'training_results.png'),
    dpi=150, bbox_inches='tight'
)
plt.show()

print(f"\nPlot saved to plots/training_results.png")