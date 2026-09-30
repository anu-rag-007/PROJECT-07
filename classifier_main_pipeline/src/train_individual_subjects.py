import time
import os, sys, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

THINGS_DIR = r"C:\\Users\\Hp\\.vscode\\PROJECT 07\\classifier_main_pipeline\\data\\things_eeg"
MODELS_DIR = r"C:\\Users\\Hp\\.vscode\\PROJECT 07\\classifier_main_pipeline\\models"
LOG_FILE   = os.path.join(MODELS_DIR, "training_log.txt")
SAVE_TRAIN = os.path.join(THINGS_DIR, 'eeg_train_individual_subjects.npy')
SAVE_TEST  = os.path.join(THINGS_DIR, 'eeg_test_individual_subjects.npy')


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\\n")

log("=== Individual Subject Training ===")
log(f"Started training run")

# Loading data
eeg_train = np.load(os.path.join(THINGS_DIR,
    "eeg_train_individual_subjects.npy"))
eeg_test  = np.load(os.path.join(THINGS_DIR,
    "eeg_test_individual_subjects.npy"))
clip_train = np.load(os.path.join(THINGS_DIR,
    "clip_IMAGE_targets_train.npy"))
clip_test  = np.load(os.path.join(THINGS_DIR,
    "clip_IMAGE_targets_test.npy"))

log(f"Train: {eeg_train.shape}")
log(f"Test:  {eeg_test.shape}")

def evaluate_per_concept(model, test_loader, n_concepts=200):
    """
    Evaluate by averaging predictions across subjects for the same concept.
    """
    model.eval()
    all_eeg_embs = []
    all_clip_embs = []
    concept_ids  = []

    with torch.no_grad():
        for xb, cb, idx in test_loader:
            emb = model(xb)
            all_eeg_embs.append(emb)
            all_clip_embs.append(cb)
            concept_ids.extend(idx.tolist())

    eeg_embs  = torch.cat(all_eeg_embs)
    clip_embs = torch.cat(all_clip_embs)
    concept_ids = np.array(concept_ids)
    
    device = eeg_embs.device

    # Average EEG embeddings per concept across subjects
    avg_eeg = torch.zeros(n_concepts, eeg_embs.shape[1], device=device)
    unique_clip = torch.zeros(n_concepts, clip_embs.shape[1], device=device)
    counts  = torch.zeros(n_concepts, device=device)

    for i, cid in enumerate(concept_ids):
        avg_eeg[cid]  += eeg_embs[i]
        counts[cid]   += 1
        # Correctly map the unique CLIP target for this concept
        unique_clip[cid] = clip_embs[i]

    # Avoid division by zero for concepts that might not appear
    counts = counts.unsqueeze(1).clamp(min=1)
    avg_eeg = avg_eeg / counts
    
    # Normalize both sets of embeddings
    avg_eeg = F.normalize(avg_eeg, dim=-1)
    unique_clip = F.normalize(unique_clip, dim=-1)

    # Retrieval
    sim  = avg_eeg @ unique_clip.T
    res  = {}
    for k in [1, 5, 10]:
        topk    = sim.topk(k, dim=-1).indices
        correct = (topk == torch.arange(n_concepts, device=device).unsqueeze(1))
        # Store the raw tensor, NOT a Python float
        res[f'top{k}'] = correct.any(-1).float().mean()

    return res

def infonce(e, c, t=0.05):
    n   = e.shape[0]
    sim = (e @ c.T) / t
    lbl = torch.arange(n)
    return (F.cross_entropy(sim, lbl) +
            F.cross_entropy(sim.T, lbl)) / 2


# ── Training ───────────────────────────────────────────────
from torch.optim.lr_scheduler import OneCycleLR

def build_individual_dataset(things_dir,save_train,save_test):
    
    if (os.path.exists(save_train) and
            os.path.exists(save_test)):
        print("Loading cached individual data...")
        train = np.load(save_train)
        test  = np.load(save_test)
        print(f"✅ Train: {train.shape}")
        print(f"✅ Test:  {test.shape}")
        return train, test

    # Find all subject folders (17-ch, not 63-ch)
    all_folders = sorted([
        f for f in os.listdir(things_dir)
        if (os.path.isdir(
            os.path.join(things_dir, f))
            and f.startswith('sub-')
            and '63_channels' not in f)
    ])

    print(f"Found {len(all_folders)} subjects "
          f"(17-channel)\n")

    all_train = []
    all_test  = []

    for i, folder in enumerate(all_folders):
        folder_path = os.path.join(
            things_dir, folder)
        npy_files   = list(
            Path(folder_path).rglob('*.npy'))

        train_files = [
            f for f in npy_files
            if 'train' in f.name.lower()]
        test_files  = [
            f for f in npy_files
            if 'test' in f.name.lower()]

        if not train_files or not test_files:
            print(f"  ⚠️  {folder}: missing files")
            continue

        try:
            tr = np.load(str(train_files[0]),
                          allow_pickle=True).item()
            te = np.load(str(test_files[0]),
                          allow_pickle=True).item()

            eeg_tr = tr['preprocessed_eeg_data']
            eeg_te = te['preprocessed_eeg_data']

            # Average repetitions only
            # (1654, 80, 17, 100) → (1654, 17, 100)
            tr_avg = eeg_tr.mean(axis=1)
            te_avg = eeg_te.mean(axis=1)

            all_train.append(tr_avg)
            all_test.append(te_avg)

            print(f"  ✅ {folder}: "
                  f"train={tr_avg.shape}, "
                  f"test={te_avg.shape}")

        except Exception as e:
            print(f"  ❌ {folder}: {e}")

    n_subjects = len(all_train)

    # Stack subjects — each subject is a new set
    # of training examples for the same 1654 concepts
    # Shape: (n_subj × 1654, 17, 100)
    train_all = np.concatenate(
        all_train, axis=0).astype(np.float32)
    test_all  = np.concatenate(
        all_test,  axis=0).astype(np.float32)

    np.save(save_train, train_all)
    np.save(save_test,  test_all)

    print(f"\n✅ Individual dataset built:")
    print(f"   Subjects: {n_subjects}")
    print(f"   Train: {train_all.shape}")
    print(f"         ({n_subjects} × 1654 = "
          f"{n_subjects * 1654} pairs)")
    print(f"   Test:  {test_all.shape}")
    print(f"   Saved to disk")

    return train_all, test_all


# Building it
t0 = time.time()
eeg_train_indiv, eeg_test_indiv = \
    build_individual_dataset(
        THINGS_DIR, SAVE_TRAIN, SAVE_TEST)

class IndividualSubjectDataset(Dataset):
    def __init__(self, eeg, clip_targets,
                  n_concepts, augment=False):
        self.n_concepts = n_concepts
        self.augment    = augment
        n_total         = len(eeg)
        self.n_subjects = n_total // n_concepts

        # Normalise each subject independently
        eeg_norm = eeg.copy().astype(np.float32)
        for s in range(self.n_subjects):
            start = s * n_concepts
            end   = (s + 1) * n_concepts
            subj  = eeg_norm[start:end]

            # Per-channel mean and std across
            # all concepts for this subject
            m = subj.mean(axis=(0, 2),
                           keepdims=True)
            s_ = (subj.std(axis=(0, 2),
                            keepdims=True) + 1e-8)
            eeg_norm[start:end] = (subj - m) / s_

        self.eeg  = eeg_norm
        # Tile clip targets for each subject
        self.clip = np.tile(
            clip_targets.astype(np.float32),
            (self.n_subjects, 1))

        print(f"Dataset: {len(self.eeg)} pairs "
              f"({self.n_subjects} subjects × "
              f"{n_concepts} concepts)")

    def __len__(self): return len(self.eeg)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.eeg[idx])
        c = torch.FloatTensor(self.clip[idx])

        # Concept index (same across subjects)
        concept_idx = idx % self.n_concepts

        if self.augment:
            x = x + torch.randn_like(x) * 0.08
            if torch.rand(1) < 0.25:
                n  = torch.randint(1, 4, (1,)).item()
                ch = torch.randperm(17)[:n]
                x[ch] = 0
            if torch.rand(1) < 0.3:
                s = torch.randint(-3, 4, (1,)).item()
                if s != 0:
                    x = torch.roll(x, s, -1)

        return x, c, concept_idx


# Loading CLIP targets
clip_img_train = np.load(os.path.join(
    THINGS_DIR, 'clip_IMAGE_targets_train.npy'))
clip_img_test  = np.load(os.path.join(
    THINGS_DIR, 'clip_IMAGE_targets_test.npy'))

n_train_concepts = 1654
n_test_concepts  = 200

train_ds = IndividualSubjectDataset(
    eeg_train_indiv, clip_img_train,
    n_train_concepts, augment=True)

test_ds  = IndividualSubjectDataset(
    eeg_test_indiv,  clip_img_test,
    n_test_concepts, augment=False)

train_loader = DataLoader(
    train_ds, batch_size=256,
    shuffle=True, drop_last=True,
    num_workers=0)

test_loader  = DataLoader(
    test_ds, batch_size=200,
    shuffle=False, num_workers=0)

class ChannelWiseAttention(nn.Module):
    def __init__(self, n_ch, d_model,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_model, n_ch),
            nn.Sigmoid())

    def forward(self, x):
        a, w = self.attn(x, x, x)
        x    = self.norm(x + self.drop(a))
        g    = self.gate(
            x.mean(1, keepdim=True)).transpose(1,2)
        return x * g, w, g


class TemporalSpatialConv(nn.Module):
    def __init__(self, n_ch, n_tp, d_model,
                 dropout=0.2):
        super().__init__()
        self.t_conv = nn.Sequential(
            nn.Conv1d(n_ch, d_model, 15, padding=7),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 5, padding=2),
            nn.BatchNorm1d(d_model), nn.GELU())
        self.s_conv = nn.Sequential(
            nn.Conv1d(n_tp, d_model,
                      min(5, n_ch),
                      padding=min(5, n_ch)//2),
            nn.BatchNorm1d(d_model), nn.GELU())
        self.fuse = nn.Linear(d_model*2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        t = self.t_conv(x).mean(-1)
        s = self.s_conv(x.transpose(1,2)).mean(-1)
        return self.norm(
            self.fuse(torch.cat([t, s], -1)))


class ATMEEGEncoder(nn.Module):
    def __init__(self, n_ch=17, n_tp=100,
                 d_model=128, clip_dim=512,
                 n_heads=4, dropout=0.3):
        super().__init__()
        self.embed   = nn.Linear(n_tp, d_model)
        self.enorm   = nn.LayerNorm(d_model)
        self.ch_attn = ChannelWiseAttention(
            n_ch, d_model, n_heads, dropout)
        self.ts_conv = TemporalSpatialConv(
            n_ch, n_tp, d_model, dropout)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model),
            nn.LayerNorm(d_model))
        self.agg  = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.GELU(), nn.LayerNorm(d_model))
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(d_model*2, clip_dim))

    def forward(self, x):
        e       = self.enorm(self.embed(x))
        a, w, g = self.ch_attn(e)
        t       = self.ts_conv(x)
        f       = self.ffn(a)
        p       = self.agg(torch.cat(
            [f.mean(1), t], -1))
        return F.normalize(self.proj(p), -1)


model = ATMEEGEncoder()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4, weight_decay=0.05)
scheduler = OneCycleLR(
    optimizer, max_lr=3e-4,
    epochs=200,
    steps_per_epoch=len(train_loader),
    pct_start=0.1)

EPOCHS    = 200
best_top5 = 0
losses    = []
top5_hist = []

print("Training ATM — individual subjects\n")
print(f"Training pairs: {len(train_ds):,}")
print(f"  (vs Week 12 averaged: 1,654)")
print(f"  (vs Scotti et al.:   ~16,540)\n")
print(f"{'Epoch':>6} | {'Loss':>8} | "
      f"{'Top-1':>7} | {'Top-5':>7} | {'Top-10':>8}")
print("-"*50)

for epoch in range(EPOCHS):
    model.train()
    ep_loss = []

    for xb, cb, _ in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = infonce(pred, cb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ep_loss.append(loss.item())

    losses.append(np.mean(ep_loss))

    if epoch % 20 == 0 or epoch == EPOCHS-1:
        # Subject-averaged evaluation
        acc = evaluate_per_concept(
            model, test_loader, n_concepts=200)
        
        # Safely extracting Python floats ONCE
        top1_val  = float(acc['top1'])
        top5_val  = float(acc['top5'])
        top10_val = float(acc['top10'])
        
        top5_hist.append(top5_val)

        if top5_val > best_top5:
            best_top5 = top5_val
            torch.save(model.state_dict(),
                os.path.join(MODELS_DIR,
                    'atm_individual_subj_best.pth'))
            flag = " ✅"
        else:
            flag = ""

        print(f"{epoch+1:>6} | "
              f"{losses[-1]:>8.4f} | "
              f"{top1_val:>7.4f} | "
              f"{top5_val:>7.4f} | "
              f"{top10_val:>8.4f}{flag}")

print(f"\nBest Top-5: {best_top5:.4f}")
print(f"Week 12 best:  0.045")
print(f"Improvement:   "
      f"{(best_top5-0.045)/0.045*100:+.1f}%")