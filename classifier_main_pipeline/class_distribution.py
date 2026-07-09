# full_distribution.py
# Loads all Sleep-EDF subjects from local physionet-sleep-data folder

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# ── Paths ─────────────────────────────────────────────────
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(CURRENT_DIR, "data",
                             "physionet-sleep-data")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Looking in:", DATA_DIR)
print("Files found:", len(os.listdir(DATA_DIR)), "\n")

# ── Settings ───────────────────────────────────────────────
EPOCH_LENGTH = 30       # seconds
EEG_CHANNEL  = "EEG Fpz-Cz"
LOW_FREQ     = 0.3
HIGH_FREQ    = 35

stage_mapping = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,   # merge into N3
    'Sleep stage R': 4,
}

stage_names  = {0:'Wake', 1:'N1', 2:'N2', 3:'N3', 4:'REM'}
stage_colors = {
    0:'#E24B4A', 1:'#BA7517',
    2:'#185FA5', 3:'#534AB7', 4:'#1D9E75'
}

# ── Find all PSG + Hypnogram pairs ─────────────────────────
all_files = os.listdir(DATA_DIR)

psg_files   = sorted([f for f in all_files
                       if f.endswith('-PSG.edf')])
hypno_files = sorted([f for f in all_files
                       if 'Hypnogram' in f and f.endswith('.edf')])

print(f"PSG files found:   {len(psg_files)}")
print(f"Hypno files found: {len(hypno_files)}")
print()

# Match PSG to Hypnogram by subject ID
# SC4001E0-PSG.edf → subject ID = SC4001
pairs = []
for psg in psg_files:
    subject_id = psg[:6]   # e.g. 'SC4001'
    # find matching hypnogram
    matching = [h for h in hypno_files
                if h.startswith(subject_id)]
    if matching:
        pairs.append((
            os.path.join(DATA_DIR, psg),
            os.path.join(DATA_DIR, matching[0])
        ))
    else:
        print(f"No hypnogram found for {psg} — skipping")

print(f"Matched pairs: {len(pairs)}\n")
print("="*50)

# ── Process all subjects ───────────────────────────────────
all_epochs = []
all_labels = []
subject_stats = []

for idx, (psg_path, hypno_path) in enumerate(pairs):
    subject = os.path.basename(psg_path)[:6]
    print(f"\nSubject {idx+1:2d}/{len(pairs)} — {subject}")

    try:
        # Load EEG
        raw = mne.io.read_raw_edf(
            psg_path, preload=True, verbose=False
        )

        # Check channel exists
        if EEG_CHANNEL not in raw.ch_names:
            print(f"  Channel {EEG_CHANNEL} not found")
            print(f"  Available: {raw.ch_names[:5]}")
            continue

        raw.pick_channels([EEG_CHANNEL])
        raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

        sfreq             = raw.info['sfreq']
        samples_per_epoch = int(EPOCH_LENGTH * sfreq)

        # Get signal
        signal = raw.get_data()[0]

        # Load hypnogram
        annotations = mne.read_annotations(hypno_path)
        raw.set_annotations(annotations, verbose=False)

        # Build label array
        labels_subj = []
        for ann in raw.annotations:
            stage = ann['description']
            dur   = ann['duration']
            if stage in stage_mapping:
                n_epochs = int(dur / EPOCH_LENGTH)
                labels_subj.extend(
                    [stage_mapping[stage]] * n_epochs
                )

        labels_subj = np.array(labels_subj)

        # Create signal epochs
        n_epochs    = len(signal) // samples_per_epoch
        signal      = signal[:n_epochs * samples_per_epoch]
        epochs_subj = signal.reshape(n_epochs, samples_per_epoch)

        # Align signal and labels
        min_len     = min(len(epochs_subj), len(labels_subj))
        epochs_subj = epochs_subj[:min_len]
        labels_subj = labels_subj[:min_len]

        # Crop to actual sleep period
        non_wake = np.where(labels_subj != 0)[0]
        if len(non_wake) == 0:
            print(f"  No sleep found — skipping")
            continue

        s = non_wake[0]
        e = non_wake[-1] + 1
        epochs_subj = epochs_subj[s:e]
        labels_subj = labels_subj[s:e]

        all_epochs.append(epochs_subj)
        all_labels.append(labels_subj)

        # Per-subject stats
        unique, counts = np.unique(labels_subj,
                                   return_counts=True)
        stage_dist = {stage_names[u]: c
                      for u, c in zip(unique, counts)}
        subject_stats.append({
            'subject': subject,
            'total':   len(labels_subj),
            'dist':    stage_dist
        })

        print(f"  Epochs: {len(epochs_subj)}")
        for u, c in zip(unique, counts):
            print(f"    {stage_names[u]:5s}: {c:4d} "
                  f"({c/len(labels_subj)*100:.1f}%)")

    except Exception as e:
        print(f"  FAILED: {e}")
        continue

# ── Combine all subjects ───────────────────────────────────
if not all_epochs:
    print("\nNo data loaded — check folder structure")
else:
    epochs_all = np.concatenate(all_epochs, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    print(f"\n{'='*50}")
    print(f"TOTAL epochs: {epochs_all.shape}")
    print(f"TOTAL labels: {labels_all.shape}")

    # Save
    np.save(os.path.join(PROCESSED_DIR, "epochs_all.npy"),
            epochs_all)
    np.save(os.path.join(PROCESSED_DIR, "labels_all.npy"),
            labels_all)
    print(f"Saved to {PROCESSED_DIR}")

    # ── Distribution ───────────────────────────────────────
    unique, counts = np.unique(labels_all, return_counts=True)
    total = len(labels_all)

    print(f"\n{'='*50}")
    print("=== FULL DATASET DISTRIBUTION ===\n")
    for stage_id, count in zip(unique, counts):
        name = stage_names[stage_id]
        pct  = count / total * 100
        hrs  = count * 30 / 3600
        bar  = '█' * int(pct)
        print(f"{name:5s}: {count:5d} ({pct:5.1f}%) "
              f"= {hrs:.1f}h  {bar}")

    # ── Plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1 — bar chart
    names  = [stage_names[i] for i in unique]
    cols   = [stage_colors[i] for i in unique]
    axes[0].bar(names, counts, color=cols, edgecolor='white')
    axes[0].set_title('Epoch count per stage\n(all subjects combined)')
    axes[0].set_ylabel('Epochs')
    for i, (n, c) in enumerate(zip(names, counts)):
        axes[0].text(i, c + 20, str(c),
                     ha='center', fontsize=10)

    # Plot 2 — pie chart
    axes[1].pie(counts, labels=names, colors=cols,
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Sleep stage proportions')

    # Plot 3 — per-subject breakdown
    sub_names = [s['subject'] for s in subject_stats]
    bottom    = np.zeros(len(subject_stats))
    for stage_id in range(5):
        name   = stage_names[stage_id]
        values = np.array([
            s['dist'].get(name, 0) for s in subject_stats
        ])
        axes[2].bar(sub_names, values,
                    bottom=bottom,
                    color=stage_colors[stage_id],
                    label=name)
        bottom += values

    axes[2].set_title('Per-subject stage breakdown')
    axes[2].set_xlabel('Subject')
    axes[2].set_ylabel('Epochs')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].tick_params(axis='x',
                        rotation=45, labelsize=7)

    plt.suptitle('Sleep-EDF Dataset — Complete Analysis',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(CURRENT_DIR, "plots",
                     "dataset_distribution.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("\nPlot saved to plots/dataset_distribution.png")