# Project 07 — Experimental Results

*Last updated: 2026-07-19*

*Total experiments: 3*

## Summary

| Exp | Name | Accuracy | REM F1 | REM Recall | Date |
|-----|------|----------|--------|------------|------|
| 001 | SleepLSTM Baseline | 0.7685 | N/A | 0.8402 | 2026-07-12 |
| 002 | CNN on EEG Spectrograms | 0.7167 | N/A | 0.6578 | 2026-07-12 |
| 003 | CNN-LSTM Hybrid | 0.0000 | N/A | N/A | 2026-07-14 |

**Current best: Experiment 001 (0.7685)**

---

## Detailed Results

### Experiment 001 — SleepLSTM Baseline

**Date**: 2026-07-12  
**Model**: SleepLSTM  
**Accuracy**: 0.7685

| Stage | F1 Score |
|-------|----------|
| Wake | 0.5429 |
| N1 | 0.3831 |
| N2 | 0.7806 |
| N3 | 0.8809 |
| REM | 0.8402 |

---

### Experiment 002 — CNN on EEG Spectrograms

**Date**: 2026-07-12  
**Model**: ResNet18 fine-tuned  
**Accuracy**: 0.7167

**vs baselines:**

---

### Experiment 003 — CNN-LSTM Hybrid

**Date**: 2026-07-14  
**Model**: CNNLSTMSleepClassifier  
**Accuracy**: 0.0000

**vs baselines:**
- vs_exp001_lstm accuracy_delta: ↑0.0329
- vs_exp001_lstm rem_recall_delta: ↓0.0131
- vs_exp002_cnn accuracy_delta: ↑0.0847
- vs_exp002_cnn rem_recall_delta: ↑0.1669

---

## Experiment 004 — EEG Transformer
**Result**: 79.29% accuracy

### Key finding
Transformer underperforms CNN-LSTM by 0.85% on this
dataset size (18,226 epochs). Consistent with literature
showing transformers require larger datasets than RNNs
to overcome their lack of sequential inductive bias.

### Scientific contribution
Attention maps provide interpretable evidence of which
3-second EEG windows drive sleep stage classification —
independent of accuracy comparisons.

### Implication for Project 07
CNN-LSTM remains primary classifier.
Transformer used for interpretability analysis.
Both results contribute to the paper.

### SUMMARY METRICS TABLE

=================================================================
TABLE II — Classification Performance
Sleep-EDF Cassette Dataset (20 subjects)
=================================================================

Stage        Prec      Rec       F1      AUC
---------------------------------------------
Wake        0.707    0.633    0.668    0.953
N1          0.385    0.609    0.472    0.903
N2          0.915    0.804    0.856    0.941
N3          0.806    0.899    0.850    0.983
REM         0.794    0.827    0.810    0.962
---------------------------------------------
Macro                         0.731    0.949
Weighted                      0.809
=================================================================

Overall Accuracy:  0.8014
Cohen's Kappa:     0.7128
MCC:               0.7161
================================================================

## Experiment 003 — CNN-LSTM (Complete Evaluation)

| Metric          | Value  | Context                    |
|-----------------|--------|----------------------------|
| Accuracy        | 80.14% | Best across 4 experiments  |
| Cohen's Kappa   | 0.7128 | "Good" agreement           |
| MCC             | 0.7161      |       
| Macro F1        | 0.731      |       
| REM F1          | 0.81      |       
| REM AUC         | 0.962      |      

### Benchmark comparison
| Model              | Kappa | Year |
|--------------------|-------|------|
| DeepSleepNet       | 0.76  | 2017 |
| **Our CNN-LSTM**  | **0.71** | 2026 |
| AttnSleep          | 0.78  | 2021 |

### Notes
- Single channel (Fpz-Cz) vs multi-channel in benchmarks
- 20 subjects vs 78 available in Sleep-EDF Cassette
- Optimised for real-time BCI deployment, not benchmark
## LUCID Phase 2 — EEG-Guided Image Generation

### Status: v0.1 Prototype

### Dataset
- THINGS-EEG (Gifford et al., 2022)
- 50 subjects × 1,654 training + 200 test concepts
- EEG: (1654, 17, 100) after averaging
- Sampling: 100 Hz, window: -200ms to 790ms

### Model: EEGAlignmentMLP v1
- Architecture: 1700 → 2048 → 1024 → 512
- Loss: InfoNCE contrastive (temperature=0.07)
- Training: 200 epochs, AdamW + OneCycleLR

### Results: EEG → CLIP Retrieval
| Metric  | Score  | Chance  | Notes |
|---------|--------|---------|-------|
| Top-1   | 0.005  | 0.005   | At chance |
| Top-5   | 0.025  | 0.025   | At chance |
| Top-10  | 0.030  | 0.050   | Below chance |

### Root cause
Text CLIP embeddings used as targets (not image embeddings).
CLIP text-image modality gap prevents effective alignment.
Fix: download THINGS images (~2GB) → use image embeddings.

### Generation pipeline: working ✅
EEG → nearest concept (NN search) → SD prompt → image
ComfyUI integration confirmed working.
Dreamlike imagery generated for all 4 test sleep stages.

### Next experiment
Fix A: image CLIP targets → retrain
Target: Top-5 > 15% (published baseline: ~22%)
