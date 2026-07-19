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
