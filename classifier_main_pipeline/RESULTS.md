## **# Experimental Results**



##### **## Experiment 001 — Baseline SleepLSTM**

\*\*Date\*\*: July 2026  

\*\*Dataset\*\*: Sleep-EDF Cassette, 20 subjects  

\*\*Model\*\*: SleepLSTM (2-layer, hidden\_dim=128, dropout=0.3)  

\*\*Training\*\*: 60 epochs, Adam lr=0.001, weighted cross-entropy  



##### **### Dataset distribution**

| Stage | Epochs | Percentage |

|-------|--------|------------|

| Wake  | 1,049  | 5.8%       |

| N1    | 1,240  | 6.8%       |

| N2    | 9,200  | 50.5%      |

| N3    | 2,981  | 16.4%      |

| REM   | 3,756  | 20.6%      |

| Total | 18,226 | 100%       |



##### **### Classification results**

| Stage | Precision | Recall | F1   | Support |

|-------|-----------|--------|------|---------|

| Wake  | 0.69      | 0.54   | 0.61 | 210     |

| N1    | 0.32      | 0.38   | 0.35 | 248     |

| N2    | 0.89      | 0.78   | 0.83 | 1841    |

| N3    | 0.84      | 0.88   | 0.86 | 596     |

| REM   | 0.67      | 0.84   | 0.74 | 751     |



\***\*Overall accuracy: 76.85%\*\***



##### **### Key findings**

\- REM recall 0.84 — sufficient for reliable trigger detection

\- N1 F1 0.35 — expected, N1 is the hardest stage to classify

\- N2 and N3 performance matches published Sleep-EDF benchmarks

\- Weighted loss successfully prevented N2 dominance



##### **### Next experiment**

\- Add CNN feature extractor before LSTM

\- Bidirectional LSTM

\- Cross-subject validation (leave-one-subject-out)


## Experiment 002 — CNN on EEG Spectrograms
**Date**: July 2026  
**Model**: ResNet18 fine-tuned on EEG spectrograms  
**Result**: 71.67% accuracy, REM recall 0.66

### Comparison vs Experiment 001
| Metric           | LSTM (001) | CNN (002) | Change  |
|------------------|------------|-----------|---------|
| Overall accuracy | 76.85%     | 71.67%    | ↓ 5.18% |
| REM recall       | 0.84       | 0.66      | ↓ 0.18  |

### Conclusion
**LSTM outperforms CNN on raw EEG classification.**

CNN on spectrograms lost because:
1. EEG is fundamentally temporal — LSTM's sequential 
   memory better captures brain state transitions
2. Spectrogram resolution is low — CNNs need rich 
   spatial structure to extract meaningful features
3. ImageNet features don't transfer to EEG domain —
   brain signals look nothing like natural images

### Implication for Project 07
SleepLSTM remains the primary classifier.
CNN approach may improve with higher resolution 
spectrograms or domain-specific pretraining.

### Next experiment
Experiment 003 — CNN-LSTM Hybrid
Use CNN to extract features from each EEG window,
then feed feature sequence into LSTM for temporal 
modelling. Combines spatial + temporal understanding.
Target: beat LSTM baseline of 76.85%.


## Experiment 003 — CNN-LSTM Hybrid
**Date**: July 2026  
**Model**: CNNLSTMSleepClassifier  
**Result**: 80.14% accuracy, REM F1 0.81   (New best)

### Three-way comparison
| Metric           | LSTM (001) | CNN (002) | Hybrid (003) |
|------------------|------------|-----------|--------------|
| Overall accuracy | 76.85%     | 71.67%    | **80.14%** ★ |
| REM recall       | **0.84** ★ | 0.66      | 0.83         |
| REM F1           | 0.74       | 0.65      | **0.81** ★   |
| N1 F1            | 0.35       | 0.32      | **0.47** ★   |

### Conclusion
**CNN-LSTM hybrid is the new primary classifier for Project 07.**

The CNN extracts local frequency features from 3-second windows.
The LSTM models how those features transition over 30 seconds.
Together they improve every stage except N3 (≈ same).

Most significant: N1 F1 improved from 0.35 → 0.47 (+34%).
N1 is the target trigger stage — better N1 detection means
more reliable lucid dream induction timing.

### Implication for Project 07
Replace SleepLSTM with CNNLSTMSleepClassifier as primary 
classifier. Update detect_rem.py and train_rem_model.py.

### Next experiment
Experiment 004 — Add self-attention over CNN windows.
Let the model learn WHICH 3-second windows matter most.