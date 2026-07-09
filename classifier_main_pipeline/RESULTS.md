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

