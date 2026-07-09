## **# Project 07 — Classifier Main Pipeline**

## 

> First component of **\*\*LUCID: Reality?\*\*** — a **closed-loop BCI** 

> system for **automated lucid dream induction** via real-time 

> **sleep stage classification** and **haptic stimulation.**



##### **## Overview**



A complete end-to-end **Brain-Computer Interface** pipeline that:

\- Reads and preprocesses **EEG** signals (Sleep-EDF dataset)

\- Classifies **sleep stages** using a stacked **LSTM** network

\- Detects **REM** epochs in real time with safety interlocks

\- Triggers haptic feedback on a connected Android device



##### **## Results**



**Trained on Sleep-EDF Cassette dataset (20 subjects, 18,226 epochs)**



**| Stage | Precision | Recall | F1-Score |**

**|-------|-----------|--------|----------|**

**| Wake  | 0.69      | 0.54   | 0.61     |**

**| N1    | 0.32      | 0.38   | 0.35     |**

**| N2    | 0.89      | 0.78   | 0.83     |**

**| N3    | 0.84      | 0.88   | 0.86     |**

**| REM   | 0.67      | 0.84   | 0.74     |**



**\*\*Overall accuracy: 76.85%\*\***  

**\*\*REM recall: 84%\*\* — detects 84 of every 100 REM epochs**



##### **## Architecture**



**Sleep-EDF EEG (.edf)**

**↓**

**preprocess\_sleep\_edf.py   —**   bandpass filter (0.3–35Hz),

30s epoch segmentation

**↓**

**lstm\_model.py   —**   SleepLSTM: 2-layer stacked LSTM,

hidden\_dim=128, dropout=0.3

**↓**

**train\_rem\_model.py   —**   weighted cross-entropy loss,

Adam optimizer, LR scheduling

**↓**

**detect\_rem.py    —**   probability threshold (>0.6)

with 5-window smoothing

**↓**

**decision\_logic.py    —**   multi-condition safety interlock:

sustained N1, no recent Wake,

anti re-trigger logic

**↓**

**trigger\_vibrations.py  —  JOIN API → Android haptic feedback**



##### 

##### **## Dataset**



**- \*\*Source\*\*: Sleep-EDF Cassette (PhysioNet)**

**- \*\*Subjects\*\*: 20**

**- \*\*Total epochs\*\*: 18,226 (30s each)**

**- \*\*Sampling rate\*\*: 100 Hz**

**- \*\*Channel\*\*: EEG Fpz-Cz**

**- \*\*Filter\*\*: Bandpass 0.3–35 Hz (AASM standard)**



##### **## Installation**



**```bash**

**git clone https://github.com/anu-rag-007/PROJECT-07.git**

**cd PROJECT-07/classifier\_main\_pipeline**

**pip install -r requirements.txt**

**```**



##### **## Usage**



**```bash**

**# 1. Preprocess raw EEG data**

**python src/preprocess\_sleep\_edf.py**



**# 2. Train the classifier**

**python src/train\_rem\_model.py**



**# 3. Run the full closed-loop pipeline**

**python src/main.py**

**```**



##### **## Project Vision**



**Project 07** is the technical foundation of **\*\*LUCID: Reality?\*\*** —

a long-term research initiative to create a **shared dream** 

**interface** where multiple users can experience a 

collaboratively generated world during **REM** sleep.



**\*\*Current status\*\*: v0.1 prototype —** offline classification  

**\*\*Next milestone\*\*:** Real-time **EEG** stream integration



##### **## Stack**



**Python 3.13 · PyTorch · MNE-Python · NumPy · scikit-learn**



##### **## Author**



**Anurag Sharma — B.Tech CSE (AI\&ML),**

**Started Building the foundation of "LUCID: Reality?" in 1st Semester.**

