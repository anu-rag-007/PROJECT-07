# PROJECT-07 — LUCID: Reality?

> **Closed-loop EEG sleep-stage classification and haptic stimulation pipeline for the LUCID: Reality? research project.**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-EEG%20Processing-8A2BE2)](https://mne.tools/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PROJECT-07 is the first major technical component of **LUCID: Reality?**, a long-term neurotechnology research project exploring whether EEG-based sleep-stage detection can be combined with controlled haptic stimulation to support lucid-dream induction.

[![Phase 1 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21885881.svg)](https://doi.org/10.5281/zenodo.21885881)
[![Phase 2 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23097798.svg)](https://doi.org/10.5281/zenodo.23097798)

The repository currently contains the EEG preprocessing, sleep-stage classification, REM detection, decision/safety logic, haptic triggering pipeline, research documentation, experimental results, figures, and the roadmap toward EEG-to-dream-imagery reconstruction.

---

## Table of Contents

- [Published Work](#published-work)
- [Project Overview](#project-overview)
- [What Has Been Built](#what-has-been-built)
- [System Architecture](#system-architecture)
- [Current Results](#current-results)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Core Components](#core-components)
- [Research & Documentation](#research--documentation)
- [Project Roadmap](#project-roadmap)
- [Current Status](#current-status)
- [Limitations & Research Notes](#limitations--research-notes)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)
- [Author](#author)

---

## Published Work

### Phase 1 — Automated Sleep Stage Classification
**"Automated Sleep Stage Classification for Closed-Loop
Lucid Dream Induction via CNN-LSTM on Single-Channel EEG"**

📄 [Paper (Zenodo)](https://doi.org/10.5281/zenodo.21885881)

- CNN-LSTM hybrid on Sleep-EDF Extended (153 recordings)
- κ = 0.68 random split · κ = 0.67 LOSO (20 subjects)
- REM F1 = 0.81 · closed-loop haptic trigger validated
- 177,411 epochs · single-channel EEG (Fpz-Cz)

### Phase 2 — EEG-to-CLIP Contrastive Alignment
**"Contrastive EEG-to-CLIP Alignment on THINGS-EEG:
An Ablation of Target Modality, Temperature,
and Channel Count"**

📄 [Paper (Zenodo)](https://doi.org/10.5281/zenodo.23097798)

- ATM encoder: 626K params · 17-channel EEG → 512-dim CLIP
- Best: Image CLIP targets · τ=0.05 · Top-5 = 4.5% (1.8× chance)
- Ablation: text vs image targets · temperature · channel count
- Finding: P3 (left parietal) highest attention weight
- Dataset: THINGS-EEG (50 subjects · 1,654 concepts)


## Project Overview

The long-term goal of **LUCID: Reality?** is to investigate a closed-loop brain-computer interface in which neural activity during sleep can be detected, interpreted, and eventually used to interact with or reconstruct dream content.

PROJECT-07 focuses on the first part of that vision:

```text
EEG Signal
    │
    ▼
Sleep-EDF Preprocessing
    │
    ▼
Sleep-Stage Classification
    │
    ▼
REM Detection
    │
    ▼
Safety / Decision Logic
    │
    ▼
Haptic Trigger
    │
    ▼
Android Device
```

The current implementation establishes the foundation for moving from **offline EEG classification** toward a future **real-time closed-loop system**.

---

## What Has Been Built

### 1. EEG preprocessing

The pipeline processes Sleep-EDF EEG recordings using MNE-Python.

Current preprocessing includes:

- EDF signal loading
- EEG channel selection
- Band-pass filtering
- 30-second sleep epochs
- Sleep-stage label alignment
- Dataset preparation for deep-learning models
- Dataset/class-distribution analysis

The current documented pipeline uses the **Fpz-Cz** EEG channel and a **0.3–35 Hz** band-pass filter.

### 2. Sleep-stage classification

A deep-learning sleep classifier has been implemented using a CNN-LSTM / stacked LSTM approach.

The classifier is designed to distinguish:

- Wake
- N1
- N2
- N3
- REM

The repository contains the model, training pipeline, evaluation artifacts, logs, plots, and research results.

### 3. REM detection

REM detection is treated as the critical event for the closed-loop stimulation stage.

The pipeline includes:

- REM probability estimation
- Configurable probability thresholding
- Temporal smoothing
- Multi-window confirmation
- Logic designed to reduce false triggers

The current classifier documentation reports **84% REM recall**, meaning the evaluated system identified 84 out of 100 REM epochs in that reported evaluation.

### 4. Safety / decision logic

The stimulation stage is not intended to trigger simply because one epoch is classified as REM.

The decision layer incorporates conditions such as:

- sustained sleep-state evidence
- recent Wake-state checks
- anti-retrigger logic
- REM probability thresholding
- temporal smoothing

This creates an additional control layer between the neural classifier and physical stimulation.

### 5. Closed-loop haptic trigger

A working haptic-trigger component connects the classification pipeline to an Android phone.

The current implementation uses the **JOIN API** to send a vibration/haptic command to the connected device.

This establishes the basic closed-loop concept:

```text
EEG → Classifier → Decision Logic → Trigger → Phone Vibration
```

### 6. Research artifacts

The repository also contains research-oriented material including:

- experimental results
- training/evaluation logs
- generated plots
- paper figures
- research report
- research roadmap
- Phase 3 planning document
- licensing and reproducibility documentation

---

## System Architecture

### High-level pipeline

```text
┌──────────────────────┐
│   Sleep-EDF / EEG    │
│       .edf data      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ EEG Preprocessing    │
│ • MNE-Python         │
│ • 0.3–35 Hz filter   │
│ • 30 s epochs        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Sleep Classifier     │
│ CNN / Stacked LSTM   │
│ • Wake               │
│ • N1                 │
│ • N2                 │
│ • N3                 │
│ • REM                │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ REM Detection        │
│ • Probability       │
│ • Thresholding       │
│ • Temporal smoothing │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Decision / Safety    │
│ • State confirmation│
│ • Wake check         │
│ • Anti-retrigger     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Haptic Trigger       │
│ JOIN → Android       │
└──────────────────────┘
```

---

## Current Results

The repository currently documents the following evaluation for the Sleep-EDF Cassette pipeline:

| Sleep Stage | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Wake | 0.69 | 0.54 | 0.61 |
| N1 | 0.32 | 0.38 | 0.35 |
| N2 | 0.89 | 0.78 | 0.83 |
| N3 | 0.84 | 0.88 | 0.86 |
| REM | 0.67 | 0.84 | 0.74 |

**Reported overall accuracy:** 76.85%

**Reported REM recall:** 84%

The broader project roadmap also records a separate LOSO evaluation of the CNN-LSTM work with **80.14% accuracy** and **κ = 0.67**. These figures should be interpreted as results from different evaluation configurations rather than as one combined metric.

---

## Dataset

The primary dataset used by the classifier pipeline is **Sleep-EDF**.

### Current documented dataset configuration

| Property | Value |
|---|---|
| Dataset | Sleep-EDF Cassette |
| Source | PhysioNet |
| Subjects | 20 in the documented classifier run |
| Epoch length | 30 seconds |
| Sampling rate | 100 Hz |
| EEG channel | Fpz-Cz |
| Filter | 0.3–35 Hz |
| Classes | Wake, N1, N2, N3, REM |
| Documented epochs | 18,226 |

The broader roadmap also describes a larger **153-subject Sleep-EDF preprocessing pipeline** used in the research program. The exact subject count therefore depends on the experiment/configuration being referenced.

> **Important:** Dataset files are not expected to be committed to this repository. Obtain the appropriate Sleep-EDF data from its official source and follow its licensing/data-use requirements.

---

## Repository Structure

```text
PROJECT-07/
│
├── classifier_main_pipeline/
│   ├── logs/
│   ├── models/
│   ├── paper_figures/
│   ├── plots/
│   │
│   ├── src/
│   │   ├── preprocess_sleep_edf.py
│   │   ├── lstm_model.py
│   │   ├── train_rem_model.py
│   │   ├── detect_rem.py
│   │   ├── decision_logic.py
│   │   └── main.py
│   │
│   ├── class_distribution.py
│   ├── trigger_vibrations.py
│   ├── RESULTS.md
│   ├── RESEARCH_REPORT.txt
│   ├── requirements.txt
│   └── README.md
│
├── Research Papers/
│   └── Phase 2/
│
├── PHASE3_PLAN.md
├── ROADMAP.md
├── LICENSE
└── README.md
```

> The structure above reflects the repository's current organization and the main documented pipeline components. Individual experimental files and generated artifacts may evolve as development continues.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/anu-rag-007/PROJECT-07.git
cd PROJECT-07/classifier_main_pipeline
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Preprocess EEG

```bash
python src/preprocess_sleep_edf.py
```

This prepares the raw Sleep-EDF recordings for model training/evaluation.

### Step 2 — Train the classifier

```bash
python src/train_rem_model.py
```

The training pipeline uses the configured deep-learning architecture and training parameters.

### Step 3 — Run the pipeline

```bash
python src/main.py
```

The full pipeline connects preprocessing/model inference with REM detection, decision logic, and the haptic trigger stage.

> **Hardware note:** The haptic component requires a configured Android device and the corresponding JOIN integration. Offline model inference can be developed/tested independently.

---

## Core Components

### `preprocess_sleep_edf.py`

Responsible for:

- loading EDF recordings
- filtering EEG
- segmenting signals into epochs
- preparing sleep-stage labels
- generating model-ready data

### `lstm_model.py`

Contains the neural-network architecture used for sleep-stage classification.

The documented architecture includes a **2-layer stacked LSTM**, with:

- hidden dimension: 128
- dropout: 0.3

### `train_rem_model.py`

Handles model training, including:

- weighted loss
- Adam optimization
- learning-rate scheduling
- model checkpointing/evaluation

### `detect_rem.py`

Converts model output into REM detection decisions using:

- REM probability
- configurable thresholding
- temporal smoothing

The documented implementation uses a probability threshold above **0.6** and five-window smoothing.

### `decision_logic.py`

Provides the safety/decision layer between classification and stimulation.

Its documented logic includes:

- sustained-state checks
- recent Wake-state protection
- anti-retrigger behavior

### `trigger_vibrations.py`

Connects the software pipeline to an Android device through JOIN and issues the haptic/vibration command.

---

## Research & Documentation

The repository is structured not only as a software prototype but also as a research workspace.

### Included research material

- **`RESULTS.md`** — experimental/evaluation results
- **`RESEARCH_REPORT.txt`** — research notes/report
- **`paper_figures/`** — figures prepared for research communication
- **`Research Papers/Phase 2/`** — Phase 2 research material
- **`ROADMAP.md`** — long-term project roadmap
- **`PHASE3_PLAN.md`** — Phase 3 development plan

The project roadmap currently describes a progression from EEG sleep staging toward EEG-to-dream-imagery reconstruction using THINGS-EEG, EEG embeddings, image-generation systems, and eventually 3D dream-environment reconstruction.

---

## Project Roadmap

### Phase 1 — EEG Sleep Staging

**Status: Substantially implemented**

- [x] Sleep-EDF preprocessing
- [x] Sleep-stage classification
- [x] REM detection
- [x] Evaluation and metrics
- [x] Haptic trigger
- [x] Closed-loop prototype foundation
- [x] Research publication/archival work

### Phase 2 — EEG-to-Dream Imagery

Planned/ongoing work includes:

- [ ] THINGS-EEG integration
- [ ] Short-window EEG representation learning
- [ ] EEG alignment/embedding model
- [ ] EEG-conditioned image generation
- [ ] ComfyUI integration
- [ ] IP-Adapter-based conditioning
- [ ] Dream-imagery reconstruction experiments

### Phase 3 — 3D Dream Reconstruction

Planned research direction:

```text
EEG
 │
 ▼
Dream Image
 │
 ▼
Multi-view Reconstruction
 │
 ▼
3D Scene / NeRF
 │
 ▼
Navigable Dream Environment
```

The current Phase 3 plan explores technologies such as Zero-1-to-3 and Instant-NGP as potential components of this future stage.

### Long-term vision

```text
Phase 1
EEG Sleep Staging
        ↓
Phase 2
EEG → Dream Imagery
        ↓
Phase 3
Dream Imagery → 3D Environment
        ↓
Phase 4
Multi-user Shared Dream World
        ↓
Phase 5
Artificial Reality
```

---

## Current Status

**Repository:** PROJECT-07  
**Project:** LUCID: Reality?  
**Development stage:** Research prototype / v0.1 foundation

### Working / implemented

- EEG preprocessing pipeline
- Sleep-stage classification
- REM detection
- Decision/safety logic
- Haptic trigger integration
- Offline end-to-end pipeline
- Evaluation and research artifacts
- Research roadmap

### Next major milestone

**Real-time EEG stream integration.**

The long-term implementation is intended to replace offline EEG recordings with a live EEG source and evaluate the classifier under real-time sleep conditions.

---

## Limitations & Research Notes

This repository represents an experimental research prototype, not a clinically validated medical or sleep-monitoring device.

Important limitations include:

1. **Offline-first evaluation**  
   Much of the current pipeline is evaluated using recorded EEG rather than continuous live EEG.

2. **Generalization**  
   Sleep-stage performance can vary substantially across subjects and recording environments.

3. **N1 classification difficulty**  
   The reported evaluation shows considerably lower N1 performance than N2/N3.

4. **REM detection is not equivalent to lucid dreaming**  
   Detecting REM does not guarantee that a subject is lucid.

5. **Haptic stimulation requires further validation**  
   The presence of a working phone trigger demonstrates the technical pathway, not the efficacy or safety of lucid-dream induction.

6. **Human experimentation requires appropriate oversight**  
   Future studies involving human subjects, sleep manipulation, or neurotechnology should follow applicable ethical, institutional, and regulatory requirements.

7. **Dream reconstruction remains an open research problem**  
   The Phase 2/3 roadmap describes research goals rather than capabilities already achieved by the current repository.

---

## Future Work

The immediate technical priorities are:

- Real-time EEG streaming
- Robust subject-independent evaluation
- Improved N1/REM classification
- Calibration and threshold optimization
- Real-time inference latency measurement
- More robust safety interlocks
- Wearable EEG integration
- Controlled experimental validation
- THINGS-EEG representation learning
- EEG-to-image reconstruction
- 3D dream-environment reconstruction

The ultimate research question is whether measurable neural activity during sleep can provide enough information to reconstruct meaningful aspects of dream imagery.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{sharma2026lucid,
  author    = {Sharma, Anurag},
  title     = {Automated Sleep Stage Classification
               for Closed-Loop Lucid Dream Induction
               via CNN-LSTM on Single-Channel EEG},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21885881},
  url       = {https://doi.org/10.5281/zenodo.21885881}
}
```

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

## Author

**Anurag Sharma**

B.Tech CSE (AI & ML)

PROJECT-07 is part of the broader **LUCID: Reality?** research initiative.

The project was started as an early-stage exploration of EEG, artificial intelligence, brain-computer interfaces, sleep-stage classification, and the possibility of interacting with dream experiences through closed-loop systems.

---

## Acknowledgement

This project builds upon publicly available EEG/sleep research resources and open-source scientific software, including:

- Sleep-EDF / PhysioNet
- MNE-Python
- PyTorch
- NumPy
- scikit-learn
- Android/automation tooling used for haptic experimentation

---

## Disclaimer

**LUCID: Reality? and PROJECT-07 are research and educational projects.**

Nothing in this repository should be interpreted as medical advice, a clinically validated diagnostic system, or a proven method for inducing lucid dreams.

The current implementation demonstrates a technical research pipeline and should not be used for unsupervised human experimentation.

---

### Project Links

- Repository: https://github.com/anu-rag-007/PROJECT-07
- DOI: https://doi.org/10.5281/zenodo.21885881
