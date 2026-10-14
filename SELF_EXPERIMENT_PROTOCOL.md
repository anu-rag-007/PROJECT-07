# LUCID Phase 3 — Self-Experiment Protocol
# EEG-Guided Dream Imagery: A Single-Subject Study
# Author: Anurag Sharma
# Status: PLANNED — awaiting Muse S hardware

---

## Scientific question

Do images generated from REM-sleep EEG embeddings
show semantic correspondence with subjective dream
content reported immediately upon waking?

This question has never been studied.
Scotti et al. studied AWAKE subjects viewing images.
This protocol targets the SLEEPING brain.

---

## Equipment required

- Muse S EEG headband (~₹30,000)
  Channels: AF7, AF8, TP9, TP10, AUX (5 channels)
  Sampling rate: 256 Hz
  SDK: muse-lsl (Python, already in pipeline)

- Laptop running overnight:
  CNN-LSTM classifier (Phase 1, trained)
  ATM encoder (Phase 2, trained)
  ComfyUI + Stable Diffusion v1.5 (running locally)
  muse_s_stream.py (already written in pipeline)

- Dream journal (physical notebook beside bed)
  Written IMMEDIATELY on waking, before phone

---

## Protocol design

### Phase A — Baseline (Nights 1-3)
  Record sleep without any trigger or generation.
  Purpose: verify Muse S data quality, calibrate
  classifier on your own EEG, check sleep staging
  accuracy against your subjective assessment.
  
  Each morning: write what stage you think you
  were in when you woke. Compare to classifier output.

### Phase B — Detection only (Nights 4-7)
  Run classifier. Log REM detections.
  No haptic trigger. No generation.
  Purpose: establish your personal REM detection rate.
  Note: do NOT look at trigger log until morning.
  
  Each morning: write dream journal FIRST.
  Then check: did classifier detect REM?

### Phase C — Generation (Nights 8-30)
  Full pipeline active:
    REM detected → save 30s EEG epoch → haptic trigger
    Morning: generate image from saved epoch
    
  Journal protocol:
    Wake → immediately write 5+ sentences about dream
    BEFORE looking at generated image
    Then: look at image, rate correspondence 0-3:
      0 = no relation
      1 = vague/thematic relation
      2 = clear semantic overlap
      3 = specific element match
    
  Log format per night:
    date, rems_detected, trigger_time,
    dream_journal_text, generated_concept,
    correspondence_score (0-3), notes

---

## Avoiding confirmation bias

This is the hardest part of a self-experiment.

Problem: you know what was generated before rating.
Solution: blind evaluation by a second person.

Protocol:
  You write dream journal (no image).
  Generate image (save, don't look yet).
  Give journal text to a friend/family member.
  They rate the image correspondence independently.
  Compare your rating to theirs.

If independent rating consistently > 0.5:
  Something real is happening.
  That is publishable.

---

## Statistical analysis plan

Primary metric:
  Mean correspondence score across 23 nights
  (Phase C, nights 8-30)

Null hypothesis:
  Correspondence = random (score ≈ 0.5 expected
  if EEG encodes no relevant information)

Test:
  One-sample t-test: mean score vs 0.5
  Or Wilcoxon signed-rank (non-parametric)

Secondary metrics:
  Retrieval accuracy (was generated concept in
  top-5 of dream journal keyword matches?)
  
  REM detection rate vs self-reported dreaming
  nights (does classifier catch dream nights?)

Power analysis:
  23 nights × effect size 0.5 SD
  → 80% power at α=0.05 with one-sample t-test
  More nights = more power, always

---

## Data format

### Per-night log (CSV)
date, night_num, muse_battery_start,
rems_detected (count), first_rem_time (min),
trigger_fired (bool), eeg_epoch_file,
dream_journal (text), generated_concept,
generated_image_file,
self_score (0-3), blind_score (0-3), notes


### EEG files
data/self_experiment/
night_001/
raw_eeg.csv (full night Muse S stream)
rem_epochs/
rem_001_t_0430.npy (30s epoch, time 04:30)
rem_002_t_0612.npy
night_002/
...


### Generated images
data/self_experiment/generated/
night_001_rem_001_concept_cat.png
night_001_rem_002_concept_lamp.png
...


---

## Paper from this experiment

Even with N=23 nights, N=1 subject:

Title: "Self-Experiment in EEG-Guided Dream Imagery:
        Semantic Correspondence Between REM Sleep
        EEG and Generated Visual Content"

If mean correspondence > chance:
  → Any workshop or journal accepts this
  → First demonstrated REM-EEG generation result
  → Strong motivation for multi-subject study

If mean correspondence = chance:
  → Still publishable as negative result
  → "EEG during REM does not encode retrievable
     visual features at consumer-grade resolution"
  → Motivates better hardware / more channels

Either way: the experiment has value.
Run it when Muse S arrives.

---

## Timeline

Month 1-5:  Save ₹30,000 (tutoring + Internshala)
Month 6:    Purchase Muse S
Month 6:    Phase A — baseline (3 nights)
Month 7:    Phase B — detection (4 nights)
Month 7-9:  Phase C — generation (23 nights)
Month 9-10: Analysis + paper writing
Month 10:   Submit to IEEE TNSRE or NeurIPS

---

## What to do right now (before Muse S)

The pipeline is ready. The code is written.
muse_s_stream.py is already in src/.

Steps to verify the pipeline will work:

1. Test muse_s_stream.py syntax (don't need device):
   python -c "import muse_s_stream; print('OK')"

2. Simulate 5-channel Muse S EEG input:
   Create synthetic (5, 256*30) EEG array
   Run through retrained classifier version for 5 ch

3. Verify ComfyUI generation is stable overnight:
   Leave ComfyUI running for 6 hours
   Generate 10 images sequentially
   Confirm no crashes, all images save correctly

4. Test the full loop on fake data:
   Simulate REM detection → save fake epoch
   Run ATM → retrieve concept → generate image
   Verify end-to-end in under 60 seconds

When Muse S arrives: connect, calibrate, sleep.