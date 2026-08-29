# Project 07 — LUCID: Reality? Roadmap
*Last updated: August 2026*

## What's done

### Phase 1 — EEG Sleep Staging ✅
- CNN-LSTM classifier: κ=0.67 LOSO, 80.14% accuracy
- 153-subject Sleep-EDF preprocessing pipeline
- Closed-loop haptic trigger (working on phone)
- Published: https://doi.org/10.5281/zenodo.21885881
- arXiv submission: in progress

---

## What's next — concrete and ordered

### Next 30 days

**Week 11: THINGS-EEG + Phase 2 training**
- Download THINGS-EEG (~10GB)
- Adapt CNN-LSTM for shorter windows (0.5s vs 30s)
- Train EEGAlignmentMLP on Colab (free GPU)
- Target: top-5 retrieval accuracy > 30%

**Week 12: Connect to ComfyUI**
- Replace text conditioning with EEG embedding
- Use IP-Adapter for direct embedding injection
- Generate first image from EEG features

**Week 13: Paper 2 draft**
- "Towards Dream Imagery Reconstruction: 
   EEG-guided Latent Diffusion during REM Sleep"
- Methods: THINGS-EEG pre-training + REM transfer
- Submit to NeurIPS ML4H Workshop (deadline: October)

**Week 14: Semester exam preparation**
- Maintain CGPA alongside research
- Research: background tasks only

---

### Next 6 months

**Month 2: Get wearable EEG**
- Save ₹30,000 via tutoring + freelancing
- Purchase Muse S headband
- Estimated: 5-6 months at ₹5,000/month

**Month 3: Real-time integration**
- Replace EEGStreamSimulator with MuseSStream
- Two lines of code change
- First live test: classify own sleep stages

**Month 4: First self-experiment**
- Sleep with Muse S for 5 nights (data only)
- Night 6-30: trigger enabled
- Log: trigger time, morning dream recall
- Target: any correlation between trigger and lucidity

**Month 5: Phase 2 live**
- Real EEG during sleep → dream image generation
- Generate image DURING sleep epoch
- Show it to subject next morning
- "Does this look like what you dreamed?"

**Month 6: Paper 2 submission**
- Results from self-experiment
- Even one confirmed reconstruction is publishable
- IEEE TNSRE or NeurIPS workshop

---

### Next 2 years

**Semester 3-4: MS/PhD preparation**
- Build research portfolio (2 papers minimum)
- Target: IISc Computational Neuroscience
           IIIT Hyderabad CVIT
           International: CMU NeuralLab, MIT BCS
- GRE if targeting abroad

**Year 2: Phase 3 prototype**
- Zero-1-to-3: 2D dream image → multiple views
- Instant-NGP: views → 3D NeRF scene
- Goal: first navigable dream environment
- Paper 3: "Neural 3D dream reconstruction"

**Year 2-3: Find collaborators**
- Sleep lab at AIIMS or equivalent
- Access to proper PSG equipment
- Multi-subject study (not just self)
- Regulatory guidance on human subjects research

---

### Next 10-15 years

Phase 4: Multi-user shared dream world
Phase 5: Artificial Reality

*These phases depend on technologies that are
converging but not yet ready:
- High-res consumer BCI (3-5 years away)
- Real-time neural rendering at 60fps (2-3 years)
- Regulatory framework for neural interfaces
The technical path is clear. The timeline is honest.*

---

## The number that changes everything

When you can show someone:

"Here is the EEG recording from your REM sleep.
 Here is the image the model generated from it.
 Here is your dream journal from that night."

And the three correlate — even loosely — that is
the moment LUCID becomes undeniable.

Everything until then is building toward that moment.