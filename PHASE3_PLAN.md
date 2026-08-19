# LUCID Phase 3 — 3D Dream World

## Technical Pipeline

1. EEG (Phase 1) -> Sleep stage + features
2. SD (Phase 2)  -> Dream images (2D, multiple)
3. Zero-1-to-3   -> Novel views from single image
4. Instant-NGP   -> 3D NeRF from views
5. VR rendering  -> Navigable dream world

## Tools
- Zero-1-to-3: github.com/cvlab-columbia/zero123
- Instant-NGP:  github.com/NVlabs/instant-ngp
- Gaussian Splatting: graphdeco-inria/gaussian-splatting
- DreamFusion (text -> 3D): github.com/ashawkey/stable-dreamfusion

## Status
Phase 1: ✅ Complete (CNN-LSTM, κ=0.68, paper published)
Phase 2: ✅ Prototype (SD local, EEG-conditioned imagery)
Phase 3: ⬜ Planned (Zero-1-to-3 + Instant-NGP)

## Blockers
- GPU access needed for Zero-1-to-3 (free on Colab)
- THINGS-EEG dataset for true EEG conditioning
- Muse S headband for real EEG input

## Timeline
Month 1-2: THINGS-EEG training (Phase 2 completion)
Month 3:   Zero-1-to-3 integration
Month 4:   Instant-NGP pipeline  
Month 5:   First navigable dream world demo
Month 6:   Paper submission
