# Low-Rank Model Compression (Exercise #1)

Implements the MNIST dense baseline (784-100-50-10) and SVD-based low-rank compression at 2x, 4x, 8x factors with refinement training.

## Steps Implemented
1. Baseline training (100 epochs default) with per-epoch metrics + confusion matrix.
2. SVD factorization of each Dense weight: rank-k (k=int(n/factor)).
3. Compressed model construction (projection + reconstruction layers) and 10-epoch refinement.
4. Multiple compression factors (2x,4x,8x) with parameter counts, metrics, confusion matrices.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Fast sanity run
BASELINE_EPOCHS=1 COMPRESS_EPOCHS=1 python test.py
# Full assignment run
python test.py
```

## Environment Overrides
- `BASELINE_EPOCHS` (default 100)
- `COMPRESS_EPOCHS` (default 10)

## Output
Console prints:
- Baseline model summary & parameter count.
- Per-epoch metrics (loss/accuracy + val_*) for baseline & each compressed model.
- Confusion matrices (10x10) for baseline and each compression factor.
- Parameter ratios vs baseline.

## File Overview
- `test.py`: Complete pipeline with modular step functions.
- `requirements.txt`: Dependencies.
- `CS 510 Programming 5 Summer 2025.pdf`: Original assignment PDF (not modified).

## Notes
- Biases are copied from baseline into reconstruction layers.
- Projection layers are linear and bias-free.
- GPU is auto-disabled if an incompatible TensorFlow wheel is detected.

## Optional Enhancements
Potential extras (not required by assignment): result JSON export, accuracy vs. compression plot, early stopping, rank auto-tuning.
