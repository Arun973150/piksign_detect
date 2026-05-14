"""Training pipeline for the trainable components of Section 8.1.

Trains:
  - EfficientNet-B4 spatial pathway classifier head (frozen backbone by default)
  - Frequency-domain filter-bank head (fixed kernels + small CNN)
  - Attention gate (W_i, b_i per Eq. 11)
  - Final binary classifier

Inference-time integration with the main detector lives in the orchestrator;
this folder is self-contained and only produces a .pth checkpoint.
"""
