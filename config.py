"""
config.py — Central configuration for the RepE pipeline.
"""

import os
import torch

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Which device to run on. "auto" will pick CUDA if available, else CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data type — float16 on GPU, float32 on CPU for stability
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ──────────────────────────────────────────────
# Representation Engineering Configuration
# ──────────────────────────────────────────────
# Layer index to extract activations from (0-indexed).
# For Llama-3.2-1B with 16 layers, layer 8 (middle) works well.
# The "sweet spot" is typically 40-60% of the way through the network.
TARGET_LAYER = 8

# How many PCA components to compute (we only use the 1st for steering)
N_PCA_COMPONENTS = 5

# Default steering coefficient.
# Positive = amplify the concept, Negative = suppress / invert it.
DEFAULT_STEERING_COEFF = 3.0

# Token position to extract activations from.
# -1 = last token (where the model "summarizes" the prompt meaning)
TOKEN_POSITION = -1

# ──────────────────────────────────────────────
# Generation Configuration
# ──────────────────────────────────────────────
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VECTORS_DIR = os.path.join(PROJECT_ROOT, "vectors")
os.makedirs(VECTORS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Web Dashboard
# ──────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
