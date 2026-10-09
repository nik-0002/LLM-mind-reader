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

# How the reading direction is computed from the contrastive difference vectors:
#   "pca"       - RepE-style PCA with random pair-order flips (recommended, matches the paper)
#   "mean_diff" - difference of means (simple, robust baseline)
#   "pca_raw"   - naive centered PCA (kept only for comparison; it discards the mean shift)
EXTRACTION_METHOD = "pca"

# Seed for PCA sign-flips, train/test splits, random control vectors and sampling.
RANDOM_SEED = 42

# Default steering coefficient, in units of the concept's typical (positive - negative) gap.
#   +1.0 ~ shifts the activation by one typical "negative -> positive" difference.
#   Sensible sweep range is roughly -2 ... +2; beyond ~3 the text usually degenerates.
# (In v2 the vector was a bare unit vector, so "3.0" did almost nothing at mid-layers.)
DEFAULT_STEERING_COEFF = 1.0
COMPARE_COEFFICIENTS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(VECTORS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Web Dashboard
# ──────────────────────────────────────────────
# Bind to localhost by default - this server can run arbitrary prompts on your GPU.
FLASK_HOST = os.environ.get("REPE_HOST", "127.0.0.1")
FLASK_PORT = 5000
