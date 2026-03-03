"""
config.py
---------
Centralized project configuration. All paths, hyperparameters,
and shared constants live here so nothing is hardcoded twice.
"""

import os

# ── paths ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
DB_PATH      = os.path.join(PROJECT_ROOT, "evaluation.db")

# ── model training ─────────────────────────────────────────────
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
EPOCHS        = 30
VAL_SPLIT     = 0.2
TEST_SPLIT    = 0.1
LEARNING_RATE = 1e-4
SEED          = 42

# subset of classes for quick testing; set to None to use all 38
CLASSES_TO_USE = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy",
]

# ── GUI theme ──────────────────────────────────────────────────
BG_DARK      = "#0A1628"
BG_CARD      = "#132040"
BG_SIDEBAR   = "#0E1A30"
ACCENT       = "#00C896"
ACCENT_HOVER = "#00A57A"
ACCENT_RED   = "#E05A5A"
RED_HOVER    = "#C44444"
TEXT_WHITE   = "#F0F4FF"
TEXT_MUTED   = "#8899BB"
BORDER       = "#1E3560"
INPUT_BG     = "#0D1F3C"

FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_HEAD  = ("Segoe UI", 14, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_ENTRY = ("Segoe UI", 12)
FONT_BTN   = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 9)

# make sure output dirs exist on import
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
