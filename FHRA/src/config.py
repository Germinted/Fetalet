"""Configuration settings for Fetal Heart Rate Analysis"""

import torch
from pathlib import Path

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plot"
MODEL_DIR = BASE_DIR / "models"

# Model hyperparameters
MODEL_CONFIG = {
    "in_channels": 2,
    "window_size": 100,
    "num_classes": 2,
    "num_prototypes": 5,
    "shapelet_length": 30,
    "num_candidates": 5,
    # Priority 1 improvements
    "dropout_rate": 0.25,
    "use_residual": True  # Use residual connections in CNN
}

# Model variant selection (not passed to model constructor)
USE_LIGHTWEIGHT_MODEL = False  # Set to True to use LightweightDualBranchModel instead

# Training hyperparameters
TRAINING_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 30,
    "seed": 42,
    "train_split": 0.8,
    "loss_alpha": 0.3,
    "loss_margin": 5
}

# Dataset parameters
DATASET_CONFIG = {
    "window_size": 100,
    "stride": None  # If None, uses window_size
}

# Visualization settings
VIZ_CONFIG = {
    "dpi": 600,
    "figsize": (16, 10),
    "font_size": 12,
    "n_samples": 3,
    "save_format": ["pdf", "jpg"]
}

# Ensure directories exist
for dir_path in [DATA_DIR, PLOT_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True)