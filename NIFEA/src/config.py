"""
Configuration module for NIFEA Fetal Heart Rate Analysis.
Contains hyperparameters and settings for the model training and evaluation.
"""

import torch

class DataConfig:
    """Data configuration parameters."""
    # Data paths
    DATA_DIR = "NIFEA/data"

    # Data parameters
    WINDOW_SIZE = 100
    STRIDE = None  # If None, uses window_size
    IN_CHANNELS = 4  # NIFEA has 4 channels from WFDB files

    # Train/test split
    TRAIN_RATIO = 0.8

    # WFDB specific settings
    MIN_RECORD_LENGTH = 100  # Skip records shorter than window_size

class ModelConfig:
    """Model architecture configuration."""
    # Prototype and shapelet parameters
    NUM_PROTOTYPES = 5
    SHAPELET_LENGTH = 30
    NUM_CANDIDATES = 5

    # Network architecture
    CNN_CHANNELS = [16, 32]
    CNN_KERNEL_SIZES = [5, 3]
    FC_HIDDEN_DIMS = [128, 64]
    NUM_CLASSES = 2

    # Loss parameters
    ALPHA = 0.3  # Weight for prototype loss
    MARGIN = 1.0  # Margin for prototype loss
    EPS = 1e-8  # Numerical stability

class TrainingConfig:
    """Training configuration parameters."""
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    OPTIMIZER = 'Adam'

    # Training settings
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prototype initialization
    INIT_PROTOTYPES = True
    N_INIT_PROTOTYPES = 5

class VisualizationConfig:
    """Visualization configuration parameters."""
    # Plot styling
    FIGURE_DPI = 300
    FIGURE_SIZE = (10, 4)
    FONT_SIZE = 12
    FONT_FAMILY = 'DejaVu Sans'  # Ensures support for various symbols

    # Color scheme
    COLORS = {
        'signal': '#1f77b4',      # Blue
        'candidate': '#2ca02c',   # Green
        'highlight': '#FFA500',   # Orange-yellow
        'background': '#f7f7f7'   # Light gray
    }

    # Line styles
    LINESTYLES = {
        'signal': '-',
        'candidate': '--',
        'highlight': ':'
    }

    # Signal processing
    GAUSSIAN_SIGMA = 2.0
    CANDIDATE_SIGMA = 1.0

# Combine all configs
class Config:
    """Main configuration class combining all sub-configs."""
    DATA = DataConfig()
    MODEL = ModelConfig()
    TRAINING = TrainingConfig()
    VIZ = VisualizationConfig()