"""
Configuration module for Fetal Heart Rate Anomaly Detection with Noise.
Contains hyperparameters and settings for the model training and evaluation.
"""

import torch

class DataConfig:
    """Data configuration parameters."""
    # Data paths
    DATA_DIR = "FHRA/data"
    PLOT_DIR = "FHRA/plot"

    # Data parameters
    WINDOW_SIZE = 100
    STRIDE = None  # If None, uses window_size
    IN_CHANNELS = 2  # FHR and TOCO channels

    # Train/test split
    TRAIN_RATIO = 0.8

    # Noise scenarios for testing
    NOISE_SCENARIOS = [
        {
            'name': 'Gaussian Noise',
            'noise_types': ['gaussian'],
            'params': {'gaussian': {'noise_level': 0.3}}
        },
        {
            'name': 'Baseline Drift',
            'noise_types': ['baseline_drift'],
            'params': {'baseline_drift': {'frequency': 0.01, 'amplitude': 5}}
        },
        {
            'name': 'Power Line Interference',
            'noise_types': ['power_line'],
            'params': {'power_line': {'frequency': 50, 'amplitude': 1.0, 'sampling_rate': 4}}
        },
        {
            'name': 'Impulse Noise',
            'noise_types': ['impulse'],
            'params': {'impulse': {'probability': 0.02, 'amplitude': 10}}
        },
        {
            'name': 'Combined Noise (Gaussian + Baseline Drift)',
            'noise_types': ['gaussian', 'baseline_drift'],
            'params': {
                'gaussian': {'noise_level': 0.3},
                'baseline_drift': {'frequency': 0.01, 'amplitude': 3}
            }
        },
        {
            'name': 'Severe Combined Noise',
            'noise_types': ['gaussian', 'power_line'],
            'params': {
                'gaussian': {'noise_level': 0.3},
                'power_line': {'frequency': 50, 'amplitude': 0.5, 'sampling_rate': 4}
            }
        }
    ]

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
    MARGIN = 5.0  # Margin for prototype loss
    EPS = 1e-8  # Numerical stability

class TrainingConfig:
    """Training configuration parameters."""
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 30
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
    FIGURE_DPI = 600
    FIGURE_SIZE = (16, 10)
    FONT_SIZE = 12
    FONT_FAMILY = 'Arial'

    # Color scheme
    COLORS = {
        'fhr': '#1f78b4',      # Blue
        'toco': '#e31a1c',     # Red
        'shapelet': '#33a02c',  # Green
        'highlight': '#fdbf6f'  # Orange
    }

    # Line styles
    LINESTYLES = {
        'fhr': '-',
        'toco': '-',
        'shapelet': '--'
    }

    # Signal processing
    GAUSSIAN_SIGMA = 1.5
    CANDIDATE_SIGMA = 0.5

    # Output formats
    SAVE_FORMATS = ['pdf', 'jpg']

# Combine all configs
class Config:
    """Main configuration class combining all sub-configs."""
    DATA = DataConfig()
    MODEL = ModelConfig()
    TRAINING = TrainingConfig()
    VIZ = VisualizationConfig()