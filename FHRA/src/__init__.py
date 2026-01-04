"""Fetal Heart Rate Analysis Module"""

__version__ = "1.0.0"
__author__ = "FHR Analysis Team"

from .config import *
from .dataset import FHRDataset, create_data_loaders
from .models import DualBranchModel, CandidateGenerator
from .loss import HybridLoss, ContrastivePrototypeLoss
from .training import Trainer, Evaluator, initialize_prototypes
from .visualization import FHRVisualizer
from .utils import setup_seed, count_parameters, save_model, load_model

__all__ = [
    'FHRDataset',
    'create_data_loaders',
    'DualBranchModel',
    'CandidateGenerator',
    'HybridLoss',
    'ContrastivePrototypeLoss',
    'Trainer',
    'Evaluator',
    'initialize_prototypes',
    'FHRVisualizer',
    'setup_seed',
    'count_parameters',
    'save_model',
    'load_model'
]