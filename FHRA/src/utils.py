"""Utility functions for Fetal Heart Rate Analysis"""

import random
import numpy as np
import torch


def setup_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: torch.nn.Module, path: str, epoch: int = None) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model to save
        path: Path to save the model
        epoch: Current epoch number (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    if epoch is not None:
        checkpoint['epoch'] = epoch

    torch.save(checkpoint, path)


def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        model: PyTorch model to load weights into
        path: Path to the checkpoint
        device: Device to load the model on

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)