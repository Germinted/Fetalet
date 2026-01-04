"""Utility functions for NIFEA Fetal Heart Rate Analysis"""

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


def save_model(model: torch.nn.Module, path: str, epoch: int = None, metrics: dict = None) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model to save
        path: Path to save the model
        epoch: Current epoch number (optional)
        metrics: Dictionary of metrics to save (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


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
    print(f"Model loaded from {path}")
    return model


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)


def print_model_info(model: torch.nn.Module, input_size: tuple) -> None:
    """
    Print model information including parameters and architecture.

    Args:
        model: PyTorch model
        input_size: Example input size (batch_size, channels, length)
    """
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)

    # Count parameters
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Model architecture
    print("\nModel Architecture:")
    print(model)

    # Test forward pass
    try:
        with torch.no_grad():
            dummy_input = torch.randn(input_size)
            output = model(dummy_input)
        print(f"\nInput shape: {input_size}")
        print(f"Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")
    except Exception as e:
        print(f"\nCould not perform forward pass: {e}")

    print("="*60)