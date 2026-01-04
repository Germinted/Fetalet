"""Dataset classes for Fetal Heart Rate Analysis"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict

# Import noise factory for noise generation
try:
    from .noise import NoiseFactory
except ImportError:
    # If running as a script, adjust import
    from noise import NoiseFactory


class FHRDataset(Dataset):
    """
    Dataset class for Fetal Heart Rate (FHR) and TOCO signals.

    This dataset loads CSV files containing FHR, TOCO signals and labels,
    then creates sliding windows for time-series classification.
    Optionally adds various types of noise to the signals.
    """

    def __init__(self,
                 data_dir: str,
                 window_size: int = 100,
                 stride: Optional[int] = None,
                 add_noise: bool = False,
                 noise_types: Optional[List[str]] = None,
                 noise_params: Optional[Dict] = None):
        """
        Initialize the FHR dataset.

        Args:
            data_dir: Directory containing CSV files
            window_size: Size of each time window
            stride: Stride between windows (if None, uses window_size)
            add_noise: Whether to add noise to the signals
            noise_types: List of noise types to add (e.g., ['gaussian', 'baseline_drift'])
            noise_params: Dictionary of parameters for each noise type
        """
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.add_noise = add_noise
        self.noise_types = noise_types if noise_types else []
        self.noise_params = noise_params if noise_params else {}
        self.segments = []

        # Load and combine all CSV files
        self._load_data(data_dir)

        # Create sliding windows
        self._create_windows()

    def _load_data(self, data_dir: str) -> None:
        """Load and preprocess all CSV files in the directory."""
        all_data = []

        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)

                # Handle missing values
                df.fillna(method='ffill', inplace=True)
                df.fillna(0, inplace=True)

                all_data.append(df)

        if not all_data:
            raise ValueError(f"No CSV files found in {data_dir}")

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.sort_values("timestamp", inplace=True)
        self.data = combined_data

    def _create_windows(self) -> None:
        """Create sliding windows from the combined data."""
        data_arr = self.data[["FHR", "TOCO"]].values.astype(np.float32)
        labels_arr = self.data["label"].values.astype(np.int64)
        n_samples = data_arr.shape[0]

        for start in range(0, n_samples - self.window_size + 1, self.stride):
            end = start + self.window_size
            window_data = data_arr[start:end, :]

            # Create window label (1 if any abnormal beat in window, else 0)
            window_label = 1 if np.any(labels_arr[start:end]) == 1 else 0

            # Store as (channels, window_size) format
            self.segments.append((window_data.T, window_label))

    def __len__(self) -> int:
        """Return the number of segments in the dataset."""
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a specific segment from the dataset.

        Args:
            idx: Index of the segment

        Returns:
            Tuple of (segment_tensor, label_tensor)
        """
        segment, label = self.segments[idx]
        segment_tensor = torch.tensor(segment, dtype=torch.float32)

        # Add noise if requested
        if self.add_noise and self.noise_types:
            # Convert to numpy for noise addition
            segment_np = segment_tensor.numpy()
            # Apply noise
            noisy_segment = NoiseFactory.apply_multiple_noises(
                segment_np,
                self.noise_types,
                self.noise_params
            )
            # Convert back to tensor
            segment_tensor = torch.tensor(noisy_segment, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return segment_tensor, label_tensor

    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary with class counts
        """
        labels = [segment[1] for segment in self.segments]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def create_data_loaders(dataset: FHRDataset,
                       batch_size: int = 64,
                       train_split: float = 0.8,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders.

    Args:
        dataset: FHRDataset instance
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        seed: Random seed for reproducible split

    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Ensure consistent batch sizes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


class FHRDatasetWithNoise(FHRDataset):
    """
    Extended FHR Dataset with built-in noise support.

    This class inherits from FHRDataset and adds methods specifically
    for handling noise scenarios in fetal heart rate analysis.
    """

    @classmethod
    def create_noisy_variants(cls,
                              base_dataset: FHRDataset,
                              noise_scenarios: List[Dict]) -> List['FHRDatasetWithNoise']:
        """
        Create multiple noisy variants of a base dataset.

        Args:
            base_dataset: Original clean dataset
            noise_scenarios: List of noise scenario dictionaries

        Returns:
            List of FHRDatasetWithNoise instances with different noise
        """
        noisy_datasets = []

        for scenario in noise_scenarios:
            noisy_dataset = cls(
                data_dir="",  # Will reuse base_dataset's data
                window_size=base_dataset.window_size,
                stride=base_dataset.stride,
                add_noise=True,
                noise_types=scenario['noise_types'],
                noise_params=scenario['params']
            )
            # Copy the segments from base dataset
            noisy_dataset.segments = base_dataset.segments.copy()
            noisy_dataset.data = base_dataset.data

            noisy_datasets.append(noisy_dataset)

        return noisy_datasets