"""Dataset classes for NIFEA Fetal Heart Rate Analysis"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import wfdb


class NIFEADataset(Dataset):
    """
    Dataset class for NIFEA (Non-Invasive Fetal ECG Analysis) dataset.

    This dataset loads WFDB .dat/.hea files containing ECG signals,
    then creates sliding windows for time-series classification.
    """

    def __init__(self,
                 data_dir: str,
                 window_size: int = 100,
                 stride: Optional[int] = None):
        """
        Initialize the NIFEA dataset.

        Args:
            data_dir: Directory containing WFDB files (.dat, .hea)
            window_size: Size of each time window
            stride: Stride between windows (if None, uses window_size)
        """
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.segments = []

        # Load and process WFDB records
        self._load_wfdb_records(data_dir)

        # Create sliding windows
        self._create_windows()

    def _load_wfdb_records(self, data_dir: str) -> None:
        """Load all WFDB records from the directory."""
        self.records = []
        self.data_dir = data_dir  # Store data directory path

        # Get all unique record names (remove .dat extension)
        for file in os.listdir(data_dir):
            if file.endswith(".dat"):
                record_name = os.path.splitext(file)[0]
                self.records.append(record_name)

        print(f"Found {len(self.records)} records in {data_dir}")

    def _create_windows(self) -> None:
        """Create sliding windows from the loaded records."""
        for record_name in self.records:
            try:
                # Read signals and header files using WFDB
                signals, fields = wfdb.rdsamp(
                    os.path.join(self.data_dir, record_name)
                )

                # Signal preprocessing
                n_samples = signals.shape[0]
                if n_samples < self.window_size:
                    print(f"Skipping {record_name}: too short ({n_samples} samples)")
                    continue

                # Select the first 4 channels and normalize
                data = signals[:, :4].astype(np.float32)
                # Normalize each channel
                data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

                # Determine label based on filename
                # ARR prefix indicates arrhythmia (abnormal)
                label = 1 if record_name.startswith("ARR") else 0

                # Segment using sliding window
                for start in range(0, n_samples - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    window = data[start:end, :]

                    # Transpose to (channels, window_size) format
                    window = window.T  # Transpose from (L, C) to (C, L)

                    self.segments.append((window, label))

            except Exception as e:
                print(f"Error processing {record_name}: {str(e)}")
                continue

        print(f"Created {len(self.segments)} windows from all records")

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

    def get_channel_info(self) -> dict:
        """
        Get information about the channels in the dataset.

        Returns:
            Dictionary with channel information
        """
        if not self.segments:
            return {}

        first_segment = self.segments[0][0]
        return {
            'num_channels': first_segment.shape[0],
            'window_size': first_segment.shape[1],
            'channel_shape': first_segment.shape
        }


def create_data_loaders(dataset: NIFEADataset,
                       batch_size: int = 64,
                       train_split: float = 0.8,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders.

    Args:
        dataset: NIFEADataset instance
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


def inspect_dataset(data_dir: str, window_size: int = 100) -> None:
    """
    Inspect a dataset directory to understand its structure.

    Args:
        data_dir: Directory containing WFDB files
        window_size: Window size for segmentation
    """
    print("\n" + "="*60)
    print("DATASET INSPECTION")
    print("="*60)

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")
        return

    # List files in directory
    files = os.listdir(data_dir)
    dat_files = [f for f in files if f.endswith('.dat')]
    hea_files = [f for f in files if f.endswith('.hea')]

    print(f"Directory: {data_dir}")
    print(f"Found {len(dat_files)} .dat files")
    print(f"Found {len(hea_files)} .hea files")

    if dat_files:
        print(f"Sample files: {dat_files[:5]}")

    # Create a temporary dataset
    try:
        temp_dataset = NIFEADataset(data_dir, window_size=window_size)

        # Print information
        print(f"\nDataset Statistics:")
        print(f"  Total records: {len(temp_dataset.records)}")
        print(f"  Total windows: {len(temp_dataset)}")
        print(f"  Window size: {window_size}")
        print(f"  Class distribution: {temp_dataset.get_class_distribution()}")

        channel_info = temp_dataset.get_channel_info()
        print(f"  Number of channels: {channel_info.get('num_channels', 'N/A')}")

        # Show record types
        arr_records = [r for r in temp_dataset.records if r.startswith("ARR")]
        normal_records = [r for r in temp_dataset.records if not r.startswith("ARR")]
        print(f"  Arrhythmia records: {len(arr_records)}")
        print(f"  Normal records: {len(normal_records)}")

    except Exception as e:
        print(f"\nERROR creating dataset: {str(e)}")
        print("Please ensure:")
        print("1. Both .dat and .hea files are present")
        print("2. Files are in the correct WFDB format")
        print("3. You have the required packages: pip install wfdb")

    print("\n" + "="*60)