"""Neural network models for Fetal Heart Rate Analysis"""

import torch
import torch.nn as nn
from typing import Tuple


class CandidateGenerator(nn.Module):
    """
    Generates candidate shapelets from input time series using CNN.

    This module uses a series of 1D convolutions to extract candidate
    shapelets from multi-channel time series data.
    """

    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 num_candidates: int = 5,
                 shapelet_length: int = 30):
        """
        Initialize the candidate generator.

        Args:
            in_channels: Number of input channels (e.g., FHR, TOCO)
            window_size: Length of input window
            num_candidates: Number of candidate shapelets to generate
            shapelet_length: Length of each candidate shapelet
        """
        super().__init__()
        self.shapelet_length = shapelet_length
        self.num_candidates = num_candidates
        self.in_channels = in_channels

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Output channels: num_candidates * in_channels
            nn.Conv1d(32, num_candidates * in_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(shapelet_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the candidate generator.

        Args:
            x: Input tensor of shape (batch_size, in_channels, window_size)

        Returns:
            Candidate shapelets of shape (batch_size, num_candidates, shapelet_length, in_channels)
        """
        batch_size = x.size(0)

        # Output shape: (batch_size, num_candidates * in_channels, shapelet_length)
        out = self.cnn(x)

        # Reshape to: (batch_size, num_candidates, in_channels, shapelet_length)
        out = out.view(batch_size, self.num_candidates, self.in_channels, self.shapelet_length)

        # Permute to: (batch_size, num_candidates, shapelet_length, in_channels)
        return out.permute(0, 1, 3, 2)


class DualBranchModel(nn.Module):
    """
    Dual-branch model combining CNN features with prototype-based shapelet matching.

    This model uses two branches:
    1. A CNN branch for learning general features
    2. A prototype-based branch for matching learned shapelets
    """

    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 num_classes: int = 2,
                 num_prototypes: int = 5,
                 shapelet_length: int = 30,
                 num_candidates: int = 5):
        """
        Initialize the dual-branch model.

        Args:
            in_channels: Number of input channels
            window_size: Length of input window
            num_classes: Number of output classes
            num_prototypes: Number of prototype shapelets
            shapelet_length: Length of each prototype
            num_candidates: Number of candidate shapelets to generate
        """
        super().__init__()

        # Store parameters
        self.in_channels = in_channels
        self.num_prototypes = num_prototypes
        self.shapelet_length = shapelet_length
        self.num_candidates = num_candidates

        # Shapelet generator branch
        self.shapelet_generator = CandidateGenerator(
            in_channels, window_size, num_candidates, shapelet_length
        )

        # Prototype layer - learnable shapelet prototypes
        # Shape: (num_prototypes, shapelet_length, in_channels)
        self.prototype_layer = nn.Parameter(
            torch.randn(num_prototypes, shapelet_length, in_channels),
            requires_grad=True
        )

        
        # CNN branch for feature extraction
        self.cnn_branch = self._build_cnn_branch(in_channels, window_size)

        # Combined classification layer
        self.fc = nn.Sequential(
            nn.Linear(64 + num_prototypes, 64),  # 64 from CNN + num_prototypes from shapelets
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _build_cnn_branch(self, in_channels: int, window_size: int) -> nn.Sequential:
        """
        Build the CNN branch for feature extraction.

        Args:
            in_channels: Number of input channels
            window_size: Length of input window

        Returns:
            Sequential CNN model
        """
        # Calculate the output size after convolutions
        # Input: (batch_size, in_channels, window_size)
        # After Conv1d + MaxPool1d: window_size // 2
        # After Conv1d + MaxPool1d: (window_size // 2) // 2 = window_size // 4
        conv_output_size = window_size // 4

        return nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * conv_output_size, 64)  # Changed to 64 to match actual output
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dual-branch model.

        Args:
            x: Input tensor of shape (batch_size, in_channels, window_size)

        Returns:
            Tuple of (logits, min_distances)
            - logits: Classification logits of shape (batch_size, num_classes)
            - min_distances: Minimum prototype distances of shape (batch_size, num_prototypes)
        """
        batch_size = x.size(0)

        # Generate candidate shapelets: (batch_size, num_candidates, shapelet_length, in_channels)
        candidates = self.shapelet_generator(x)

        # Expand prototypes for broadcasting: (1, num_prototypes, 1, shapelet_length, in_channels)
        prototypes = self.prototype_layer.unsqueeze(0).unsqueeze(2)

        # Expand candidates for broadcasting: (batch_size, 1, num_candidates, shapelet_length, in_channels)
        candidates = candidates.unsqueeze(1)

        # Compute pairwise distances
        # Shape after subtraction: (batch_size, num_prototypes, num_candidates, shapelet_length, in_channels)
        diff = prototypes - candidates

        # Compute L2 norm along shapelet_length and in_channels dimensions
        # Shape: (batch_size, num_prototypes, num_candidates)
        distances = torch.norm(diff.view(diff.size(0), diff.size(1), diff.size(2), -1), dim=3)

        # Double minimization: first over shapelet_length positions, then over candidates
        # Actually, we need to minimize over the sliding window positions
        # For now, we'll minimize over candidates (assuming positions are already handled)
        min_distances, _ = torch.min(distances, dim=2)  # (batch_size, num_prototypes)

        # CNN branch features
        cnn_features = self.cnn_branch(x)  # (batch_size, 64)

        # Combine features
        combined_features = torch.cat([cnn_features, min_distances], dim=1)

        
        # Classification
        logits = self.fc(combined_features)

        return logits, min_distances

    def get_prototypes(self) -> torch.Tensor:
        """
        Get the learned prototype shapelets.

        Returns:
            Prototype tensor of shape (num_prototypes, shapelet_length, in_channels)
        """
        return self.prototype_layer.data.clone()

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """
        Set the prototype shapelets.

        Args:
            prototypes: New prototype values of shape (num_prototypes, shapelet_length, in_channels)
        """
        with torch.no_grad():
            self.prototype_layer.copy_(prototypes)