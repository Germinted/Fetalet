"""Neural network models for Fetal Heart Rate Analysis

Improved architecture with:
- BatchNorm/LayerNorm for stable training
- Deeper CNN with residual connections (4 layers)
- Proper sliding window shapelet extraction
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with BatchNorm and activation.

    This block provides:
    - Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm
    - Residual connection for better gradient flow
    - Optional dropout for regularization
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Dropout for regularization
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        # Residual projection if channel dimensions don't match
        self.residual = None
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply residual connection
        if self.residual is not None:
            identity = self.residual(identity)

        out += identity
        out = self.relu(out)

        return out


class SlidingWindowExtractor(nn.Module):
    """
    Extracts all possible sliding window shapelets from input time series.

    This implements proper sliding window extraction, allowing shapelet matching
    at any position within the input window.
    """

    def __init__(self, shapelet_length: int):
        """
        Initialize the sliding window extractor.

        Args:
            shapelet_length: Length of each shapelet window
        """
        super().__init__()
        self.shapelet_length = shapelet_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract sliding windows from input.

        Args:
            x: Input tensor of shape (batch_size, in_channels, window_size)

        Returns:
            Windows of shape (batch_size, in_channels, num_windows, shapelet_length)
            where num_windows = window_size - shapelet_length + 1
        """
        batch_size, in_channels, window_size = x.shape

        # Use unfold to extract all sliding windows
        # unfold(dimension, size, step) extracts sliding windows
        windows = x.unfold(2, self.shapelet_length, 1)

        # Shape: (batch_size, in_channels, num_windows, shapelet_length)
        return windows


class CandidateGenerator(nn.Module):
    """
    Enhanced candidate shapelet generator with proper architecture improvements.

    Improvements:
    - BatchNorm after each convolution
    - Deeper network (4 conv layers)
    - Residual connections
    - Proper sliding window extraction
    """

    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 num_candidates: int = 5,
                 shapelet_length: int = 30,
                 use_dropout: bool = True,
                 dropout_rate: float = 0.2):
        """
        Initialize the candidate generator.

        Args:
            in_channels: Number of input channels (e.g., FHR, TOCO)
            window_size: Length of input window
            num_candidates: Number of candidate shapelets to generate
            shapelet_length: Length of each candidate shapelet
            use_dropout: Whether to use dropout
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.shapelet_length = shapelet_length
        self.num_candidates = num_candidates
        self.in_channels = in_channels

        # Sliding window extractor for proper position-aware shapelet extraction
        self.sliding_window = SlidingWindowExtractor(shapelet_length)

        # Enhanced CNN with 4 layers, BatchNorm, and residual connections
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Layer 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Layer 3
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Layer 4
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Output projection to num_candidates * in_channels
            nn.Conv1d(64, num_candidates * in_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(shapelet_length)
        )

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

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

        if self.use_dropout and self.training:
            out = self.dropout(out)

        # Reshape to: (batch_size, num_candidates, in_channels, shapelet_length)
        out = out.view(batch_size, self.num_candidates, self.in_channels, self.shapelet_length)

        # Permute to: (batch_size, num_candidates, shapelet_length, in_channels)
        return out.permute(0, 1, 3, 2)


class PositionAwarePrototypeMatcher(nn.Module):
    """
    Position-aware prototype matching with proper sliding window.

    This module:
    1. Extracts all sliding window positions from the input
    2. Matches each position against learned prototypes
    3. Returns the minimum distance across all positions
    """

    def __init__(self,
                 num_prototypes: int,
                 shapelet_length: int,
                 in_channels: int):
        """
        Initialize the position-aware prototype matcher.

        Args:
            num_prototypes: Number of prototype shapelets
            shapelet_length: Length of each prototype
            in_channels: Number of input channels
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.shapelet_length = shapelet_length
        self.in_channels = in_channels

        # Sliding window extractor
        self.sliding_window = SlidingWindowExtractor(shapelet_length)

        # Learnable prototype layer
        self.prototype_layer = nn.Parameter(
            torch.randn(num_prototypes, shapelet_length, in_channels),
            requires_grad=True
        )

        # Layer normalization for stable training
        self.prototype_norm = nn.LayerNorm((shapelet_length, in_channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with position-aware matching.

        Args:
            x: Input tensor of shape (batch_size, in_channels, window_size)

        Returns:
            Tuple of (min_distances, all_distances)
            - min_distances: Minimum distance across positions (batch_size, num_prototypes)
            - all_distances: All distances for interpretability (batch_size, num_prototypes, num_positions)
        """
        batch_size, in_channels, window_size = x.shape

        # Extract sliding windows: (batch_size, in_channels, num_windows, shapelet_length)
        windows = self.sliding_window(x)
        num_windows = windows.size(2)

        # Permute to: (batch_size, num_windows, shapelet_length, in_channels)
        windows = windows.permute(0, 2, 3, 1)

        # Get normalized prototypes
        prototypes = self.prototype_norm(self.prototype_layer)

        # Expand prototypes: (1, 1, num_prototypes, shapelet_length, in_channels)
        prototypes_expanded = prototypes.unsqueeze(0).unsqueeze(0)

        # Expand windows: (batch_size, num_windows, 1, shapelet_length, in_channels)
        windows_expanded = windows.unsqueeze(2)

        # Compute pairwise distances
        diff = prototypes_expanded - windows_expanded
        distances = torch.norm(diff, dim=(3, 4))  # (batch_size, num_windows, num_prototypes)

        # Permute to: (batch_size, num_prototypes, num_windows)
        distances = distances.permute(0, 2, 1)

        # Find minimum distance across all window positions
        min_distances, min_positions = torch.min(distances, dim=2)

        return min_distances, distances

    def get_prototypes(self) -> torch.Tensor:
        """Get the learned prototype shapelets."""
        return self.prototype_layer.data.clone()

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the prototype shapelets."""
        with torch.no_grad():
            self.prototype_layer.copy_(prototypes)


class DualBranchModel(nn.Module):
    """
    Enhanced dual-branch model with Priority 1 improvements:

    1. BatchNorm/LayerNorm on all CNN layers
    2. Proper sliding window shapelet extraction
    3. Deeper CNN with 4 layers and residual connections
    4. Dropout for regularization (0.2-0.3)
    """

    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 num_classes: int = 2,
                 num_prototypes: int = 5,
                 shapelet_length: int = 30,
                 num_candidates: int = 5,
                 dropout_rate: float = 0.25,
                 use_residual: bool = True):
        """
        Initialize the enhanced dual-branch model.

        Args:
            in_channels: Number of input channels
            window_size: Length of input window
            num_classes: Number of output classes
            num_prototypes: Number of prototype shapelets
            shapelet_length: Length of each prototype
            num_candidates: Number of candidate shapelets to generate
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
        """
        super().__init__()

        # Store parameters
        self.in_channels = in_channels
        self.num_prototypes = num_prototypes
        self.shapelet_length = shapelet_length
        self.num_candidates = num_candidates
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

        # Position-aware prototype matcher with sliding window
        self.prototype_matcher = PositionAwarePrototypeMatcher(
            num_prototypes, shapelet_length, in_channels
        )

        # Also keep the candidate generator for backward compatibility
        self.shapelet_generator = CandidateGenerator(
            in_channels, window_size, num_candidates, shapelet_length,
            use_dropout=True, dropout_rate=dropout_rate
        )

        # Enhanced CNN branch with 4 layers, BatchNorm, and optional residual connections
        self.cnn_branch = self._build_cnn_branch(in_channels, window_size, use_residual)

        # Combined classification layer with dropout
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64 + num_prototypes, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

    def _build_cnn_branch(self, in_channels: int, window_size: int,
                          use_residual: bool) -> nn.Sequential:
        """
        Build the enhanced CNN branch with 4 layers.

        Args:
            in_channels: Number of input channels
            window_size: Length of input window
            use_residual: Whether to use residual connections

        Returns:
            Sequential CNN model with BatchNorm and optional residual blocks
        """
        # Calculate output size after convolutions
        # Input: (batch_size, in_channels, window_size)
        # After Conv+Pool: window_size -> window_size//2 -> window_size//4
        conv_output_size = window_size // 4

        if use_residual:
            # Use residual blocks for better gradient flow
            return nn.Sequential(
                # Initial conv
                nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Residual block 1
                ResidualConvBlock(32, 64, kernel_size=3, use_dropout=True, dropout_rate=self.dropout_rate),
                nn.MaxPool1d(2),

                # Residual block 2
                ResidualConvBlock(64, 64, kernel_size=3, use_dropout=True, dropout_rate=self.dropout_rate),

                # Flatten and FC
                nn.Flatten(),
                nn.Linear(64 * conv_output_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )
        else:
            # Standard deep CNN with BatchNorm
            return nn.Sequential(
                # Layer 1
                nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Layer 2
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Layer 3
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),

                # Layer 4
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),

                # Flatten and FC
                nn.Flatten(),
                nn.Linear(64 * conv_output_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )

    def forward(self, x: torch.Tensor,
                return_all_distances: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the enhanced dual-branch model.

        Args:
            x: Input tensor of shape (batch_size, in_channels, window_size)
            return_all_distances: Whether to return all position-wise distances

        Returns:
            Tuple of (logits, min_distances) or (logits, min_distances, all_distances)
        """
        batch_size = x.size(0)

        # Position-aware prototype matching with sliding window
        min_distances, all_distances = self.prototype_matcher(x)

        # CNN branch features
        cnn_features = self.cnn_branch(x)  # (batch_size, 64)

        # Combine features
        combined_features = torch.cat([cnn_features, min_distances], dim=1)

        # Classification
        logits = self.fc(combined_features)

        if return_all_distances:
            return logits, min_distances, all_distances
        return logits, min_distances

    def get_prototypes(self) -> torch.Tensor:
        """Get the learned prototype shapelets."""
        return self.prototype_matcher.get_prototypes()

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the prototype shapelets."""
        self.prototype_matcher.set_prototypes(prototypes)

    def get_prototype_match_positions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the positions where prototypes best match the input.

        Useful for visualization and interpretability.

        Args:
            x: Input tensor of shape (batch_size, in_channels, window_size)

        Returns:
            Position indices of shape (batch_size, num_prototypes)
        """
        with torch.no_grad():
            _, all_distances = self.prototype_matcher(x)
            # Position with minimum distance for each prototype
            min_positions = torch.argmin(all_distances, dim=2)
            return min_positions


class LightweightDualBranchModel(nn.Module):
    """
    A lightweight version of the dual-branch model for faster training/inference.

    Uses fewer channels but maintains all architectural improvements.
    """

    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 num_classes: int = 2,
                 num_prototypes: int = 5,
                 shapelet_length: int = 30,
                 dropout_rate: float = 0.2):
        """
        Initialize the lightweight dual-branch model.

        Args:
            in_channels: Number of input channels
            window_size: Length of input window
            num_classes: Number of output classes
            num_prototypes: Number of prototype shapelets
            shapelet_length: Length of each prototype
            dropout_rate: Dropout rate
        """
        super().__init__()

        # Store parameters
        self.in_channels = in_channels
        self.num_prototypes = num_prototypes
        self.shapelet_length = shapelet_length

        # Simplified position-aware matcher
        self.prototype_matcher = PositionAwarePrototypeMatcher(
            num_prototypes, shapelet_length, in_channels
        )

        # Lightweight CNN branch (fewer channels)
        conv_output_size = window_size // 4

        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(32 * conv_output_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classification layer
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32 + num_prototypes, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Prototype matching
        min_distances, _ = self.prototype_matcher(x)

        # CNN features
        cnn_features = self.cnn_branch(x)

        # Combine and classify
        combined = torch.cat([cnn_features, min_distances], dim=1)
        logits = self.fc(combined)

        return logits, min_distances

    def get_prototypes(self) -> torch.Tensor:
        """Get the learned prototype shapelets."""
        return self.prototype_matcher.get_prototypes()

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the prototype shapelets."""
        self.prototype_matcher.set_prototypes(prototypes)
