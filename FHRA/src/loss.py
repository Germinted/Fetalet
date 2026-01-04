"""Loss functions for Fetal Heart Rate Analysis"""

import torch
import torch.nn as nn
from typing import Tuple


class HybridLoss(nn.Module):
    """
    Hybrid loss combining cross-entropy and prototype-based losses.

    This loss function combines:
    1. Cross-entropy loss for classification
    2. Prototype loss for encouraging proper shapelet learning
    """

    def __init__(self,
                 alpha: float = 0.5,
                 margin: float = 1.0,
                 eps: float = 1e-8):
        """
        Initialize the hybrid loss.

        Args:
            alpha: Weight for prototype loss (0 <= alpha <= 1)
            margin: Margin for negative samples in prototype loss
            eps: Small value for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.eps = eps
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                logits: torch.Tensor,
                distances: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the hybrid loss.

        Args:
            logits: Classification logits (batch_size, num_classes)
            distances: Prototype distances (batch_size, num_prototypes)
            labels: Ground truth labels (batch_size,)

        Returns:
            Tuple of (total_loss, ce_loss, prototype_loss)
        """
        # Cross-entropy loss
        ce = self.ce_loss(logits, labels)

        # Convert labels to float for mask computation
        labels_float = labels.float()

        # Positive sample handling (label == 1)
        pos_mask = labels_float == 1
        if pos_mask.sum() > 0:
            # For positive samples, minimize distance to nearest prototype
            pos_dist = torch.min(distances[pos_mask], dim=1)[0]
            pos_loss = torch.mean(pos_dist)
        else:
            pos_loss = torch.tensor(0.0, device=logits.device)

        # Negative sample handling (label == 0)
        neg_mask = labels_float == 0
        if neg_mask.sum() > 0:
            # For negative samples, enforce margin
            neg_dist = torch.relu(self.margin - torch.min(distances[neg_mask], dim=1)[0])
            neg_loss = torch.mean(neg_dist)
        else:
            neg_loss = torch.tensor(0.0, device=logits.device)

        # Weighted prototype loss
        valid_samples = pos_mask.sum() + neg_mask.sum() + self.eps
        proto_loss = (pos_loss * pos_mask.sum() + neg_loss * neg_mask.sum()) / valid_samples

        # Combine losses
        total_loss = (1 - self.alpha) * ce + self.alpha * proto_loss

        return total_loss, ce, proto_loss


class ContrastivePrototypeLoss(nn.Module):
    """
    Contrastive loss for prototype learning.

    This loss encourages:
    1. Positive samples to be close to their assigned prototypes
    2. Negative samples to be far from all prototypes
    3. Prototypes to be well-separated from each other
    """

    def __init__(self,
                 temperature: float = 0.1,
                 margin: float = 1.0,
                 prototype_regularization: float = 0.01):
        """
        Initialize the contrastive prototype loss.

        Args:
            temperature: Temperature parameter for softmax
            margin: Margin for negative samples
            prototype_regularization: Weight for prototype separation term
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.prototype_regularization = prototype_regularization

    def forward(self,
                distances: torch.Tensor,
                labels: torch.Tensor,
                prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive prototype loss.

        Args:
            distances: Prototype distances (batch_size, num_prototypes)
            labels: Ground truth labels (batch_size,)
            prototypes: Prototype parameters (num_prototypes, shapelet_length, in_channels)

        Returns:
            Total loss value
        """
        batch_size, num_prototypes = distances.shape

        # Convert distances to similarities (lower distance = higher similarity)
        similarities = -distances / self.temperature

        # Sample-specific loss
        positive_mask = labels == 1
        negative_mask = labels == 0

        sample_loss = 0.0

        # Positive samples: minimize distance to nearest prototype
        if positive_mask.sum() > 0:
            pos_similarities = similarities[positive_mask]
            # Maximize similarity to best prototype
            pos_loss = -torch.mean(torch.max(pos_similarities, dim=1)[0])
            sample_loss += pos_loss

        # Negative samples: enforce margin
        if negative_mask.sum() > 0:
            neg_distances = distances[negative_mask]
            # All distances should be above margin
            neg_loss = torch.mean(torch.relu(self.margin - torch.min(neg_distances, dim=1)[0]))
            sample_loss += neg_loss

        # Prototype regularization: encourage prototype separation
        if num_prototypes > 1:
            # Reshape prototypes for pairwise distance computation
            p_flat = prototypes.view(num_prototypes, -1)
            # Compute pairwise distances
            proto_diff = p_flat.unsqueeze(1) - p_flat.unsqueeze(0)
            proto_distances = torch.norm(proto_diff, dim=2)

            # Remove diagonal (distance to self)
            mask = torch.eye(num_prototypes, device=proto_distances.device, dtype=torch.bool)
            proto_distances = proto_distances[~mask]

            # Encourage minimum separation between prototypes
            min_proto_dist = torch.min(proto_distances)
            proto_reg_loss = -min_proto_dist  # Maximize minimum distance

            total_loss = sample_loss + self.prototype_regularization * proto_reg_loss
        else:
            total_loss = sample_loss

        return total_loss