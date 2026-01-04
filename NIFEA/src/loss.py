"""Loss functions for NIFEA Fetal Heart Rate Analysis"""

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
                 alpha: float = 0.3,
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

        # Positive sample handling (label == 1) - arrhythmia
        pos_mask = labels_float == 1
        if pos_mask.sum() > 0:
            # For positive samples, minimize distance to nearest prototype
            pos_dist = torch.min(distances[pos_mask], dim=1)[0]
            pos_loss = torch.mean(pos_dist)
        else:
            pos_loss = torch.tensor(0.0, device=logits.device)

        # Negative sample handling (label == 0) - normal
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