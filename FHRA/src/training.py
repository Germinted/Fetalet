"""Training and evaluation functions for Fetal Heart Rate Analysis"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           classification_report, roc_auc_score,
                           roc_curve, f1_score, precision_recall_curve)
from sklearn.cluster import KMeans
from typing import Tuple, Dict, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for the dual-branch FHR model.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-4):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device for computation
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model.to(device)

    def train_epoch(self, criterion: nn.Module) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            criterion: Loss function

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        ce_loss = 0.0
        proto_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits, distances = self.model(inputs)
            loss, ce, proto = criterion(logits, distances, labels)

            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            ce_loss += ce.item() * batch_size
            proto_loss += proto.item() * batch_size
            correct += (logits.argmax(1) == labels).sum().item()
            total += batch_size

            # Log progress
            if batch_idx % 50 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.4f}')

        # Calculate average metrics
        metrics = {
            'loss': total_loss / total,
            'ce_loss': ce_loss / total,
            'proto_loss': proto_loss / total,
            'acc': correct / total
        }

        return metrics

    def validate(self, criterion: nn.Module) -> Dict[str, float]:
        """
        Validate the model on the test set.

        Args:
            criterion: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        ce_loss = 0.0
        proto_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logits, distances = self.model(inputs)
                loss, ce, proto = criterion(logits, distances, labels)

                # Accumulate metrics
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                ce_loss += ce.item() * batch_size
                proto_loss += proto.item() * batch_size
                correct += (logits.argmax(1) == labels).sum().item()
                total += batch_size

        metrics = {
            'loss': total_loss / total,
            'ce_loss': ce_loss / total,
            'proto_loss': proto_loss / total,
            'acc': correct / total
        }

        return metrics

    def train(self,
              criterion: nn.Module,
              epochs: int,
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            criterion: Loss function
            epochs: Number of training epochs
            save_path: Path to save the best model

        Returns:
            Training history dictionary
        """
        history = {
            'train_loss': [],
            'train_ce_loss': [],
            'train_proto_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_ce_loss': [],
            'val_proto_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(criterion)

            # Validation
            val_metrics = self.validate(criterion)

            # Update history
            for key in train_metrics:
                history[f'train_{key}'].append(train_metrics[key])
                history[f'val_{key}'].append(val_metrics[key])

            epoch_time = time.time() - start_time

            # Log epoch results
            logger.info(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.2f}s)")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"CE: {train_metrics['ce_loss']:.4f}, "
                       f"Proto: {train_metrics['proto_loss']:.4f}, "
                       f"Acc: {train_metrics['acc']:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"CE: {val_metrics['ce_loss']:.4f}, "
                       f"Proto: {val_metrics['proto_loss']:.4f}, "
                       f"Acc: {val_metrics['acc']:.4f}")

            # Save best model
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': best_val_acc,
                    }, save_path)
                    logger.info(f"New best model saved with accuracy: {best_val_acc:.4f}")

        return history


class Evaluator:
    """
    Evaluator class for detailed model evaluation.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the evaluator.

        Args:
            model: Trained PyTorch model
            device: Device for computation
        """
        self.model = model
        self.device = device

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive evaluation of the model.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits, _ = self.model(inputs)
                probs = torch.softmax(logits, dim=1)

                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds),
            'roc_auc': roc_auc_score(all_labels, all_probs),
            'precision': None,  # Will be calculated below
            'recall': None,     # Will be calculated below
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }

        # Calculate precision and recall for each class
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        metrics['precision'] = precision
        metrics['recall'] = recall

        # Print detailed report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(all_labels, all_preds, zero_division=0))
        print("\nMETRICS SUMMARY")
        print("="*50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        return metrics

    def evaluate_with_timing(self, test_loader: DataLoader) -> Tuple[Dict[str, float], float]:
        """
        Evaluate model and measure inference time.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (metrics, average_inference_time)
        """
        self.model.eval()
        times = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                start_time = time.time()
                _ = self.model(inputs)
                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = np.mean(times) / inputs.size(0)  # Per sample

        metrics = self.evaluate(test_loader)

        print(f"\nAverage inference time per sample: {avg_time*1000:.2f} ms")

        return metrics, avg_time


def initialize_prototypes(model: nn.Module,
                         train_subset: torch.utils.data.Subset,
                         device: torch.device,
                         n_prototypes: int = 5) -> None:
    """
    Initialize model prototypes using K-means clustering on abnormal samples.

    Args:
        model: Dual-branch model with prototype layer
        train_subset: Training dataset subset
        device: Device for computation
        n_prototypes: Number of prototypes to initialize
    """
    logger.info("Initializing prototypes using K-means clustering...")

    # Collect abnormal samples
    abnormal_samples = []
    all_samples = []

    for idx in train_subset.indices:
        seg_np, label = train_subset.dataset.segments[idx]
        seg_tensor = torch.from_numpy(seg_np).float().to(device)

        all_samples.append(seg_tensor.unsqueeze(0))
        if label == 1:
            abnormal_samples.append(seg_tensor.unsqueeze(0))

    if not abnormal_samples:
        logger.warning("No abnormal samples found for prototype initialization")
        return

    # Concatenate samples
    abnormal_tensor = torch.cat(abnormal_samples, dim=0)
    all_tensor = torch.cat(all_samples, dim=0)

    # Generate candidate shapelets
    model.eval()
    with torch.no_grad():
        candidates = model.shapelet_generator(all_tensor)  # (N, K, L, C)

    # Reshape for clustering
    shapelet_length = model.shapelet_generator.shapelet_length
    in_channels = model.shapelet_generator.in_channels
    candidates_flat = candidates.cpu().numpy().reshape(
        -1, shapelet_length * in_channels
    )

    # K-means clustering
    kmeans = KMeans(n_clusters=n_prototypes, random_state=42)
    kmeans.fit(candidates_flat)

    # Set prototypes
    with torch.no_grad():
        prototype_centers = kmeans.cluster_centers_.reshape(
            n_prototypes, shapelet_length, in_channels
        )
        model.prototype_layer.copy_(
            torch.tensor(prototype_centers, dtype=torch.float32, device=device)
        )

    logger.info("Prototype initialization complete")