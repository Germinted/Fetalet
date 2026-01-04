"""
Main script for NIFEA Fetal Heart Rate Anomaly Detection.
This script demonstrates the use of the refactored modular codebase for NIFEA dataset.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import from modular components
from src.config import Config
from src.dataset import NIFEADataset, create_data_loaders, inspect_dataset
from src.models import DualBranchModel
from src.loss import HybridLoss
from src.training import Trainer, Evaluator, initialize_prototypes
from src.visualization import NIFEAVisualizer
from src.utils import setup_seed, ensure_dir, print_model_info


def main():
    """Main function to run the NIFEA FHR anomaly detection."""
    # Setup
    print("="*60)
    print("NIFEA Fetal Heart Rate Anomaly Detection")
    print("="*60)

    # Set random seed for reproducibility
    setup_seed(Config.TRAINING.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Inspect dataset first
    print("\n1. Inspecting dataset...")
    inspect_dataset(
        Config.DATA.DATA_DIR,
        window_size=Config.DATA.WINDOW_SIZE
    )

    # Load dataset
    print("\n2. Loading dataset...")
    dataset = NIFEADataset(
        data_dir=Config.DATA.DATA_DIR,
        window_size=Config.DATA.WINDOW_SIZE,
        stride=Config.DATA.STRIDE
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    print(f"Channel info: {dataset.get_channel_info()}")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=Config.TRAINING.BATCH_SIZE,
        train_split=Config.DATA.TRAIN_RATIO,
        seed=Config.TRAINING.SEED
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    print("\n3. Initializing model...")
    model = DualBranchModel(
        in_channels=Config.DATA.IN_CHANNELS,
        window_size=Config.DATA.WINDOW_SIZE,
        num_classes=Config.MODEL.NUM_CLASSES,
        num_prototypes=Config.MODEL.NUM_PROTOTYPES,
        shapelet_length=Config.MODEL.SHAPELET_LENGTH,
        num_candidates=Config.MODEL.NUM_CANDIDATES
    ).to(device)

    # Print model information
    print_model_info(model, (1, Config.DATA.IN_CHANNELS, Config.DATA.WINDOW_SIZE))

    # Initialize prototypes
    if Config.TRAINING.INIT_PROTOTYPES:
        print("\n4. Initializing prototypes...")
        # Get the training subset from the data loader
        train_dataset = train_loader.dataset.dataset
        train_indices = train_loader.dataset.indices

        from torch.utils.data import Subset
        train_subset = Subset(train_dataset, train_indices)

        initialize_prototypes(
            model,
            train_subset,
            device,
            Config.TRAINING.N_INIT_PROTOTYPES
        )

    # Training setup
    print("\n5. Setting up training...")
    criterion = HybridLoss(
        alpha=Config.MODEL.ALPHA,
        margin=Config.MODEL.MARGIN,
        eps=Config.MODEL.EPS
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=Config.TRAINING.LEARNING_RATE
    )

    # Training
    print("\n6. Training model...")
    print(f"Training for {Config.TRAINING.EPOCHS} epochs...")
    start_time = time.time()

    # Create save directory for model
    ensure_dir("models")
    save_path = os.path.join("models", "nifea_best_model.pth")

    history = trainer.train(
        criterion=criterion,
        epochs=Config.TRAINING.EPOCHS,
        save_path=save_path
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Evaluation
    print("\n7. Evaluating model...")
    evaluator = Evaluator(model, device)
    metrics, inference_time = evaluator.evaluate_with_timing(test_loader)

    # Visualizations
    print("\n8. Generating visualizations...")
    visualizer = NIFEAVisualizer(
        figsize=Config.VIZ.FIGURE_SIZE,
        dpi=Config.VIZ.FIGURE_DPI
    )

    # Create output directory
    ensure_dir("outputs")

    # Plot shapelet explanations for different channels
    print("Generating shapelet explanations...")
    for channel in range(min(4, Config.DATA.IN_CHANNELS)):  # Visualize up to 4 channels
        visualizer.plot_shapelet_explanation(
            model=model,
            dataset=test_loader.dataset.dataset,
            device=device,
            channel=channel,
            n_samples=2,
            save_dir="outputs",
            show=False  # Don't display, just save
        )

    # Plot training curves
    if history:
        visualizer.plot_training_curves(
            history,
            save_path=os.path.join("outputs", "training_curves.png"),
            show=False
        )

    print(f"\nVisualizations saved to: outputs/")

    # Summary
    print("\n9. Summary of Results")
    print("="*60)
    print(f"Final Test Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Inference Time: {inference_time*1000:.2f} ms/sample")
    print(f"Training Time: {training_time:.2f} seconds")
    print("\nModel saved to: models/nifea_best_model.pth")
    print("Analysis complete!")


if __name__ == "__main__":
    main()