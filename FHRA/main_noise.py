"""
Main script for Fetal Heart Rate Anomaly Detection with Noise.
This script demonstrates the use of the refactored modular codebase.
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
from src.config_noise import Config
from src.dataset import FHRDataset, create_data_loaders
from src.models import DualBranchModel
from src.loss import HybridLoss
from src.training import Trainer, Evaluator, initialize_prototypes
from src.visualization import FHRVisualizer
from src.utils import setup_seed, ensure_dir


def main():
    """Main function to run the FHR anomaly detection with noise."""
    # Setup
    print("="*60)
    print("Fetal Heart Rate Anomaly Detection with Noise")
    print("="*60)

    # Set random seed for reproducibility
    setup_seed(Config.TRAINING.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create necessary directories
    ensure_dir(Config.DATA.PLOT_DIR)

    # Load clean dataset
    print("\n1. Loading dataset...")
    dataset = FHRDataset(
        data_dir=Config.DATA.DATA_DIR,
        window_size=Config.DATA.WINDOW_SIZE,
        stride=Config.DATA.STRIDE
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=Config.TRAINING.BATCH_SIZE,
        train_split=Config.DATA.TRAIN_RATIO,
        seed=Config.TRAINING.SEED
    )

    # Initialize model
    print("\n2. Initializing model...")
    model = DualBranchModel(
        in_channels=Config.DATA.IN_CHANNELS,
        window_size=Config.DATA.WINDOW_SIZE,
        num_classes=Config.MODEL.NUM_CLASSES,
        num_prototypes=Config.MODEL.NUM_PROTOTYPES,
        shapelet_length=Config.MODEL.SHAPELET_LENGTH,
        num_candidates=Config.MODEL.NUM_CANDIDATES
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize prototypes
    if Config.TRAINING.INIT_PROTOTYPES:
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
    print("\n3. Setting up training...")
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
    print("\n4. Training model...")
    print(f"Training for {Config.TRAINING.EPOCHS} epochs...")
    start_time = time.time()

    history = trainer.train(
        criterion=criterion,
        epochs=Config.TRAINING.EPOCHS
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Evaluation on clean test set
    print("\n5. Evaluating on clean test set...")
    evaluator = Evaluator(model, device)
    clean_metrics, inference_time = evaluator.evaluate_with_timing(test_loader)

    # Test with different noise scenarios
    print("\n6. Testing with noise scenarios...")
    noise_results = {}

    for scenario in Config.DATA.NOISE_SCENARIOS:
        print(f"\nTesting scenario: {scenario['name']}")

        # Create noisy dataset
        noisy_dataset = FHRDataset(
            data_dir=Config.DATA.DATA_DIR,
            window_size=Config.DATA.WINDOW_SIZE,
            stride=Config.DATA.STRIDE,
            add_noise=True,
            noise_types=scenario['noise_types'],
            noise_params=scenario['params']
        )

        # Create test loader for noisy data (using same split as clean)
        _, noisy_test_loader = create_data_loaders(
            noisy_dataset,
            batch_size=Config.TRAINING.BATCH_SIZE,
            train_split=Config.DATA.TRAIN_RATIO,
            seed=Config.TRAINING.SEED
        )

        # Evaluate on noisy test set
        noisy_metrics = evaluator.evaluate(noisy_test_loader)
        noise_results[scenario['name']] = noisy_metrics

        print(f"  Accuracy: {noisy_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {noisy_metrics['f1_score']:.4f}")
        print(f"  ROC AUC: {noisy_metrics['roc_auc']:.4f}")

    # Summary
    print("\n7. Summary of Results")
    print("="*60)
    print(f"Clean Test Set:")
    print(f"  Accuracy: {clean_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {clean_metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {clean_metrics['roc_auc']:.4f}")
    print(f"  Inference Time: {inference_time*1000:.2f} ms/sample")

    print("\nNoisy Test Sets:")
    for scenario_name, metrics in noise_results.items():
        print(f"  {scenario_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1_score']:.4f}")
        print(f"    ROC AUC: {metrics['roc_auc']:.4f}")

    # Visualizations
    print("\n8. Generating visualizations...")
    visualizer = FHRVisualizer(
        figsize=Config.VIZ.FIGURE_SIZE,
        dpi=Config.VIZ.FIGURE_DPI
    )

    # Plot shapelet explanations
    visualizer.plot_shapelet_explanation(
        model=model,
        dataset=test_loader.dataset.dataset,
        device=device,
        n_samples=3,
        save_dir=Config.DATA.PLOT_DIR
    )

    # Plot training curves
    if history:
        visualizer.plot_training_curves(
            history,
            save_path=os.path.join(Config.DATA.PLOT_DIR, "training_curves.png")
        )

    print(f"\nVisualizations saved to: {Config.DATA.PLOT_DIR}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()