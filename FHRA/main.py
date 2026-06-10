"""Main entry point for Fetal Heart Rate Analysis"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import (
    DEVICE, DATA_DIR, MODEL_DIR, PLOT_DIR,
    MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG,
    VIZ_CONFIG, USE_LIGHTWEIGHT_MODEL
)
from src.utils import setup_seed, ensure_dir
from src.dataset import FHRDataset, create_data_loaders
from src.models import DualBranchModel, LightweightDualBranchModel
from src.loss import HybridLoss
from src.training import Trainer, Evaluator, initialize_prototypes
from src.visualization import FHRVisualizer


def main():
    """Main training and evaluation pipeline."""
    print("="*60)
    print("FETAL HEART RATE ANOMALY DETECTION")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print("="*60)

    # Set random seed for reproducibility
    setup_seed(TRAINING_CONFIG['seed'])

    # Load dataset
    print("\n[1/5] Loading dataset...")
    try:
        dataset = FHRDataset(
            data_dir=str(DATA_DIR),
            window_size=DATASET_CONFIG['window_size'],
            stride=DATASET_CONFIG['stride']
        )
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(dataset)}")

        # Show class distribution
        class_dist = dataset.get_class_distribution()
        print(f"Class distribution: {class_dist}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Create data loaders
    print("\n[2/5] Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        train_split=TRAINING_CONFIG['train_split'],
        seed=TRAINING_CONFIG['seed']
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    print("\n[3/5] Initializing model...")
    # Select model variant based on config
    if USE_LIGHTWEIGHT_MODEL:
        model = LightweightDualBranchModel(**MODEL_CONFIG).to(DEVICE)
        print("Using Lightweight model variant")
    else:
        model = DualBranchModel(**MODEL_CONFIG).to(DEVICE)
        print("Using Enhanced model with residual connections")

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize prototypes
    train_subset = train_loader.dataset
    initialize_prototypes(
        model=model,
        train_subset=train_subset,
        device=DEVICE,
        n_prototypes=MODEL_CONFIG['num_prototypes']
    )

    # Setup training
    print("\n[4/5] Setting up training...")
    criterion = HybridLoss(
        alpha=TRAINING_CONFIG['loss_alpha'],
        margin=TRAINING_CONFIG['loss_margin']
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=TRAINING_CONFIG['learning_rate']
    )

    # Train model
    print("\n[4/5] Training model...")
    print(f"Training for {TRAINING_CONFIG['epochs']} epochs...")
    print("-"*60)

    model_save_path = MODEL_DIR / "best_model.pth"
    start_time = time.time()

    history = trainer.train(
        criterion=criterion,
        epochs=TRAINING_CONFIG['epochs'],
        save_path=str(model_save_path)
    )

    end_time = time.time()
    training_time = end_time - start_time

    print("-"*60)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best model saved to: {model_save_path}")

    # Evaluate model
    print("\n[5/5] Evaluating model...")
    print("-"*60)

    evaluator = Evaluator(model=model, device=DEVICE)
    metrics, inference_time = evaluator.evaluate_with_timing(test_loader)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = FHRVisualizer(
        figsize=VIZ_CONFIG['figsize'],
        dpi=VIZ_CONFIG['dpi']
    )

    # Plot shapelet explanations
    visualizer.plot_shapelet_explanation(
        model=model,
        dataset=test_loader.dataset,
        device=DEVICE,
        n_samples=VIZ_CONFIG['n_samples'],
        save_dir=str(PLOT_DIR)
    )
    print(f"Shapelet explanations saved to: {PLOT_DIR}")

    # Plot training curves
    if history:
        visualizer.plot_training_curves(
            history=history,
            save_path=str(PLOT_DIR / "training_curves.png")
        )
        print("Training curves saved to:", PLOT_DIR / "training_curves.png")

    # Save final results
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results = {
        'training_time': training_time,
        'inference_time_per_sample': inference_time,
        'metrics': convert_numpy(metrics),
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG
    }

    import json
    with open(MODEL_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Final Results:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 Score: {metrics['f1_score']:.4f}")
    print(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  - Training Time: {training_time:.2f} seconds")
    print(f"  - Inference Time: {inference_time*1000:.2f} ms/sample")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()