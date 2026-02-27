# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FETALET is a deep learning project for fetal heart rate (FHR) anomaly detection using a dual-branch neural architecture combining CNN features with prototype-based shapelet matching. The project supports two datasets:

- **FHRA** (`FHRA/`): Private dataset using CSV files with FHR and TOCO signals (2 channels)
- **NIFEA** (`NIFEA/`): Public dataset using WFDB format files with 4-channel ECG signals

## Running the Project

### FHRA (Private Dataset)

```bash
# Standard training and evaluation
python FHRA/main.py

# Training with noise scenarios
python FHRA/main_noise.py
```

### NIFEA (Public Dataset)

```bash
python NIFEA/main.py
```

## Architecture Overview

### Core Model: DualBranchModel

The model (`src/models.py`) uses two parallel branches:

1. **CNN Branch**: Extracts general features via Conv1d layers with pooling
2. **Prototype Branch**: Matches learnable shapelet prototypes against candidate shapelets generated from input

Key components:
- `CandidateGenerator`: CNN that generates candidate shapelets from input time series
- `prototype_layer`: Learnable parameters of shape `(num_prototypes, shapelet_length, in_channels)`
- Hybrid loss combining cross-entropy and prototype-based distance loss

### Dataset Branches

Both datasets share the same model architecture but use different dataset loaders:

| Feature | FHRA | NIFEA |
|---------|------|-------|
| Input format | CSV files | WFDB .dat/.hea files |
| Channels | 2 (FHR, TOCO) | 4 (ECG signals) |
| Dataset class | `FHRDataset` | `NIFEADataset` |
| Labels | From "label" column | From filename (ARR* = abnormal) |

### Configuration Pattern

- **FHRA**: Uses `config.py` with flat dictionaries (`MODEL_CONFIG`, `TRAINING_CONFIG`, etc.)
- **NIFEA**: Uses nested `Config` class with sub-configs (`Config.DATA`, `Config.MODEL`, etc.)

## Module Organization

Each dataset directory has an identical `src/` structure:
- `config.py` / `config_noise.py`: Hyperparameters and paths
- `dataset.py`: Dataset loaders (FHRDataset/NIFEADataset)
- `models.py`: Neural network architectures
- `loss.py`: Loss functions (HybridLoss, ContrastivePrototypeLoss)
- `training.py`: Trainer and Evaluator classes
- `visualization.py`: Plotting utilities (FHRVisualizer/NIFEAVisualizer)
- `utils.py`: Helper functions (setup_seed, ensure_dir, etc.)
- `noise.py` (FHRA only): NoiseFactory for adding signal noise

## Key Training Flow

1. Dataset creates sliding windows with stride
2. Prototypes initialized via K-means on abnormal samples (`initialize_prototypes()`)
3. `Trainer.train()` runs epochs with `HybridLoss`
4. Best model saved based on validation accuracy
5. `Evaluator` produces metrics (accuracy, F1, ROC-AUC)

## Output Directories

- `models/`: Saved model checkpoints
- `plot/` or `outputs/`: Visualizations and training curves
- `data/`: Input data location

## Dependencies

- PyTorch for deep learning
- pandas (FHRA only): CSV loading
- wfdb (NIFEA only): PhysioNet file format
- sklearn: Metrics and clustering (K-means for prototype init)
