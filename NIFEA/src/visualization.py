"""Visualization functions for NIFEA Fetal Heart Rate Analysis"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
from pathlib import Path
from typing import Optional, Tuple


class NIFEAVisualizer:
    """
    Visualizer for NIFEA analysis results and model interpretability.
    """

    def __init__(self,
                 figsize: Tuple[int, int] = (10, 4),
                 dpi: int = 300,
                 style: str = 'seaborn-whitegrid'):
        """
        Initialize the visualizer.

        Args:
            figsize: Figure size for plots
            dpi: Resolution for saved figures
            style: Matplotlib style
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # Color scheme optimized for publication
        self.colors = {
            'signal': '#1f77b4',      # Blue
            'candidate': '#2ca02c',   # Green
            'highlight': '#FFA500',   # Orange-yellow
            'background': '#f7f7f7'   # Light gray
        }

        # Line styles
        self.linestyles = {
            'signal': '-',
            'candidate': '--',
            'highlight': ':'
        }

        # Set up matplotlib
        plt.style.use(style)
        self._setup_rc_params()

    def _setup_rc_params(self) -> None:
        """Configure matplotlib parameters for academic publication."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': self.dpi,
            'figure.figsize': self.figsize,
            'lines.linewidth': 1.5,
            'font.family': 'DejaVu Sans',  # Ensures support for various symbols
            'axes.edgecolor': '0.15',
            'axes.linewidth': 1.2,
            'axes.facecolor': self.colors['background']
        })

    def plot_shapelet_explanation(self,
                                 model: torch.nn.Module,
                                 dataset: torch.utils.data.Dataset,
                                 device: torch.device,
                                 channel: int = 0,
                                 n_samples: int = 3,
                                 save_dir: Optional[str] = None,
                                 show: bool = True) -> None:
        """
        Generate visualization showing shapelet matching in ECG signals.

        Args:
            model: Trained model for generating shapelets
            dataset: Dataset containing ECG signals
            device: Device for computation
            channel: Channel to visualize (0-3 for NIFEA)
            n_samples: Number of samples to visualize
            save_dir: Directory to save plots (optional)
            show: Whether to display the plots
        """
        model.eval()

        # Select random samples
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

        for idx in indices:
            self._plot_single_sample(model, dataset, idx, device, channel, save_dir, show)

    def _plot_single_sample(self,
                           model: torch.nn.Module,
                           dataset: torch.utils.data.Dataset,
                           idx: int,
                           device: torch.device,
                           channel: int,
                           save_dir: Optional[str],
                           show: bool) -> None:
        """Plot a single sample with shapelet matching."""
        x, label = dataset[idx]

        # Extract original signal
        orig_signal = x[channel].numpy()

        # Generate shapelet candidates
        with torch.no_grad():
            x_tensor = x.unsqueeze(0).to(device)
            candidates = model.shapelet_generator(x_tensor)  # Get candidate shapelets
            candidates = candidates.cpu().numpy()[0]  # (num_candidates, L, C)

        # Apply smoothing for better visualization
        smooth_signal = gaussian_filter1d(orig_signal, sigma=2.0)

        # Find best matching candidate
        best_candidate = None
        best_pos = 0
        min_dist = float('inf')
        candidate_length = candidates.shape[1]

        # Iterate over all candidate shapelets
        for i, candidate in enumerate(candidates):
            # Extract and smooth current candidate
            current_candidate = gaussian_filter1d(candidate[:, channel], sigma=1.0)

            # Sliding window matching
            for pos in range(len(smooth_signal) - len(current_candidate) + 1):
                segment = smooth_signal[pos:pos + len(current_candidate)]
                dist = np.linalg.norm(segment - current_candidate)

                if dist < min_dist:
                    min_dist = dist
                    best_candidate = current_candidate
                    best_pos = pos

        # Create visualization
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot original signal
        ax.plot(smooth_signal,
                color=self.colors['signal'],
                linestyle=self.linestyles['signal'],
                alpha=0.9,
                label=f'ECG Channel {channel}',
                linewidth=1.5)

        # Plot best matching region
        if best_candidate is not None:
            ax.axvspan(best_pos, best_pos + len(best_candidate),
                      color=self.colors['highlight'],
                      alpha=0.3,
                      label='Matched Region')

            # Plot candidate shapelet (aligned position)
            x_vals = np.linspace(best_pos, best_pos + len(best_candidate), len(best_candidate))
            ax.plot(x_vals, best_candidate,
                    color=self.colors['candidate'],
                    linestyle=self.linestyles['candidate'],
                    linewidth=1.8,
                    alpha=0.9,
                    label='Best Candidate')

            # Add distance annotation
            ax.text(best_pos + len(best_candidate)/2, np.min(smooth_signal) - 0.5,
                   f'Distance: {min_dist:.2f}',
                   ha='center', va='top', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))

        # Chart decoration
        ax.set_xlabel("Time Steps", fontweight='bold')
        ax.set_ylabel("Normalized Value", fontweight='bold')
        ax.set_title(f"Sample {idx} - {'Arrhythmia' if label else 'Normal'}",
                    fontweight='bold')

        # Smart legend handling
        handles, labels = ax.get_legend_handles_labels()
        legend_dict = {label: handle for label, handle in zip(labels, handles)}
        ax.legend(legend_dict.values(), legend_dict.keys(),
                 loc='upper right',
                 frameon=True,
                 shadow=True,
                 fancybox=True,
                 borderpad=0.8)

        # Beautify axes
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(np.min(smooth_signal) - 1, np.max(smooth_signal) + 1)

        plt.tight_layout()

        # Save plot
        if save_dir:
            self._save_plot(fig, idx, channel, save_dir)

        if show:
            plt.show()
        else:
            plt.close()

    def _save_plot(self, fig: plt.Figure, idx: int, channel: int, save_dir: str) -> None:
        """Save the plot in multiple formats."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save as PNG (raster format)
        fig.savefig(save_path / f"sample_{idx}_ch{channel}_shapelet.png",
                   dpi=self.dpi,
                   bbox_inches='tight',
                   format='png',
                   transparent=True)

        # Save as PDF (vector format) if available
        try:
            fig.savefig(save_path / f"sample_{idx}_ch{channel}_shapelet.pdf",
                       dpi=self.dpi,
                       bbox_inches='tight',
                       format='pdf')
        except:
            pass  # PDF might not be available on all systems

    def plot_training_curves(self,
                           history: dict,
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
        """
        Plot training curves for loss and metrics.

        Args:
            history: Dictionary containing training history
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Cross-entropy loss
        if 'train_ce_loss' in history and 'val_ce_loss' in history:
            axes[0, 1].plot(history['train_ce_loss'], label='Train CE Loss', color='blue')
            axes[0, 1].plot(history['val_ce_loss'], label='Val CE Loss', color='red')
            axes[0, 1].set_title('Cross-Entropy Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('CE Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Prototype loss
        if 'train_proto_loss' in history and 'val_proto_loss' in history:
            axes[1, 0].plot(history['train_proto_loss'], label='Train Proto Loss', color='green')
            axes[1, 0].plot(history['val_proto_loss'], label='Val Proto Loss', color='orange')
            axes[1, 0].set_title('Prototype Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Proto Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[1, 1].plot(history['train_acc'], label='Train Acc', color='blue')
            axes[1, 1].plot(history['val_acc'], label='Val Acc', color='red')
            axes[1, 1].set_title('Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()