"""Visualization functions for Fetal Heart Rate Analysis"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class FHRVisualizer:
    """
    Visualizer for FHR analysis results and model interpretability.
    """

    def __init__(self,
                 figsize: Tuple[int, int] = (16, 10),
                 dpi: int = 600,
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

        # ColorBrewer colorblind-friendly color scheme
        self.colors = {
            'fhr': '#1f78b4',      # Blue
            'toco': '#e31a1c',     # Red
            'shapelet': '#33a02c',  # Green
            'highlight': '#fdbf6f',  # Orange
            'background': '#f7f7f7'  # Light gray
        }

        # Line styles
        self.linestyles = {
            'fhr': '-',
            'toco': '-',
            'shapelet': '--',
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
            'lines.linewidth': 2.0,
            'font.family': 'Arial',
            'axes.edgecolor': '0.15',
            'axes.linewidth': 1.2,
            'axes.facecolor': self.colors['background']
        })

    def plot_shapelet_explanation(self,
                                 model: torch.nn.Module,
                                 dataset: torch.utils.data.Dataset,
                                 device: torch.device,
                                 n_samples: int = 3,
                                 save_dir: Optional[str] = None) -> None:
        """
        Generate visualization showing shapelet matching in FHR signals.

        Args:
            model: Trained model for generating shapelets
            dataset: Dataset containing FHR signals
            device: Device for computation
            n_samples: Number of samples to visualize
            save_dir: Directory to save plots (optional)
        """
        model.eval()

        # Select random samples
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

        for idx in indices:
            self._plot_single_sample(model, dataset, idx, device, save_dir)

    def _plot_single_sample(self,
                           model: torch.nn.Module,
                           dataset: torch.utils.data.Dataset,
                           idx: int,
                           device: torch.device,
                           save_dir: Optional[str]) -> None:
        """Plot a single sample with shapelet matching."""
        x, label = dataset[idx]

        # Extract original signals
        orig_fhr = x[0].numpy()
        orig_toco = x[1].numpy()

        # Generate shapelet candidates
        with torch.no_grad():
            x_tensor = x.unsqueeze(0).to(device)
            candidates = model.shapelet_generator(x_tensor).cpu().numpy()[0]
            candidates = candidates[..., 0]  # Keep only FHR channel

        # Apply smoothing for better visualization
        smooth_fhr = gaussian_filter1d(orig_fhr, sigma=1.5)
        smooth_toco = gaussian_filter1d(orig_toco, sigma=1.5)

        # Find best matching position
        best_pos, best_candidate, min_dist = self._find_best_match(
            smooth_fhr, candidates
        )

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        # Highlight anomaly region
        for ax in [ax1, ax2]:
            ax.axvspan(best_pos, best_pos + len(best_candidate),
                      color=self.colors['highlight'], alpha=0.3, zorder=0)

        # Plot FHR signal
        ax1.plot(smooth_fhr,
                color=self.colors['fhr'],
                linestyle=self.linestyles['fhr'],
                alpha=0.9,
                label='FHR Signal',
                linewidth=2.0)

        # Plot best matching shapelet
        x_vals = np.linspace(best_pos, best_pos + len(best_candidate), len(best_candidate))
        ax1.plot(x_vals, best_candidate,
                color=self.colors['shapelet'],
                linestyle=self.linestyles['shapelet'],
                linewidth=2.2,
                alpha=0.9,
                label='Learned Shapelet')

        # Add annotation
        ax1.annotate('Key anomaly pattern',
                    xy=(best_pos + len(best_candidate)/2, np.max(best_candidate)),
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

        # Plot TOCO signal
        ax2.plot(smooth_toco,
                color=self.colors['toco'],
                linestyle=self.linestyles['toco'],
                alpha=0.9,
                label='TOCO Signal',
                linewidth=2.0)

        # Add label information
        if label == 1:
            ax1.text(0.05, 0.9, 'Late Deceleration',
                    transform=ax1.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8,
                             edgecolor='#cccccc', boxstyle='round,pad=0.3'))

        # Styling
        self._style_subplots(ax1, ax2, idx)

        # Save plot
        if save_dir:
            self._save_plot(fig, idx, save_dir)

        plt.close()

    def _find_best_match(self,
                        signal: np.ndarray,
                        candidates: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Find the best matching shapelet position in the signal.

        Args:
            signal: FHR signal array
            candidates: Candidate shapelets (num_candidates, shapelet_length)

        Returns:
            Tuple of (best_position, best_candidate, min_distance)
        """
        best_pos = 0
        min_dist = float('inf')
        best_candidate = None
        candidate_length = candidates.shape[1]

        for candidate in candidates:
            smooth_candidate = gaussian_filter1d(candidate, sigma=0.5)
            for pos in range(len(signal) - candidate_length + 1):
                segment = signal[pos:pos + candidate_length]
                dist = np.linalg.norm(segment - smooth_candidate)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = pos
                    best_candidate = smooth_candidate

        return best_pos, best_candidate, min_dist

    def _style_subplots(self, ax1: plt.Axes, ax2: plt.Axes, idx: int) -> None:
        """Apply styling to subplots."""
        # Titles and labels
        ax1.set_title(f"FHR Anomaly Detection Example (Sample {idx})",
                     fontweight='bold',
                     pad=15)
        ax1.set_ylabel("FHR (bpm)", fontweight='bold', labelpad=10)
        ax2.set_ylabel("TOCO", fontweight='bold', labelpad=10)
        ax2.set_xlabel("Time (seconds)", fontweight='bold', labelpad=10)

        # Time ticks (assuming 4 samples per second)
        time_ticks = np.arange(0, 100, 60)  # Every 15 seconds
        time_labels = [f"{int(t/4)}" for t in time_ticks]
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)

        # Grid and spines
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', length=4, width=1.0)

        # Combined legend
        self._add_combined_legend(ax1, ax2)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.subplots_adjust(hspace=0.2)

    def _add_combined_legend(self, ax1: plt.Axes, ax2: plt.Axes) -> None:
        """Add combined legend to the figure."""
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Remove duplicates and add highlight
        from matplotlib.lines import Line2D
        all_handles = handles1 + handles2 + [
            Line2D([0], [0], color='gray', alpha=0.3, linewidth=10, label='Anomaly Region')
        ]
        all_labels = labels1 + labels2 + ['Anomaly Region']

        # Remove duplicates
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(all_handles, all_labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)

        ax1.figure.legend(unique_handles, unique_labels,
                          loc='lower center',
                          bbox_to_anchor=(0.5, 0.01),
                          ncol=4,
                          frameon=True,
                          framealpha=0.95,
                          edgecolor='0.8',
                          columnspacing=1.0)

    def _save_plot(self, fig: plt.Figure, idx: int, save_dir: str) -> None:
        """Save the plot in multiple formats."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save as PDF (vector format)
        fig.savefig(save_path / f"{idx}_shapelet_explanation.pdf",
                   dpi=self.dpi,
                   bbox_inches='tight',
                   format='pdf')

        # Save as PNG (raster format)
        fig.savefig(save_path / f"{idx}_shapelet_explanation.png",
                   dpi=self.dpi,
                   bbox_inches='tight',
                   format='png',
                   transparent=True)

    def plot_training_curves(self,
                           history: Dict[str, list],
                           save_path: Optional[str] = None) -> None:
        """
        Plot training curves for loss and metrics.

        Args:
            history: Dictionary containing training history
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        if 'total_loss' in history:
            axes[0, 0].plot(history['total_loss'], label='Total Loss', color='blue')
        if 'ce_loss' in history:
            axes[0, 0].plot(history['ce_loss'], label='CE Loss', color='red')
        if 'proto_loss' in history:
            axes[0, 0].plot(history['proto_loss'], label='Proto Loss', color='green')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        if 'train_acc' in history:
            axes[0, 1].plot(history['train_acc'], label='Train Acc', color='blue')
        if 'val_acc' in history:
            axes[0, 1].plot(history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], color='purple')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)

        # Remove empty subplot
        fig.delaxes(axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.close()