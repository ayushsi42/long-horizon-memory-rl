from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor

from ..memory import HierarchicalMemory


class MemoryAnalyzer:
    """Tools for analyzing memory behavior and patterns."""

    def __init__(self, memory: HierarchicalMemory):
        """Initialize memory analyzer.

        Args:
            memory: Hierarchical memory instance
        """
        self.memory = memory

    def visualize_memory_structure(
        self,
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize hierarchical memory structure.

        Args:
            save_path: Optional path to save visualization
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle("Memory Structure Analysis", fontsize=16)

        # Plot immediate memory
        self._plot_memory_embeddings(
            self.memory.immediate.embeddings[:self.memory.immediate.current_size],
            "Immediate Memory",
            axes[0, 0],
        )

        # Plot short-term memory
        if self.memory.short_term.compressed:
            embeddings = self.memory.short_term.compressed_embeddings
        else:
            embeddings = self.memory.short_term.embeddings
        self._plot_memory_embeddings(
            embeddings[:self.memory.short_term.current_size],
            "Short-Term Memory",
            axes[0, 1],
        )

        # Plot long-term memory
        if self.memory.long_term.templates:
            template_embeddings = torch.stack([
                t.embedding for t in self.memory.long_term.templates
            ])
            self._plot_memory_embeddings(
                template_embeddings,
                "Long-Term Memory",
                axes[1, 0],
            )

        # Plot memory usage over time
        self._plot_memory_usage(axes[1, 1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def analyze_retrieval_patterns(
        self,
        episode_data: List[Dict[str, Tensor]],
        save_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Analyze memory retrieval patterns.

        Args:
            episode_data: List of episode transitions
            save_path: Optional path to save visualization

        Returns:
            Dictionary of analysis results
        """
        # Extract retrieval data
        retrievals = []
        attention_weights = []
        timestamps = []

        for data in episode_data:
            if "memory_retrieval" in data:
                retrievals.append(data["memory_retrieval"])
                attention_weights.append(data["attention_weights"])
                timestamps.append(data["timestamp"])

        retrievals = torch.stack(retrievals)
        attention_weights = np.stack(attention_weights)
        timestamps = np.array(timestamps)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle("Memory Retrieval Analysis", fontsize=16)

        # Plot retrieval trajectories
        self._plot_retrieval_trajectories(
            retrievals,
            timestamps,
            axes[0, 0],
        )

        # Plot attention patterns
        self._plot_attention_patterns(
            attention_weights,
            timestamps,
            axes[0, 1],
        )

        # Plot retrieval statistics
        self._plot_retrieval_stats(
            retrievals,
            attention_weights,
            axes[1, 0],
        )

        # Plot temporal analysis
        self._plot_temporal_analysis(
            retrievals,
            attention_weights,
            timestamps,
            axes[1, 1],
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

        # Return analysis results
        return {
            "retrievals": retrievals.detach().cpu().numpy(),
            "attention_weights": attention_weights,
            "temporal_stats": self._compute_temporal_stats(
                retrievals,
                attention_weights,
                timestamps,
            ),
        }

    def analyze_compression_efficiency(
        self,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Analyze memory compression efficiency.

        Args:
            save_path: Optional path to save visualization

        Returns:
            Dictionary of compression metrics
        """
        metrics = {}

        # Calculate compression ratios
        if self.memory.short_term.compressed:
            metrics["compression_ratio"] = (
                self.memory.short_term.compressed_embeddings.shape[1] /
                self.memory.short_term.embeddings.shape[1]
            )

            # Calculate reconstruction error
            with torch.no_grad():
                reconstructed = self.memory.short_term.compressor.decode(
                    self.memory.short_term.compressed_embeddings
                )
                metrics["reconstruction_error"] = torch.nn.functional.mse_loss(
                    reconstructed,
                    self.memory.short_term.embeddings,
                ).item()

        # Create visualization
        if save_path:
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle("Compression Efficiency Analysis", fontsize=16)

            # Plot original vs compressed embeddings
            self._plot_compression_comparison(axes[0, 0])

            # Plot reconstruction quality
            self._plot_reconstruction_quality(axes[0, 1])

            # Plot compression statistics
            self._plot_compression_stats(axes[1, 0])

            # Plot memory savings
            self._plot_memory_savings(axes[1, 1])

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return metrics

    def _plot_memory_embeddings(
        self,
        embeddings: Tensor,
        title: str,
        ax: plt.Axes,
    ) -> None:
        """Plot memory embeddings using dimensionality reduction.

        Args:
            embeddings: Memory embeddings
            title: Plot title
            ax: Matplotlib axes
        """
        if len(embeddings) == 0:
            ax.text(
                0.5,
                0.5,
                "No embeddings",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title(title)
            return

        # Reduce dimensionality for visualization
        embeddings_np = embeddings.detach().cpu().numpy()
        if embeddings_np.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings_np)
        else:
            embeddings_2d = embeddings_np

        # Create scatter plot
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=np.arange(len(embeddings_2d)),
            cmap="viridis",
        )
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label="Time")

    def _plot_memory_usage(self, ax: plt.Axes) -> None:
        """Plot memory usage statistics.

        Args:
            ax: Matplotlib axes
        """
        # Collect usage statistics
        usage_data = {
            "Immediate": self.memory.immediate.current_size / self.memory.immediate.capacity,
            "Short-Term": self.memory.short_term.current_size / self.memory.short_term.capacity,
            "Long-Term": len(self.memory.long_term.templates) / self.memory.long_term.capacity,
        }

        # Create bar plot
        bars = ax.bar(usage_data.keys(), usage_data.values())
        ax.set_title("Memory Usage")
        ax.set_ylabel("Utilization")
        ax.set_ylim(0, 1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

    def _plot_retrieval_trajectories(
        self,
        retrievals: Tensor,
        timestamps: np.ndarray,
        ax: plt.Axes,
    ) -> None:
        """Plot memory retrieval trajectories.

        Args:
            retrievals: Retrieved embeddings
            timestamps: Retrieval timestamps
            ax: Matplotlib axes
        """
        # Reduce dimensionality for visualization
        retrievals_np = retrievals.detach().cpu().numpy()
        if retrievals_np.shape[1] > 2:
            pca = PCA(n_components=2)
            retrievals_2d = pca.fit_transform(retrievals_np)
        else:
            retrievals_2d = retrievals_np

        # Plot trajectories
        ax.plot(retrievals_2d[:, 0], retrievals_2d[:, 1], "b-", alpha=0.5)
        scatter = ax.scatter(
            retrievals_2d[:, 0],
            retrievals_2d[:, 1],
            c=timestamps,
            cmap="viridis",
        )
        ax.set_title("Retrieval Trajectories")
        plt.colorbar(scatter, ax=ax, label="Time")

    def _plot_attention_patterns(
        self,
        attention_weights: np.ndarray,
        timestamps: np.ndarray,
        ax: plt.Axes,
    ) -> None:
        """Plot attention weight patterns.

        Args:
            attention_weights: Attention weights
            timestamps: Attention timestamps
            ax: Matplotlib axes
        """
        # Create heatmap
        sns.heatmap(
            attention_weights.T,
            ax=ax,
            cmap="viridis",
            xticklabels=False,
        )
        ax.set_title("Attention Patterns")
        ax.set_xlabel("Time")
        ax.set_ylabel("Memory Index")

    def _plot_retrieval_stats(
        self,
        retrievals: Tensor,
        attention_weights: np.ndarray,
        ax: plt.Axes,
    ) -> None:
        """Plot retrieval statistics.

        Args:
            retrievals: Retrieved embeddings
            attention_weights: Attention weights
            ax: Matplotlib axes
        """
        # Calculate statistics
        retrieval_norms = torch.norm(retrievals, dim=1).cpu().numpy()
        attention_entropy = -np.sum(
            attention_weights * np.log(attention_weights + 1e-10),
            axis=1,
        )

        # Create scatter plot
        ax.scatter(retrieval_norms, attention_entropy, alpha=0.5)
        ax.set_title("Retrieval Statistics")
        ax.set_xlabel("Retrieval Norm")
        ax.set_ylabel("Attention Entropy")

    def _plot_temporal_analysis(
        self,
        retrievals: Tensor,
        attention_weights: np.ndarray,
        timestamps: np.ndarray,
        ax: plt.Axes,
    ) -> None:
        """Plot temporal analysis.

        Args:
            retrievals: Retrieved embeddings
            attention_weights: Attention weights
            timestamps: Retrieval timestamps
            ax: Matplotlib axes
        """
        # Calculate temporal metrics
        retrieval_change = torch.norm(
            retrievals[1:] - retrievals[:-1],
            dim=1,
        ).cpu().numpy()
        attention_change = np.linalg.norm(
            attention_weights[1:] - attention_weights[:-1],
            axis=1,
        )

        # Create line plot
        ax.plot(timestamps[1:], retrieval_change, label="Retrieval Change")
        ax.plot(timestamps[1:], attention_change, label="Attention Change")
        ax.set_title("Temporal Analysis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Change Magnitude")
        ax.legend()

    def _plot_compression_comparison(self, ax: plt.Axes) -> None:
        """Plot comparison of original and compressed embeddings.

        Args:
            ax: Matplotlib axes
        """
        if not self.memory.short_term.compressed:
            ax.text(
                0.5,
                0.5,
                "Memory not compressed",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title("Compression Comparison")
            return

        # Get original and compressed embeddings
        original = self.memory.short_term.embeddings[:self.memory.short_term.current_size]
        compressed = self.memory.short_term.compressed_embeddings[:self.memory.short_term.current_size]

        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        original_2d = pca.fit_transform(original.detach().cpu().numpy())
        compressed_2d = pca.transform(compressed.detach().cpu().numpy())

        # Plot embeddings
        ax.scatter(
            original_2d[:, 0],
            original_2d[:, 1],
            label="Original",
            alpha=0.5,
        )
        ax.scatter(
            compressed_2d[:, 0],
            compressed_2d[:, 1],
            label="Compressed",
            alpha=0.5,
        )
        ax.set_title("Compression Comparison")
        ax.legend()

    def _plot_reconstruction_quality(self, ax: plt.Axes) -> None:
        """Plot reconstruction quality analysis.

        Args:
            ax: Matplotlib axes
        """
        if not self.memory.short_term.compressed:
            ax.text(
                0.5,
                0.5,
                "Memory not compressed",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title("Reconstruction Quality")
            return

        # Get original and reconstructed embeddings
        original = self.memory.short_term.embeddings[:self.memory.short_term.current_size]
        with torch.no_grad():
            reconstructed = self.memory.short_term.compressor.decode(
                self.memory.short_term.compressed_embeddings[:self.memory.short_term.current_size]
            )

        # Calculate reconstruction errors
        errors = torch.norm(
            original - reconstructed,
            dim=1,
        ).cpu().numpy()

        # Plot error distribution
        sns.histplot(errors, ax=ax)
        ax.set_title("Reconstruction Quality")
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Count")

    def _plot_compression_stats(self, ax: plt.Axes) -> None:
        """Plot compression statistics.

        Args:
            ax: Matplotlib axes
        """
        if not self.memory.short_term.compressed:
            ax.text(
                0.5,
                0.5,
                "Memory not compressed",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title("Compression Statistics")
            return

        # Calculate statistics
        original_dim = self.memory.short_term.embeddings.shape[1]
        compressed_dim = self.memory.short_term.compressed_embeddings.shape[1]
        compression_ratio = compressed_dim / original_dim

        stats = {
            "Original Dim": original_dim,
            "Compressed Dim": compressed_dim,
            "Compression Ratio": compression_ratio,
        }

        # Create bar plot
        bars = ax.bar(stats.keys(), stats.values())
        ax.set_title("Compression Statistics")
        ax.set_ylabel("Value")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

    def _plot_memory_savings(self, ax: plt.Axes) -> None:
        """Plot memory savings analysis.

        Args:
            ax: Matplotlib axes
        """
        if not self.memory.short_term.compressed:
            ax.text(
                0.5,
                0.5,
                "Memory not compressed",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title("Memory Savings")
            return

        # Calculate memory usage
        original_size = (
            self.memory.short_term.embeddings.element_size() *
            self.memory.short_term.embeddings.nelement()
        )
        compressed_size = (
            self.memory.short_term.compressed_embeddings.element_size() *
            self.memory.short_term.compressed_embeddings.nelement()
        )
        savings = 1 - (compressed_size / original_size)

        # Create pie chart
        ax.pie(
            [compressed_size, original_size - compressed_size],
            labels=["Used", "Saved"],
            autopct="%1.1f%%",
        )
        ax.set_title(f"Memory Savings: {savings:.1%}")

    def _compute_temporal_stats(
        self,
        retrievals: Tensor,
        attention_weights: np.ndarray,
        timestamps: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute temporal statistics.

        Args:
            retrievals: Retrieved embeddings
            attention_weights: Attention weights
            timestamps: Retrieval timestamps

        Returns:
            Dictionary of temporal statistics
        """
        # Calculate retrieval changes
        retrieval_changes = torch.norm(
            retrievals[1:] - retrievals[:-1],
            dim=1,
        ).cpu().numpy()

        # Calculate attention changes
        attention_changes = np.linalg.norm(
            attention_weights[1:] - attention_weights[:-1],
            axis=1,
        )

        # Calculate time intervals
        time_intervals = timestamps[1:] - timestamps[:-1]

        return {
            "retrieval_changes": retrieval_changes,
            "attention_changes": attention_changes,
            "time_intervals": time_intervals,
            "change_rates": {
                "retrieval": retrieval_changes / time_intervals,
                "attention": attention_changes / time_intervals,
            },
        }
