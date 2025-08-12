from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ..memory import HierarchicalMemory


class ValidationMetrics:
    """Metrics for validating memory-augmented RL performance."""

    def __init__(self, memory: HierarchicalMemory):
        """Initialize validation metrics.

        Args:
            memory: Hierarchical memory instance
        """
        self.memory = memory

    def memory_efficiency(self) -> Dict[str, float]:
        """Calculate memory efficiency metrics.

        Returns:
            Dictionary of memory efficiency metrics
        """
        # Calculate memory usage
        immediate_usage = (
            self.memory.immediate.current_size /
            self.memory.immediate.capacity
        )
        short_term_usage = (
            self.memory.short_term.current_size /
            self.memory.short_term.capacity
        )
        long_term_usage = (
            len(self.memory.long_term.templates) /
            self.memory.long_term.capacity
        )

        # Calculate compression ratios
        short_term_ratio = (
            self.memory.short_term.compressed_embeddings.shape[1] /
            self.memory.short_term.embeddings.shape[1]
            if self.memory.short_term.compressed
            else 1.0
        )

        # Calculate memory coherence
        immediate_coherence = self._calculate_memory_coherence(
            self.memory.immediate.embeddings[:self.memory.immediate.current_size]
        )
        short_term_coherence = self._calculate_memory_coherence(
            self.memory.short_term.embeddings[:self.memory.short_term.current_size]
        )
        long_term_coherence = self._calculate_memory_coherence(
            torch.stack([t.embedding for t in self.memory.long_term.templates])
            if self.memory.long_term.templates
            else torch.zeros(0)
        )

        return {
            "immediate_usage": immediate_usage,
            "short_term_usage": short_term_usage,
            "long_term_usage": long_term_usage,
            "short_term_compression": short_term_ratio,
            "immediate_coherence": immediate_coherence,
            "short_term_coherence": short_term_coherence,
            "long_term_coherence": long_term_coherence,
        }

    def retrieval_quality(
        self,
        queries: Tensor,
        expected: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Calculate retrieval quality metrics.

        Args:
            queries: Query tensors
            expected: Optional expected retrievals

        Returns:
            Dictionary of retrieval quality metrics
        """
        # Get retrievals from each memory level
        immediate_retrievals, immediate_meta = self.memory.immediate.read(queries)
        short_term_retrievals, short_term_meta = self.memory.short_term.read(queries)
        long_term_retrievals, long_term_meta = self.memory.long_term.read(queries)

        metrics = {}

        # Calculate retrieval diversity
        metrics["immediate_diversity"] = self._calculate_diversity(
            immediate_retrievals
        )
        metrics["short_term_diversity"] = self._calculate_diversity(
            short_term_retrievals
        )
        metrics["long_term_diversity"] = self._calculate_diversity(
            long_term_retrievals
        )

        # Calculate attention statistics
        if immediate_meta and "attention_weights" in immediate_meta:
            metrics["immediate_attention"] = self._analyze_attention(
                immediate_meta["attention_weights"]
            )
        if short_term_meta and "attention_weights" in short_term_meta:
            metrics["short_term_attention"] = self._analyze_attention(
                short_term_meta["attention_weights"]
            )
        if long_term_meta and "attention_weights" in long_term_meta:
            metrics["long_term_attention"] = self._analyze_attention(
                long_term_meta["attention_weights"]
            )

        # Calculate retrieval accuracy if expected retrievals provided
        if expected is not None:
            metrics["immediate_accuracy"] = self._calculate_accuracy(
                immediate_retrievals,
                expected,
            )
            metrics["short_term_accuracy"] = self._calculate_accuracy(
                short_term_retrievals,
                expected,
            )
            metrics["long_term_accuracy"] = self._calculate_accuracy(
                long_term_retrievals,
                expected,
            )

        return metrics

    def temporal_analysis(
        self,
        episode_data: List[Dict[str, Tensor]],
    ) -> Dict[str, List[float]]:
        """Analyze temporal aspects of memory usage.

        Args:
            episode_data: List of episode transitions

        Returns:
            Dictionary of temporal metrics
        """
        # Initialize metrics
        metrics = {
            "immediate_usage": [],
            "short_term_usage": [],
            "long_term_usage": [],
            "compression_frequency": [],
            "consolidation_frequency": [],
            "retrieval_staleness": [],
        }

        # Track memory usage over time
        for step_data in episode_data:
            metrics["immediate_usage"].append(
                self.memory.immediate.current_size /
                self.memory.immediate.capacity
            )
            metrics["short_term_usage"].append(
                self.memory.short_term.current_size /
                self.memory.short_term.capacity
            )
            metrics["long_term_usage"].append(
                len(self.memory.long_term.templates) /
                self.memory.long_term.capacity
            )

        # Calculate compression frequency
        compression_events = [
            i for i, data in enumerate(episode_data)
            if data.get("compressed", False)
        ]
        metrics["compression_frequency"] = [
            compression_events[i + 1] - compression_events[i]
            for i in range(len(compression_events) - 1)
        ] if len(compression_events) > 1 else []

        # Calculate consolidation frequency
        consolidation_events = [
            i for i, data in enumerate(episode_data)
            if data.get("consolidated", False)
        ]
        metrics["consolidation_frequency"] = [
            consolidation_events[i + 1] - consolidation_events[i]
            for i in range(len(consolidation_events) - 1)
        ] if len(consolidation_events) > 1 else []

        # Calculate retrieval staleness
        for data in episode_data:
            if "retrieval_age" in data:
                metrics["retrieval_staleness"].append(
                    data["retrieval_age"]
                )

        return metrics

    def _calculate_memory_coherence(self, embeddings: Tensor) -> float:
        """Calculate coherence of memory embeddings.

        Args:
            embeddings: Memory embeddings

        Returns:
            Coherence score
        """
        if len(embeddings) == 0:
            return 0.0

        # Calculate pairwise similarities
        similarities = torch.matmul(embeddings, embeddings.T)
        similarities = similarities / torch.norm(
            embeddings, dim=1
        ).unsqueeze(0) / torch.norm(
            embeddings, dim=1
        ).unsqueeze(1)

        # Calculate average similarity excluding self-similarity
        mask = ~torch.eye(len(embeddings), dtype=torch.bool)
        coherence = similarities[mask].mean().item()

        return coherence

    def _calculate_diversity(self, embeddings: Tensor) -> float:
        """Calculate diversity of retrieved embeddings.

        Args:
            embeddings: Retrieved embeddings

        Returns:
            Diversity score
        """
        if len(embeddings) == 0:
            return 0.0

        # Calculate pairwise distances
        distances = torch.cdist(embeddings, embeddings)
        
        # Calculate average distance excluding self-distance
        mask = ~torch.eye(len(embeddings), dtype=torch.bool)
        diversity = distances[mask].mean().item()

        return diversity

    def _analyze_attention(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Analyze attention weight patterns.

        Args:
            attention_weights: Attention weight matrix

        Returns:
            Dictionary of attention metrics
        """
        # Calculate entropy of attention distribution
        entropy = -np.sum(
            attention_weights * np.log(attention_weights + 1e-10),
            axis=-1,
        ).mean()

        # Calculate attention concentration (Gini coefficient)
        sorted_weights = np.sort(attention_weights, axis=-1)
        n = attention_weights.shape[-1]
        indices = np.arange(1, n + 1)
        concentration = (
            (np.sum(indices * sorted_weights, axis=-1) / n) -
            (n + 1) / (2 * n)
        ).mean()

        # Calculate top-k attention ratio
        top_k = 3
        top_k_ratio = np.mean(
            np.sort(attention_weights, axis=-1)[:, -top_k:].sum(axis=-1)
        )

        return {
            "entropy": entropy,
            "concentration": concentration,
            "top_k_ratio": top_k_ratio,
        }

    def _calculate_accuracy(
        self,
        retrieved: Tensor,
        expected: Tensor,
        threshold: float = 0.8,
    ) -> float:
        """Calculate retrieval accuracy.

        Args:
            retrieved: Retrieved embeddings
            expected: Expected embeddings
            threshold: Similarity threshold

        Returns:
            Accuracy score
        """
        # Calculate cosine similarity
        similarities = torch.matmul(
            retrieved / torch.norm(retrieved, dim=1, keepdim=True),
            (expected / torch.norm(expected, dim=1, keepdim=True)).T,
        )

        # Calculate accuracy
        accuracy = (similarities.max(dim=1)[0] > threshold).float().mean().item()

        return accuracy


class ResourceMetrics:
    """Metrics for monitoring resource usage."""

    @staticmethod
    def calculate_memory_footprint(
        memory: HierarchicalMemory,
    ) -> Dict[str, int]:
        """Calculate memory footprint in bytes.

        Args:
            memory: Hierarchical memory instance

        Returns:
            Dictionary of memory usage statistics
        """
        # Calculate immediate memory usage
        immediate_size = (
            memory.immediate.embeddings.element_size() *
            memory.immediate.embeddings.nelement()
        )

        # Calculate short-term memory usage
        short_term_size = (
            memory.short_term.embeddings.element_size() *
            memory.short_term.embeddings.nelement()
        )
        if memory.short_term.compressed:
            short_term_size += (
                memory.short_term.compressed_embeddings.element_size() *
                memory.short_term.compressed_embeddings.nelement()
            )

        # Calculate long-term memory usage
        long_term_size = sum(
            template.embedding.element_size() * template.embedding.nelement()
            for template in memory.long_term.templates
        )

        # Calculate total size
        total_size = immediate_size + short_term_size + long_term_size

        return {
            "immediate_memory": immediate_size,
            "short_term_memory": short_term_size,
            "long_term_memory": long_term_size,
            "total_memory": total_size,
        }

    @staticmethod
    def calculate_computation_time(
        episode_data: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Calculate computation time statistics.

        Args:
            episode_data: List of episode timing data

        Returns:
            Dictionary of timing statistics
        """
        metrics = {}

        # Calculate average times
        if episode_data:
            metrics["policy_time"] = np.mean([
                d["policy_time"] for d in episode_data
            ])
            metrics["memory_time"] = np.mean([
                d["memory_time"] for d in episode_data
            ])
            metrics["exploration_time"] = np.mean([
                d["exploration_time"] for d in episode_data
            ])
            metrics["total_time"] = np.mean([
                d["total_time"] for d in episode_data
            ])

            # Calculate time distributions
            metrics["policy_ratio"] = metrics["policy_time"] / metrics["total_time"]
            metrics["memory_ratio"] = metrics["memory_time"] / metrics["total_time"]
            metrics["exploration_ratio"] = (
                metrics["exploration_time"] / metrics["total_time"]
            )

        return metrics
