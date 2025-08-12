from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .base import CompressibleMemory


class StrategyTemplate:
    """Represents a strategic pattern in long-term memory."""

    def __init__(
        self,
        embedding: Tensor,
        success_rate: float = 0.0,
        usage_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a strategy template.

        Args:
            embedding: Template embedding
            success_rate: Historical success rate
            usage_count: Number of times template was used
            metadata: Additional metadata
        """
        self.embedding = embedding
        self.success_rate = success_rate
        self.usage_count = usage_count
        self.metadata = metadata if metadata is not None else {}
        self.last_update = 0  # Timestamp of last update


class LongTermMemory(CompressibleMemory):
    """Long-term memory storing strategic patterns."""

    def __init__(
        self,
        capacity: int = 8,
        embedding_dim: int = 128,
        min_cluster_size: int = 10,
        success_threshold: float = 0.6,
    ):
        """Initialize long-term memory.

        Args:
            capacity: Maximum number of templates to store
            embedding_dim: Dimension of stored embeddings
            min_cluster_size: Minimum size for creating new template
            success_threshold: Minimum success rate for template retention
        """
        super().__init__(capacity, embedding_dim)

        self.min_cluster_size = min_cluster_size
        self.success_threshold = success_threshold
        
        # Initialize templates
        self.templates: List[StrategyTemplate] = []
        self.update_counter = 0

        # Template update network
        self.update_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        ).to(self.device)

    def write(self, embeddings: Tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write new strategic patterns to memory.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            metadata: Optional metadata with success indicators
        """
        success_indicators = (
            metadata.get("success", torch.ones(len(embeddings)))
            if metadata is not None
            else torch.ones(len(embeddings))
        )

        # Group similar embeddings
        clusters = self._cluster_embeddings(embeddings)

        for cluster_indices in clusters:
            if len(cluster_indices) < self.min_cluster_size:
                continue

            # Calculate cluster statistics
            cluster_embeddings = embeddings[cluster_indices]
            cluster_success = success_indicators[cluster_indices].mean().item()
            
            # Create or update template
            cluster_center = cluster_embeddings.mean(dim=0)
            self._add_or_update_template(
                cluster_center,
                cluster_success,
                len(cluster_indices),
                metadata,
            )

    def read(self, query: Tensor) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """Read most relevant templates using attention.

        Args:
            query: Tensor of shape (batch_size, query_dim)

        Returns:
            Tuple of (retrieved_embeddings, metadata)
        """
        if not self.templates:
            return torch.zeros_like(query), None

        # Stack template embeddings
        template_embeddings = torch.stack(
            [t.embedding for t in self.templates]
        )

        # Calculate attention weights
        attention = torch.matmul(query, template_embeddings.T)
        attention = attention / np.sqrt(self.embedding_dim)
        
        # Apply success rate and usage count as attention bias
        success_bias = torch.tensor(
            [t.success_rate for t in self.templates],
            device=self.device
        ).unsqueeze(0)
        usage_bias = torch.tensor(
            [np.log1p(t.usage_count) for t in self.templates],
            device=self.device
        ).unsqueeze(0)
        
        attention = attention + 0.1 * success_bias + 0.05 * usage_bias
        weights = torch.softmax(attention, dim=-1)

        # Retrieve templates
        retrieved = torch.matmul(weights, template_embeddings)

        # Gather metadata
        metadata = {
            "attention_weights": weights.detach().cpu().numpy(),
            "template_success_rates": [t.success_rate for t in self.templates],
            "template_usage_counts": [t.usage_count for t in self.templates],
        }

        return retrieved, metadata

    def update(self, indices: Tensor, embeddings: Tensor) -> None:
        """Update specific templates.

        Args:
            indices: Tensor of indices to update
            embeddings: New embeddings for templates
        """
        for idx, embedding in zip(indices.tolist(), embeddings):
            if 0 <= idx < len(self.templates):
                # Update template using update network
                old_embedding = self.templates[idx].embedding
                combined = torch.cat([old_embedding, embedding])
                updated_embedding = self.update_network(combined)
                
                self.templates[idx].embedding = updated_embedding
                self.templates[idx].last_update = self.update_counter
                
        self.update_counter += 1

    def consolidate(self) -> None:
        """Consolidate templates by merging similar ones and removing unused ones."""
        if len(self.templates) < 2:
            return

        # Calculate similarity matrix
        embeddings = torch.stack([t.embedding for t in self.templates])
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )

        # Find templates to merge (similarity > 0.9)
        merged_indices = set()
        new_templates = []

        for i in range(len(self.templates)):
            if i in merged_indices:
                continue

            similar_indices = torch.where(similarities[i] > 0.9)[0].tolist()
            similar_indices = [j for j in similar_indices if j not in merged_indices]

            if similar_indices:
                # Merge similar templates
                similar_templates = [self.templates[j] for j in similar_indices]
                merged_template = self._merge_templates(similar_templates)
                new_templates.append(merged_template)
                merged_indices.update(similar_indices)

        # Add unmerged templates
        for i, template in enumerate(self.templates):
            if i not in merged_indices:
                new_templates.append(template)

        # Remove unsuccessful templates if we're over capacity
        if len(new_templates) > self.capacity:
            new_templates.sort(key=lambda t: t.success_rate * np.log1p(t.usage_count),
                             reverse=True)
            new_templates = new_templates[:self.capacity]

        self.templates = new_templates

    def clear(self) -> None:
        """Clear all templates."""
        self.templates = []
        self.update_counter = 0

    def _cluster_embeddings(self, embeddings: Tensor) -> List[List[int]]:
        """Cluster embeddings using cosine similarity.

        Args:
            embeddings: Input embeddings to cluster

        Returns:
            List of cluster index lists
        """
        n = len(embeddings)
        if n == 0:
            return []

        # Calculate pairwise similarities
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )

        # Simple clustering based on similarity threshold
        clusters = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            # Find similar embeddings
            similar_indices = torch.where(similarities[i] > 0.8)[0].tolist()
            similar_indices = [j for j in similar_indices if j not in used]

            if len(similar_indices) >= self.min_cluster_size:
                clusters.append(similar_indices)
                used.update(similar_indices)

        return clusters

    def _add_or_update_template(
        self,
        embedding: Tensor,
        success_rate: float,
        count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add new template or update existing similar template.

        Args:
            embedding: Template embedding
            success_rate: Success rate of the template
            count: Number of examples in cluster
            metadata: Optional metadata
        """
        # Find most similar existing template
        max_similarity = -1
        similar_template_idx = -1

        for i, template in enumerate(self.templates):
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0),
                template.embedding.unsqueeze(0)
            ).item()
            if similarity > max_similarity:
                max_similarity = similarity
                similar_template_idx = i

        if max_similarity > 0.9 and similar_template_idx >= 0:
            # Update existing template
            template = self.templates[similar_template_idx]
            total_count = template.usage_count + count
            template.success_rate = (
                template.success_rate * template.usage_count + success_rate * count
            ) / total_count
            template.usage_count = total_count
            template.last_update = self.update_counter
            
            # Update embedding using update network
            combined = torch.cat([template.embedding, embedding])
            template.embedding = self.update_network(combined)

        elif len(self.templates) < self.capacity:
            # Add new template
            self.templates.append(
                StrategyTemplate(embedding, success_rate, count, metadata)
            )

        elif success_rate > min(t.success_rate for t in self.templates):
            # Replace least successful template
            worst_idx = min(
                range(len(self.templates)),
                key=lambda i: self.templates[i].success_rate
            )
            self.templates[worst_idx] = StrategyTemplate(
                embedding, success_rate, count, metadata
            )

        self.update_counter += 1

    def _merge_templates(self, templates: List[StrategyTemplate]) -> StrategyTemplate:
        """Merge multiple templates into one.

        Args:
            templates: List of templates to merge

        Returns:
            Merged template
        """
        # Calculate weighted average of embeddings
        total_count = sum(t.usage_count for t in templates)
        weights = torch.tensor(
            [t.usage_count / total_count for t in templates],
            device=self.device
        )
        embeddings = torch.stack([t.embedding for t in templates])
        merged_embedding = torch.sum(weights.unsqueeze(1) * embeddings, dim=0)

        # Calculate combined statistics
        success_rate = sum(
            t.success_rate * t.usage_count for t in templates
        ) / total_count
        usage_count = total_count

        # Merge metadata
        merged_metadata = {}
        for t in templates:
            if t.metadata:
                for k, v in t.metadata.items():
                    if k not in merged_metadata:
                        merged_metadata[k] = []
                    merged_metadata[k].append(v)

        return StrategyTemplate(
            merged_embedding,
            success_rate,
            usage_count,
            merged_metadata
        )

    def __len__(self) -> int:
        """Return number of stored templates."""
        return len(self.templates)
