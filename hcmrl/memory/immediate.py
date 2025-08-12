from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .base import MemoryBase


class ImmediateMemory(MemoryBase):
    """Immediate memory buffer for storing recent experiences."""

    def __init__(self, capacity: int = 32, embedding_dim: int = 128):
        """Initialize immediate memory buffer.

        Args:
            capacity: Maximum number of items to store (default: 32)
            embedding_dim: Dimension of stored embeddings (default: 128)
        """
        super().__init__(capacity, embedding_dim)
        
        # Initialize storage
        self.embeddings = torch.zeros((capacity, embedding_dim), device=self.device)
        self.metadata_list: List[Optional[Dict[str, Any]]] = [None] * capacity
        self.importance = torch.zeros(capacity, device=self.device)
        
        # Tracking variables
        self.current_size = 0
        self.write_idx = 0
        self.timestamps = torch.zeros(capacity, device=self.device)

    def write(self, embeddings: Tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write embeddings to memory with FIFO and importance-based replacement.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            metadata: Optional metadata associated with embeddings
        """
        batch_size = embeddings.shape[0]
        
        for i in range(batch_size):
            # Calculate importance of new embedding
            importance = self._calculate_importance(embeddings[i])
            
            if self.current_size < self.capacity:
                # If space available, write to next position
                idx = self.current_size
                self.current_size += 1
            else:
                # Replace least important item if new item is more important
                least_important_idx = torch.argmin(self.importance)
                if importance > self.importance[least_important_idx]:
                    idx = least_important_idx
                else:
                    continue
            
            # Write embedding and metadata
            self.embeddings[idx] = embeddings[i]
            self.metadata_list[idx] = metadata[i] if metadata is not None else None
            self.importance[idx] = importance
            self.timestamps[idx] = self.write_idx
            self.write_idx += 1

    def read(self, query: Tensor) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """Read from memory using attention over stored embeddings.

        Args:
            query: Tensor of shape (batch_size, query_dim)

        Returns:
            Tuple of (retrieved_embeddings, metadata)
        """
        if self.current_size == 0:
            # Return zero embedding if memory is empty
            return (
                torch.zeros_like(query),
                None
            )

        # Calculate attention weights
        attention = torch.matmul(query, self.embeddings[:self.current_size].T)
        attention = attention / np.sqrt(self.embedding_dim)
        weights = torch.softmax(attention, dim=-1)

        # Retrieve embeddings using attention
        retrieved = torch.matmul(weights, self.embeddings[:self.current_size])

        # Gather metadata for retrieved items
        metadata = {
            "attention_weights": weights.detach().cpu().numpy(),
            "timestamps": self.timestamps[:self.current_size].detach().cpu().numpy(),
            "importance": self.importance[:self.current_size].detach().cpu().numpy(),
        }

        return retrieved, metadata

    def update(self, indices: Tensor, embeddings: Tensor) -> None:
        """Update specific memory locations.

        Args:
            indices: Tensor of indices to update
            embeddings: New embeddings to store at the indices
        """
        self.embeddings[indices] = embeddings
        self.importance[indices] = torch.tensor(
            [self._calculate_importance(emb) for emb in embeddings],
            device=self.device
        )
        self.timestamps[indices] = self.write_idx
        self.write_idx += 1

    def clear(self) -> None:
        """Clear all memory contents."""
        self.embeddings.zero_()
        self.metadata_list = [None] * self.capacity
        self.importance.zero_()
        self.timestamps.zero_()
        self.current_size = 0
        self.write_idx = 0

    def get_oldest(self) -> Tuple[Tensor, List[Optional[Dict[str, Any]]]]:
        """Get the oldest items in memory.

        Returns:
            Tuple of (embeddings, metadata)
        """
        if self.current_size == 0:
            return torch.empty(0, self.embedding_dim), []

        # Sort by timestamp
        sorted_indices = torch.argsort(self.timestamps[:self.current_size])
        oldest_embeddings = self.embeddings[sorted_indices]
        oldest_metadata = [self.metadata_list[i] for i in sorted_indices.tolist()]

        return oldest_embeddings, oldest_metadata

    def remove_oldest(self, count: int = 1) -> None:
        """Remove the oldest items from memory.

        Args:
            count: Number of items to remove
        """
        if self.current_size == 0:
            return

        # Sort by timestamp
        sorted_indices = torch.argsort(self.timestamps[:self.current_size])
        remove_indices = sorted_indices[:count]

        # Shift remaining items
        keep_indices = sorted_indices[count:]
        self.embeddings[:len(keep_indices)] = self.embeddings[keep_indices]
        self.importance[:len(keep_indices)] = self.importance[keep_indices]
        self.timestamps[:len(keep_indices)] = self.timestamps[keep_indices]
        self.metadata_list[:len(keep_indices)] = [self.metadata_list[i] for i in keep_indices.tolist()]

        # Update size
        self.current_size = max(0, self.current_size - count)

    def _calculate_importance(self, embedding: Tensor) -> float:
        """Calculate importance score for an embedding.

        Args:
            embedding: Input embedding tensor

        Returns:
            Importance score
        """
        # Simple importance metric based on L2 norm and uniqueness
        l2_norm = torch.norm(embedding)
        
        if self.current_size == 0:
            return l2_norm.item()

        # Calculate similarity to existing embeddings
        similarities = torch.matmul(
            embedding,
            self.embeddings[:self.current_size].T
        )
        max_similarity = torch.max(similarities)

        # Combine L2 norm and uniqueness
        importance = l2_norm * (1 - max_similarity)
        return importance.item()

    def __len__(self) -> int:
        """Return current number of items in memory."""
        return self.current_size
