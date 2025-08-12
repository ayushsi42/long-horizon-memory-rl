from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


class MemoryBase(ABC):
    """Base class for all memory components."""

    def __init__(self, capacity: int, embedding_dim: int):
        """Initialize the memory component.

        Args:
            capacity: Maximum number of items that can be stored
            embedding_dim: Dimension of the stored embeddings
        """
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def write(self, embeddings: Tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write embeddings to memory.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            metadata: Optional metadata associated with the embeddings
        """
        pass

    @abstractmethod
    def read(self, query: Tensor) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """Read from memory using attention over stored embeddings.

        Args:
            query: Tensor of shape (batch_size, query_dim)

        Returns:
            Tuple of (retrieved_embeddings, metadata)
        """
        pass

    @abstractmethod
    def update(self, indices: Tensor, embeddings: Tensor) -> None:
        """Update specific memory locations.

        Args:
            indices: Tensor of indices to update
            embeddings: New embeddings to store at the indices
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory contents."""
        pass


class CompressibleMemory(MemoryBase):
    """Base class for memory components that support compression."""

    def __init__(
        self,
        capacity: int,
        embedding_dim: int,
        compression_ratio: float = 0.5,
        compression_threshold: int = 1000,
    ):
        """Initialize the compressible memory component.

        Args:
            capacity: Maximum number of items that can be stored
            embedding_dim: Dimension of the stored embeddings
            compression_ratio: Ratio to compress embeddings (e.g., 0.5 = half size)
            compression_threshold: Number of items before triggering compression
        """
        super().__init__(capacity, embedding_dim)
        self.compression_ratio = compression_ratio
        self.compression_threshold = compression_threshold
        self.compressed = False

    @abstractmethod
    def compress(self) -> None:
        """Compress the stored embeddings."""
        pass

    @abstractmethod
    def decompress(self) -> None:
        """Decompress the stored embeddings."""
        pass


class HierarchicalMemory:
    """Hierarchical memory system combining immediate, short-term, and long-term memory."""

    def __init__(
        self,
        immediate_size: int = 32,
        short_term_size: int = 16,
        long_term_size: int = 8,
        embedding_dim: int = 128,
    ):
        """Initialize the hierarchical memory system.

        Args:
            immediate_size: Size of immediate memory buffer
            short_term_size: Size of short-term memory
            long_term_size: Size of long-term memory
            embedding_dim: Base embedding dimension
        """
        from .immediate import ImmediateMemory
        from .short_term import ShortTermMemory
        from .long_term import LongTermMemory

        self.immediate = ImmediateMemory(immediate_size, embedding_dim)
        self.short_term = ShortTermMemory(short_term_size, embedding_dim)
        self.long_term = LongTermMemory(long_term_size, embedding_dim)

        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def write(
        self, embeddings: Tensor, memory_type: str = "immediate", metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Write embeddings to specified memory type.

        Args:
            embeddings: Tensor of embeddings to store
            memory_type: One of ["immediate", "short_term", "long_term"]
            metadata: Optional metadata to store with embeddings
        """
        if memory_type == "immediate":
            self.immediate.write(embeddings, metadata)
        elif memory_type == "short_term":
            self.short_term.write(embeddings, metadata)
        elif memory_type == "long_term":
            self.long_term.write(embeddings, metadata)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    def read(self, query: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Read from all memory levels using hierarchical attention.

        Args:
            query: Tensor of shape (batch_size, query_dim)

        Returns:
            Tuple of (retrieved_embeddings, metadata)
        """
        # Read from each memory level
        imm_emb, imm_meta = self.immediate.read(query)
        st_emb, st_meta = self.short_term.read(query)
        lt_emb, lt_meta = self.long_term.read(query)

        # Combine embeddings using attention
        all_embeddings = torch.cat([imm_emb, st_emb, lt_emb], dim=1)
        attention_weights = torch.softmax(
            torch.matmul(query, all_embeddings.transpose(1, 2)) / np.sqrt(self.embedding_dim),
            dim=-1,
        )
        combined_embedding = torch.matmul(attention_weights, all_embeddings)

        # Combine metadata
        combined_meta = {
            "immediate": imm_meta,
            "short_term": st_meta,
            "long_term": lt_meta,
            "attention_weights": attention_weights.detach().cpu().numpy(),
        }

        return combined_embedding, combined_meta

    def consolidate(self) -> None:
        """Trigger memory consolidation across all levels."""
        # Transfer from immediate to short-term if needed
        if len(self.immediate) >= self.immediate.capacity * 0.8:
            embeddings, metadata = self.immediate.get_oldest()
            self.short_term.write(embeddings, metadata)
            self.immediate.remove_oldest()

        # Compress and transfer from short-term to long-term if needed
        if len(self.short_term) >= self.short_term.compression_threshold:
            self.short_term.compress()
            embeddings, metadata = self.short_term.get_compressed()
            self.long_term.write(embeddings, metadata)

        # Maintain long-term memory
        self.long_term.consolidate()

    def clear(self) -> None:
        """Clear all memory levels."""
        self.immediate.clear()
        self.short_term.clear()
        self.long_term.clear()
