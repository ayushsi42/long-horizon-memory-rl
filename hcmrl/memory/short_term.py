from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import CompressibleMemory


class VariationalCompressor(nn.Module):
    """Variational autoencoder for compressing memory embeddings."""

    def __init__(self, input_dim: int, compressed_dim: int, hidden_dim: int = 256):
        """Initialize the VAE compressor.

        Args:
            input_dim: Dimension of input embeddings
            compressed_dim: Dimension of compressed embeddings
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, compressed_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, compressed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input embeddings to latent space.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mu, log_var)
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors to reconstructed embeddings.

        Args:
            z: Latent vectors

        Returns:
            Reconstructed embeddings
        """
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick for VAE.

        Args:
            mu: Mean vectors
            log_var: Log variance vectors

        Returns:
            Sampled latent vectors
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstructed, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class ShortTermMemory(CompressibleMemory):
    """Short-term memory with compression capabilities."""

    def __init__(
        self,
        capacity: int = 16,
        embedding_dim: int = 128,
        compressed_dim: int = 64,
        compression_ratio: float = 0.5,
        compression_threshold: int = 1000,
    ):
        """Initialize short-term memory.

        Args:
            capacity: Maximum number of items to store
            embedding_dim: Dimension of stored embeddings
            compressed_dim: Dimension of compressed embeddings
            compression_ratio: Ratio to compress embeddings
            compression_threshold: Number of items before triggering compression
        """
        super().__init__(capacity, embedding_dim, compression_ratio, compression_threshold)

        # Initialize storage
        self.embeddings = torch.zeros((capacity, embedding_dim), device=self.device)
        self.compressed_embeddings = torch.zeros((capacity, compressed_dim), device=self.device)
        self.metadata_list: List[Optional[Dict[str, Any]]] = [None] * capacity
        self.current_size = 0

        # Initialize compressor
        self.compressor = VariationalCompressor(
            embedding_dim, compressed_dim
        ).to(self.device)
        self.compressed = False

    def write(self, embeddings: Tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write embeddings to memory.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            metadata: Optional metadata associated with embeddings
        """
        batch_size = embeddings.shape[0]
        available_space = self.capacity - self.current_size

        if batch_size > available_space:
            # If not enough space, only write what fits
            embeddings = embeddings[:available_space]
            if metadata is not None:
                metadata = {k: v[:available_space] for k, v in metadata.items()}

        # Write embeddings
        end_idx = self.current_size + len(embeddings)
        self.embeddings[self.current_size:end_idx] = embeddings

        # Write metadata
        if metadata is not None:
            self.metadata_list[self.current_size:end_idx] = [
                {k: v[i] for k, v in metadata.items()}
                for i in range(len(embeddings))
            ]

        self.current_size = end_idx

        # Check if compression is needed
        if self.current_size >= self.compression_threshold and not self.compressed:
            self.compress()

    def read(self, query: Tensor) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """Read from memory using attention.

        Args:
            query: Tensor of shape (batch_size, query_dim)

        Returns:
            Tuple of (retrieved_embeddings, metadata)
        """
        if self.current_size == 0:
            return torch.zeros_like(query), None

        # Use appropriate embeddings based on compression state
        memory_embeddings = (
            self.compressed_embeddings[:self.current_size]
            if self.compressed
            else self.embeddings[:self.current_size]
        )

        # Calculate attention weights
        attention = torch.matmul(query, memory_embeddings.T)
        attention = attention / np.sqrt(
            self.compressed_embeddings.shape[1] if self.compressed else self.embedding_dim
        )
        weights = torch.softmax(attention, dim=-1)

        # Retrieve embeddings
        retrieved = torch.matmul(weights, memory_embeddings)

        # If compressed, decompress retrieved embeddings
        if self.compressed:
            retrieved = self.compressor.decode(retrieved)

        # Gather metadata
        metadata = {
            "attention_weights": weights.detach().cpu().numpy(),
            "is_compressed": self.compressed,
        }

        return retrieved, metadata

    def update(self, indices: Tensor, embeddings: Tensor) -> None:
        """Update specific memory locations.

        Args:
            indices: Tensor of indices to update
            embeddings: New embeddings to store at the indices
        """
        if self.compressed:
            # Compress new embeddings before storing
            with torch.no_grad():
                compressed, _, _ = self.compressor(embeddings)
                self.compressed_embeddings[indices] = compressed
        else:
            self.embeddings[indices] = embeddings

    def compress(self) -> None:
        """Compress the stored embeddings using VAE."""
        if self.compressed or self.current_size == 0:
            return

        # Train compressor on current embeddings
        optimizer = torch.optim.Adam(self.compressor.parameters(), lr=1e-3)
        for _ in range(100):  # Training iterations
            optimizer.zero_grad()
            recon, mu, log_var = self.compressor(self.embeddings[:self.current_size])
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(recon, self.embeddings[:self.current_size])
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            optimizer.step()

        # Compress embeddings
        with torch.no_grad():
            self.compressed_embeddings[:self.current_size], _, _ = self.compressor(
                self.embeddings[:self.current_size]
            )

        self.compressed = True

    def decompress(self) -> None:
        """Decompress the stored embeddings."""
        if not self.compressed:
            return

        # Decompress embeddings
        with torch.no_grad():
            self.embeddings[:self.current_size] = self.compressor.decode(
                self.compressed_embeddings[:self.current_size]
            )

        self.compressed = False

    def get_compressed(self) -> Tuple[Tensor, List[Optional[Dict[str, Any]]]]:
        """Get compressed embeddings and metadata.

        Returns:
            Tuple of (compressed_embeddings, metadata)
        """
        if not self.compressed:
            self.compress()

        return (
            self.compressed_embeddings[:self.current_size].clone(),
            self.metadata_list[:self.current_size],
        )

    def clear(self) -> None:
        """Clear all memory contents."""
        self.embeddings.zero_()
        self.compressed_embeddings.zero_()
        self.metadata_list = [None] * self.capacity
        self.current_size = 0
        self.compressed = False

    def __len__(self) -> int:
        """Return current number of items in memory."""
        return self.current_size
