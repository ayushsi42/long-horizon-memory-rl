from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MemoryCompressor:
    """Utility class for memory compression operations."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        compression_method: str = "vae",
        device: Optional[torch.device] = None,
    ):
        """Initialize memory compressor.

        Args:
            input_dim: Dimension of input embeddings
            target_dim: Target dimension after compression
            compression_method: Compression method ('vae' or 'pca')
            device: Torch device to use
        """
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.compression_method = compression_method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if compression_method == "vae":
            self.compressor = VariationalCompressor(
                input_dim=input_dim,
                compressed_dim=target_dim,
                hidden_dim=min(input_dim * 2, 512),
            ).to(self.device)
        elif compression_method == "pca":
            self.compressor = PCACompressor(
                input_dim=input_dim,
                target_dim=target_dim,
            )
        else:
            raise ValueError(f"Unknown compression method: {compression_method}")

    def compress(
        self,
        embeddings: Tensor,
        train: bool = False,
        num_epochs: int = 100,
    ) -> Tensor:
        """Compress embeddings to lower dimension.

        Args:
            embeddings: Input embeddings to compress
            train: Whether to train the compressor
            num_epochs: Number of training epochs if training

        Returns:
            Compressed embeddings
        """
        if self.compression_method == "vae":
            if train:
                self._train_vae(embeddings, num_epochs)
            with torch.no_grad():
                compressed, _, _ = self.compressor(embeddings)
            return compressed
        else:  # PCA
            if train:
                self.compressor.fit(embeddings)
            return self.compressor.transform(embeddings)

    def decompress(self, compressed: Tensor) -> Tensor:
        """Decompress embeddings back to original dimension.

        Args:
            compressed: Compressed embeddings

        Returns:
            Decompressed embeddings
        """
        if self.compression_method == "vae":
            with torch.no_grad():
                return self.compressor.decode(compressed)
        else:  # PCA
            return self.compressor.inverse_transform(compressed)

    def _train_vae(self, embeddings: Tensor, num_epochs: int) -> None:
        """Train the VAE compressor.

        Args:
            embeddings: Training data
            num_epochs: Number of training epochs
        """
        optimizer = torch.optim.Adam(self.compressor.parameters(), lr=1e-3)
        batch_size = min(128, len(embeddings))

        for epoch in range(num_epochs):
            # Shuffle data
            perm = torch.randperm(len(embeddings))
            total_loss = 0.0

            # Train in batches
            for i in range(0, len(embeddings), batch_size):
                batch_idx = perm[i:i + batch_size]
                batch = embeddings[batch_idx]

                optimizer.zero_grad()
                recon, mu, log_var = self.compressor(batch)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon, batch)

                # KL divergence
                kl_loss = -0.5 * torch.sum(
                    1 + log_var - mu.pow(2) - log_var.exp()
                ) / batch.size(0)

                # Total loss with beta-VAE weighting
                loss = recon_loss + 0.1 * kl_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (len(embeddings) / batch_size)
                print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")


class PCACompressor:
    """PCA-based memory compression."""

    def __init__(self, input_dim: int, target_dim: int):
        """Initialize PCA compressor.

        Args:
            input_dim: Input dimension
            target_dim: Target dimension after compression
        """
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.components: Optional[Tensor] = None
        self.mean: Optional[Tensor] = None

    def fit(self, embeddings: Tensor) -> None:
        """Fit PCA to data.

        Args:
            embeddings: Training data
        """
        # Center data
        self.mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - self.mean

        # Compute covariance matrix
        cov = torch.mm(centered.T, centered) / (len(embeddings) - 1)

        # Compute eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.components = eigenvectors[:, sorted_indices[:self.target_dim]]

    def transform(self, embeddings: Tensor) -> Tensor:
        """Transform data to lower dimension.

        Args:
            embeddings: Input embeddings

        Returns:
            Compressed embeddings
        """
        if self.components is None or self.mean is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        centered = embeddings - self.mean
        return torch.mm(centered, self.components)

    def inverse_transform(self, compressed: Tensor) -> Tensor:
        """Transform compressed data back to original dimension.

        Args:
            compressed: Compressed embeddings

        Returns:
            Reconstructed embeddings
        """
        if self.components is None or self.mean is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        reconstructed = torch.mm(compressed, self.components.T)
        return reconstructed + self.mean


class VariationalCompressor(nn.Module):
    """Variational autoencoder for memory compression."""

    def __init__(
        self,
        input_dim: int,
        compressed_dim: int,
        hidden_dim: int = 256,
        activation: nn.Module = nn.ReLU,
    ):
        """Initialize VAE compressor.

        Args:
            input_dim: Input dimension
            compressed_dim: Compressed dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function to use
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, compressed_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, compressed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim // 2),
            activation(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights.

        Args:
            module: Neural network module
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to latent space.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mu, log_var)
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors.

        Args:
            z: Latent vectors

        Returns:
            Reconstructed tensor
        """
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick.

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
        """Forward pass through VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstructed, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def compress_sequence(
    sequence: Union[List[Tensor], Tensor],
    window_size: int,
    stride: int,
    compressor: MemoryCompressor,
) -> Tensor:
    """Compress a sequence of embeddings using sliding window.

    Args:
        sequence: List of embeddings or tensor of shape (seq_len, dim)
        window_size: Size of sliding window
        stride: Stride for sliding window
        compressor: Compressor instance to use

    Returns:
        Compressed sequence
    """
    if isinstance(sequence, list):
        sequence = torch.stack(sequence)

    # Pad sequence if needed
    if len(sequence) < window_size:
        pad_size = window_size - len(sequence)
        sequence = F.pad(sequence, (0, 0, 0, pad_size))

    # Create sliding windows
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[i:i + window_size]
        windows.append(window.flatten())

    # Compress windows
    windows_tensor = torch.stack(windows)
    compressed = compressor.compress(windows_tensor)

    return compressed


def decompress_sequence(
    compressed: Tensor,
    window_size: int,
    stride: int,
    original_length: int,
    compressor: MemoryCompressor,
) -> Tensor:
    """Decompress a sequence of compressed embeddings.

    Args:
        compressed: Compressed sequence
        window_size: Original window size
        stride: Original stride
        original_length: Original sequence length
        compressor: Compressor instance to use

    Returns:
        Decompressed sequence
    """
    # Decompress windows
    decompressed_windows = compressor.decompress(compressed)

    # Reshape windows
    window_dim = window_size * compressed.shape[-1] // compressor.target_dim
    decompressed_windows = decompressed_windows.view(-1, window_size, window_dim)

    # Reconstruct sequence using overlap-add
    result = torch.zeros((original_length, window_dim))
    counts = torch.zeros(original_length)

    for i, window in enumerate(decompressed_windows):
        pos = i * stride
        result[pos:pos + window_size] += window
        counts[pos:pos + window_size] += 1

    # Average overlapping regions
    result = result / counts.unsqueeze(1).clamp(min=1)

    return result[:original_length]
