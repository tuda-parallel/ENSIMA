"""
This module provides:
- A generic PointNet that extracts latent features from 3D point clouds.
- A PointNetAutoEncoder for unsupervised representation learning and clustering.
- A PointNetClassifier for supervised classification.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import torch
import torch.nn as nn


class PointNet(nn.Module):
    """PointNet-style encoder for extracting global embeddings from 3D point clouds.

    This module serves as a reusable backbone for both supervised and unsupervised tasks,
    taking a batch of 3D point coordinates and producing a fixed-size embedding vector
    through MLPs and global max pooling.

    Args:
        out_dim: Dimensionality of the output embedding vector.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute global max pooled embedding.

        Args:
            x: Input tensor of shape (B, N, 3) representing point clouds.

        Returns:
            Tensor of shape (B, out_dim) with pooled embeddings.
        """
        x = self.mlp(x)
        x_max = torch.max(x, dim=1)[0]  # global max pooling
        return x_max


class PointNetAutoEncoder(nn.Module):
    """PointNet-based autoencoder for unsupervised representation learning of point clouds.

    Uses a PointNet as encoder and a simple MLP decoder to reconstruct point clouds.
    Useful for learning embeddings for clustering or other unsupervised downstream tasks.

    Args:
        latent_dim: Dimensionality of the latent embedding space.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = PointNet(out_dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024 * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for autoencoding.

        Args:
            x: Input point cloud of shape (B, N, 3).

        Returns:
            Reconstructed point cloud of shape (B, 1024, 3).
        """
        z = self.encoder(x)
        recon = self.decoder(z).view(-1, 1024, 3)
        return recon


class PointNetClassifier(nn.Module):
    """Classifier network combining a PointNet backbone with a linear output head.

    Designed for supervised learning where each point cloud is associated with a label.

    Args:
        num_classes: Number of output classes.
        embedding_dim: Dimensionality of the intermediate embedding.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = PointNet(out_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification.

        Args:
            x: Input point cloud tensor of shape (B, N, 3).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding vector before the classification head.

        Args:
            x: Input point cloud tensor.

        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        return self.encoder(x)
