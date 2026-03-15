"""
Passive Party Bottom Models.

Two bottom model variants:
  - TabularBottomModel : MLP for CiferAI tabular features  → 128-dim embedding
  - CNNBottomModel     : CNN for MNIST half-images         → 128-dim embedding

Both output a raw 128-dim linear embedding (no final activation).
The top model learns the appropriate non-linear combination.

Designed to be the Opacus DP attachment point (Epic 3):
  engine = PrivacyEngine()
  bottom_model, optimizer, loader = engine.make_private(bottom_model, optimizer, loader)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TabularBottomModel(nn.Module):
    """
    MLP feature extractor for tabular (CiferAI) data.

    Architecture
    ------------
    Linear(input_dim → hidden_dim) → BatchNorm1d → ReLU → Dropout
    Linear(hidden_dim → hidden_dim) → BatchNorm1d → ReLU → Dropout
    Linear(hidden_dim → embedding_dim)

    Parameters
    ----------
    input_dim : int
        Width of the input feature vector for this party.
        Derived from ``CiferVerticalDataset.feature_dim``.
    embedding_dim : int
        Output embedding size (default 128).
    hidden_dim : int
        Width of the two hidden layers (default 256).
    dropout_rate : float
        Dropout probability after each hidden layer (default 0.3).

    Notes
    -----
    BatchNorm1d requires batch size > 1. Ensure ``drop_last=True`` in the
    DataLoader (already set by VFLDataLoaderFactory) to avoid single-sample
    batches at epoch boundaries.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)

        Returns
        -------
        Tensor, shape (batch, 128)
        """
        return self.network(x)


class CNNBottomModel(nn.Module):
    """
    CNN feature extractor for half-image (MNIST vertical split) data.

    Architecture
    ------------
    Conv2d(in_channels, 32, 3, pad=1) → ReLU → MaxPool2d(2)
    Conv2d(32, 64, 3, pad=1)          → ReLU → MaxPool2d(2)
    Flatten
    Linear(flat_size → 256) → ReLU
    Linear(256 → embedding_dim)

    The flattened size after the two pooling layers is computed
    automatically via a dummy forward pass, making this robust to
    any input resolution (e.g. future datasets or different split rows).

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 for MNIST).
    input_height : int
        Height of the input image slice (14 for a 28-px-tall MNIST half).
    input_width : int
        Width of the input image slice (28 for MNIST).
    embedding_dim : int
        Output embedding size (default 128).
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_height: int = 14,
        input_width: int = 28,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flat_size = self._compute_flat_size(in_channels, input_height, input_width)

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def _compute_flat_size(self, in_channels: int, h: int, w: int) -> int:
        """Pass a dummy tensor through conv_layers to find the flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            out = self.conv_layers(dummy)
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 1, 14, 28)  for standard MNIST half-split

        Returns
        -------
        Tensor, shape (batch, 128)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)