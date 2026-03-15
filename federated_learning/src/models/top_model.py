"""
Active Party Top Model (Aggregation Head).

Receives concatenated embeddings from all passive parties and produces
the final classification logits.

For 2-party VFL with embedding_dim=128:
  Input: concat([emb_a, emb_b]) → shape (batch, 256)
  Output: logits → shape (batch, num_classes)

Works for both:
  - MNIST digit classification  : num_classes=10
  - CiferAI fraud detection     : num_classes=2
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VFLTopModel(nn.Module):
    """
    Aggregation head (server-side top model) for Vertical Federated Learning.

    Architecture
    ------------
    Linear(input_dim → hidden_dim) → ReLU → Dropout
    Linear(hidden_dim → hidden_dim // 2) → ReLU
    Linear(hidden_dim // 2 → num_classes)

    Parameters
    ----------
    embedding_dim : int
        Per-party embedding size. Default 128.
    num_parties : int
        Number of passive parties. Default 2.
        input_dim is derived as ``num_parties * embedding_dim``.
    num_classes : int
        Number of output classes. 10 for MNIST, 2 for CiferAI.
    hidden_dim : int
        Width of the first hidden layer. Default 128.
    dropout_rate : float
        Dropout after the first hidden layer. Default 0.3.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_parties: int = 2,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_parties = num_parties
        self.num_classes = num_classes

        input_dim = num_parties * embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        Primary forward entry point — takes separate party embeddings.

        Parameters
        ----------
        emb_a : Tensor, shape (batch, embedding_dim)
        emb_b : Tensor, shape (batch, embedding_dim)

        Returns
        -------
        logits : Tensor, shape (batch, num_classes)
        """
        combined = torch.cat([emb_a, emb_b], dim=1)  # (batch, 2 * embedding_dim)
        return self.classifier(combined)

    def forward_from_concat(self, concat_emb: torch.Tensor) -> torch.Tensor:
        """
        Alternate entry point — takes a pre-concatenated embedding tensor.

        Used in the Flower pathway where embeddings arrive pre-stacked after
        numpy serialisation.

        Parameters
        ----------
        concat_emb : Tensor, shape (batch, num_parties * embedding_dim)

        Returns
        -------
        logits : Tensor, shape (batch, num_classes)
        """
        return self.classifier(concat_emb)