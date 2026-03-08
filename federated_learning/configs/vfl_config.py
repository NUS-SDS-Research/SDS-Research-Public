"""
VFL system configuration — plain dataclasses, no framework coupling.
Designed to be consumed directly by Dagster ops (Later) and LangGraph state (Later).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DPConfig:
    """
    Differential Privacy configuration for VFL embedding-level noise injection.

    When enabled, every embedding transmitted across the VFL cut-layer is:
      1. Clipped to have L2 norm ≤ clip_norm  (bounds sensitivity)
      2. Perturbed with Gaussian noise N(0, (noise_multiplier * clip_norm)²)

    This defends against gradient inversion attacks (Zhu et al., NeurIPS 2019)
    that attempt to reconstruct passive party features from the embedding stream.

    Privacy accounting is handled by DPBudgetAccountant (Opacus RDPAccountant)
    and provides (ε, δ)-DP guarantees over the full training run.
    """
    enabled: bool = False
    clip_norm: float = 1.0         # C: maximum L2 norm per embedding row
    noise_multiplier: float = 1.0  # σ: noise_std = noise_multiplier * clip_norm
    delta: float = 1e-5            # δ: failure probability for (ε, δ)-DP


@dataclass
class CiferConfig:
    """Configuration for CiferAI fraud detection dataset."""

    dataset_id: str = "CiferAI/Cifer-Fraud-Detection-Dataset-AF"

    # Vertical split column assignment
    # Party A: identity/temporal features (who, when)
    party_a_columns: list[str] = field(default_factory=lambda: [
        "step", "nameOrig", "nameDest",
    ])
    # Party B: transaction/behavioral features (what, how much)
    party_b_columns: list[str] = field(default_factory=lambda: [
        "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "isFlaggedFraud",
    ])

    label_column: str = "isFraud"

    # Columns requiring encoding (before scaling)
    categorical_columns: list[str] = field(default_factory=lambda: [
        "type",       # LabelEncoder  (low cardinality: CASH_IN, CASH_OUT, ...)
        "nameOrig",   # Frequency encoding (millions of unique values)
        "nameDest",   # Frequency encoding (millions of unique values)
    ])

    # Subsample cap (None = use full 21M rows)
    max_samples: int = 100_000
    train_split: float = 0.8
    batch_size: int = 256
    random_state: int = 42

    # Minority class oversampling (random repetition of fraud rows).
    # With ~0.1% fraud rate and max_samples=100k, only ~94 positive samples
    # exist — far too few for the model to learn a fraud signal.
    # oversample_target_ratio=0.1 repeats fraud rows until they are 10% of
    # the training set (~8,800 fraud vs 80k non-fraud).
    oversample_minority: bool = True
    oversample_target_ratio: float = 0.1    # minority / (minority + majority)

    # CiferAI-specific learning rate (M2).
    # Lower than the MNIST default (1e-3) because the oversampled training
    # distribution is much more sensitive to large gradient steps; using 1e-3
    # causes limit-cycle oscillation between "predict all-fraud" and "predict
    # no-fraud" states across rounds.
    learning_rate: float = 1e-4

    # Fraud decision threshold for inference (C2).
    # The model is calibrated to the oversampled training distribution (10%
    # fraud), but the real val set has only 0.12% fraud.  Increasing this
    # threshold above 0.5 trades recall for precision — i.e. fewer false
    # positives at the cost of missing some true fraud cases.
    # Tune between 0.5 (high recall / low precision) and 0.9 (balanced F1).
    # NOTE: reverted to 0.5 after Attempt 4 showed threshold=0.7 suppresses
    # all detections when LR=1e-4 keeps fraud softmax probs in 0.5–0.7 range.
    fraud_threshold: float = 0.5


@dataclass
class MNISTConfig:
    """Configuration for MNIST vertical spatial split."""

    batch_size: int = 128
    data_dir: str = "./data"
    download: bool = True
    # Spatial split: Party A = top half, Party B = bottom half
    # MNIST images are (1, 28, 28); split at row 14
    split_row: int = 14


@dataclass
class VFLConfig:
    """Master VFL configuration."""

    # Model architecture
    embedding_dim: int = 128          # Per-party embedding size
    hidden_dim: int = 256             # Bottom model hidden layer width

    # Training
    num_rounds: int = 25              # Flower rounds to simulate
    local_epochs: int = 1             # Epochs per Flower round (client-side)
    learning_rate: float = 1e-3

    # Runtime
    device: str = "cpu"              # "cpu" | "cuda" | "mps"
    seed: int = 42

    # Dataset selection
    dataset: Literal["mnist", "cifer"] = "mnist"

    # Dataset-specific configs
    cifer: CiferConfig = field(default_factory=CiferConfig)
    mnist: MNISTConfig = field(default_factory=MNISTConfig)

    # Differential Privacy (Epic 3)
    dp: DPConfig = field(default_factory=DPConfig)

    def num_classes(self) -> int:
        """Returns number of output classes for the selected dataset."""
        return 10 if self.dataset == "mnist" else 2

    def batch_size(self) -> int:
        """Returns batch size for the selected dataset."""
        if self.dataset == "mnist":
            return self.mnist.batch_size
        return self.cifer.batch_size
