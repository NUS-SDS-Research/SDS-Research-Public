"""
VFL system configuration — plain dataclasses, no framework coupling.
Designed to be consumed directly by Dagster ops (Later) and LangGraph state (Later).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


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

    def num_classes(self) -> int:
        """Returns number of output classes for the selected dataset."""
        return 10 if self.dataset == "mnist" else 2

    def batch_size(self) -> int:
        """Returns batch size for the selected dataset."""
        if self.dataset == "mnist":
            return self.mnist.batch_size
        return self.cifer.batch_size
