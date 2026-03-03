"""
Entry point: Vertical Federated Learning on CiferAI Fraud Detection.

Prerequisites
-------------
1. HuggingFace authentication (dataset may require login):
       huggingface-cli login
2. Dependencies installed:
       pip install -r requirements.txt

Quick start
-----------
    cd federated_learning
    python scripts/run_cifer_vfl.py

Expected output:
    [CiferAI] Loading dataset from HuggingFace...
    [CiferAI] Loaded 80000 train samples | fraud rate: 0.0013 | ...
    [Round  1] train loss=X.XXXX  train acc=X.XXXX | val loss=X.XXXX  val acc=X.XXXX
    ...

Notes
-----
- CiferAI is heavily class-imbalanced (~0.1% fraud rate).
  Accuracy alone is misleading; track loss convergence as a proxy.
- max_samples=100_000 (default) keeps first runs fast.
  Set config.cifer.max_samples = None for the full 6.3M row dataset.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from configs.vfl_config import VFLConfig
from src.datasets.cifer_dataset import CiferVerticalDataset
from src.datasets.dataloader_factory import VFLDataLoaderFactory
from src.models.bottom_models import TabularBottomModel
from src.models.top_model import VFLTopModel
from src.vfl.training_loop import VFLTrainer


def run_pure_pytorch(config: VFLConfig) -> None:
    """Run CiferAI VFL training in pure PyTorch mode."""
    print("=" * 60)
    print(" CiferAI Fraud Detection VFL — Pure PyTorch Mode")
    print("=" * 60)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    # ── Load and split dataset ──────────────────────────────────────────
    # build_aligned_pair returns TRAIN datasets only.
    # For validation, re-load with a held-out subset (simplified here:
    # we use the same train loaders and report train metrics only).
    ds_a, ds_b, ds_server = CiferVerticalDataset.build_aligned_pair(config.cifer)

    loader_a, loader_b, loader_server = VFLDataLoaderFactory.get_cifer_loaders(
        config=config.cifer,
        batch_size=config.cifer.batch_size,
        seed=config.seed,
    )

    # ── Infer feature dimensions from datasets ──────────────────────────
    dim_a = ds_a.feature_dim    # e.g. 3 (step, nameOrig_freq, nameDest_freq)
    dim_b = ds_b.feature_dim    # e.g. 7 (type, amount, 4 balances, isFlaggedFraud)

    print(f"Party A feature dim: {dim_a}  |  Party B feature dim: {dim_b}")

    # ── Models ──────────────────────────────────────────────────────────
    bottom_a = TabularBottomModel(
        input_dim=dim_a,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    bottom_b = TabularBottomModel(
        input_dim=dim_b,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    top_model = VFLTopModel(
        embedding_dim=config.embedding_dim,
        num_parties=2,
        num_classes=2,            # Binary: fraud (1) or not (0)
    )

    # ── Optimizers ──────────────────────────────────────────────────────
    opt_a   = torch.optim.Adam(bottom_a.parameters(), lr=config.learning_rate)
    opt_b   = torch.optim.Adam(bottom_b.parameters(), lr=config.learning_rate)
    opt_top = torch.optim.Adam(top_model.parameters(), lr=config.learning_rate)

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = VFLTrainer(
        bottom_model_a=bottom_a,
        bottom_model_b=bottom_b,
        top_model=top_model,
        optimizer_a=opt_a,
        optimizer_b=opt_b,
        optimizer_top=opt_top,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        verbose=True,
    )

    # ── Training loop ───────────────────────────────────────────────────
    val_metrics = {"loss": float("nan"), "accuracy": float("nan")}
    for rnd in range(1, config.num_rounds + 1):
        train_metrics = trainer.train_one_epoch(loader_a, loader_b, loader_server)
        # Reuse train loaders for validation (simple proxy; replace with
        # a separate val split for production use)
        val_metrics = trainer.evaluate(loader_a, loader_b, loader_server)

        print(
            f"[Round {rnd:>3}] "
            f"train loss={train_metrics['loss']:.4f}  "
            f"train acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}  "
            f"val acc={val_metrics['accuracy']:.4f}"
        )

    print("\nTraining complete.")
    print(
        f"Final val accuracy: {val_metrics['accuracy']:.4f}  "
        f"(note: imbalanced dataset — loss convergence is the key signal)"
    )


if __name__ == "__main__":
    cfg = VFLConfig(dataset="cifer")
    run_pure_pytorch(cfg)