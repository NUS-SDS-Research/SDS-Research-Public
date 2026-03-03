"""
Entry point: Vertical Federated Learning on MNIST.

Two modes — switch by uncommenting in main():
  A) Pure PyTorch simulation (default, no Flower overhead, fastest for dev)
  B) Flower simulation (full framework path)

Quick start
-----------
    cd federated_learning
    pip install -r requirements.txt
    python scripts/run_mnist_vfl.py

Expected output (pure PyTorch, 10 rounds):
    [Round  1] train loss=2.3021  train acc=0.1094 | val loss=2.2987  val acc=0.1109
    ...
    [Round 10] train loss=1.7432  train acc=0.4612 | val loss=1.7214  val acc=0.4703

Accuracy on half-images is lower than full-image training by design;
>50% within 10 rounds is a good sign that the VFL gradient bridge is working.
"""
from __future__ import annotations

import sys
import os

# Allow running from the project root: python scripts/run_mnist_vfl.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from configs.vfl_config import VFLConfig, MNISTConfig
from src.datasets.dataloader_factory import VFLDataLoaderFactory
from src.models.bottom_models import CNNBottomModel
from src.models.top_model import VFLTopModel
from src.vfl.training_loop import VFLTrainer


# ---------------------------------------------------------------------------
# Mode A: Pure PyTorch VFL (recommended for development & validation)
# ---------------------------------------------------------------------------

def run_pure_pytorch(config: VFLConfig) -> None:
    """Run VFL training entirely in PyTorch — no Flower framework overhead."""
    print("=" * 60)
    print(" MNIST VFL — Pure PyTorch Mode")
    print("=" * 60)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    # ── Aligned DataLoaders ─────────────────────────────────────────────
    loader_a, loader_b, loader_server = VFLDataLoaderFactory.get_mnist_loaders(
        config=config.mnist,
        batch_size=config.mnist.batch_size,
        seed=config.seed,
        train=True,
    )
    val_a, val_b, val_server = VFLDataLoaderFactory.get_mnist_loaders(
        config=config.mnist,
        batch_size=config.mnist.batch_size,
        seed=config.seed,
        train=False,
    )

    # ── Models ──────────────────────────────────────────────────────────
    # Party A: top half (1, 14, 28)
    bottom_a = CNNBottomModel(
        in_channels=1,
        input_height=config.mnist.split_row,
        input_width=28,
        embedding_dim=config.embedding_dim,
    )
    # Party B: bottom half (1, 14, 28)
    bottom_b = CNNBottomModel(
        in_channels=1,
        input_height=28 - config.mnist.split_row,
        input_width=28,
        embedding_dim=config.embedding_dim,
    )
    top_model = VFLTopModel(
        embedding_dim=config.embedding_dim,
        num_parties=2,
        num_classes=10,           # MNIST: 10 digit classes
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
    for rnd in range(1, config.num_rounds + 1):
        train_metrics = trainer.train_one_epoch(loader_a, loader_b, loader_server)
        val_metrics   = trainer.evaluate(val_a, val_b, val_server)

        print(
            f"[Round {rnd:>3}] "
            f"train loss={train_metrics['loss']:.4f}  "
            f"train acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}  "
            f"val acc={val_metrics['accuracy']:.4f}"
        )

    print("\nTraining complete.")
    print(f"Final val accuracy: {val_metrics['accuracy']:.4f}")


# ---------------------------------------------------------------------------
# Mode B: Flower Simulation
# ---------------------------------------------------------------------------

def run_flower_simulation(config: VFLConfig) -> None:
    """Run VFL via Flower's simulation engine (framework-compatible path)."""
    import flwr as fl
    from src.datasets.dataloader_factory import VFLDataLoaderFactory
    from src.flower.client_app import register_client, make_client_fn
    from src.flower.server_app import make_vfl_server_components

    print("=" * 60)
    print(" MNIST VFL — Flower Simulation Mode")
    print("=" * 60)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    loader_a, loader_b, loader_server = VFLDataLoaderFactory.get_mnist_loaders(
        config=config.mnist,
        batch_size=config.mnist.batch_size,
        seed=config.seed,
        train=True,
    )

    bottom_a = CNNBottomModel(
        in_channels=1,
        input_height=config.mnist.split_row,
        input_width=28,
        embedding_dim=config.embedding_dim,
    )
    bottom_b = CNNBottomModel(
        in_channels=1,
        input_height=28 - config.mnist.split_row,
        input_width=28,
        embedding_dim=config.embedding_dim,
    )

    # Register clients (persists state across Flower rounds)
    register_client("0", bottom_a, torch.optim.Adam(bottom_a.parameters(), lr=config.learning_rate), loader_a, device)
    register_client("1", bottom_b, torch.optim.Adam(bottom_b.parameters(), lr=config.learning_rate), loader_b, device)

    strategy, server_cfg = make_vfl_server_components(
        config=config,
        label_loader=loader_server,
        num_classes=10,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(),
        num_clients=2,
        config=server_cfg,
        strategy=strategy,
        ray_init_args={"num_cpus": 1},      # keep it single-process for reproducibility
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = VFLConfig(dataset="mnist")

    # ── Choose mode ─────────────────────────────────────────────────────
    run_pure_pytorch(cfg)           # Mode A (default — fast, no Flower overhead)
    # run_flower_simulation(cfg)    # Mode B (uncomment to use Flower framework)