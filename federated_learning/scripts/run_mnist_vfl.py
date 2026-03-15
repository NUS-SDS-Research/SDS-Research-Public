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

MLflow tracking 
------------------------
Each run is logged to the "VFL-MNIST" experiment (./mlruns by default).
View results:
    mlflow ui          # opens http://localhost:5000
Params logged: num_rounds, lr, embedding_dim, dp_*, batch_size, seed
Metrics logged per round: train_loss, train_accuracy, val_loss, val_accuracy, lr
Summary metrics: dp_epsilon (if DP enabled)
Model artefacts: top_model, bottom_model_a, bottom_model_b
"""
from __future__ import annotations

import sys
import os

# Allow running from the project root: python scripts/run_mnist_vfl.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.pytorch
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

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

    # ── open MLflow run ──────────────────────────────────────────
    # Use file-based backend (mlruns/) anchored to the project root.
    # This avoids the SQLite backend that triggers MLflow's auth middleware.
    # View runs with: mlflow ui --backend-store-uri ./mlruns
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("VFL-MNIST")
    run_name = (
        f"mnist-dp{'on' if config.dp.enabled else 'off'}"
        f"-r{config.num_rounds}-s{config.seed}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "num_rounds":          config.num_rounds,
            "learning_rate":       config.learning_rate,
            "embedding_dim":       config.embedding_dim,
            "hidden_dim":          config.hidden_dim,
            "batch_size":          config.mnist.batch_size,
            "seed":                config.seed,
            "split_row":           config.mnist.split_row,
            "dp_enabled":          config.dp.enabled,
            "dp_clip_norm":        config.dp.clip_norm,
            "dp_noise_multiplier": config.dp.noise_multiplier,
            "dp_delta":            config.dp.delta,
        })

        # ── Aligned DataLoaders ──────────────────────────────────────────
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

        # ── Models ──────────────────────────────────────────────────────
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

        # ── Optimizers ──────────────────────────────────────────────────
        opt_a   = torch.optim.Adam(bottom_a.parameters(), lr=config.learning_rate)
        opt_b   = torch.optim.Adam(bottom_b.parameters(), lr=config.learning_rate)
        opt_top = torch.optim.Adam(top_model.parameters(), lr=config.learning_rate)

        # T1: CosineAnnealingLR smoothly decays LR over all rounds, preventing the
        #     sharp round 1→2 accuracy drop caused by constant LR overshooting.
        sched_a   = CosineAnnealingLR(opt_a,   T_max=config.num_rounds, eta_min=1e-5)
        sched_b   = CosineAnnealingLR(opt_b,   T_max=config.num_rounds, eta_min=1e-5)
        sched_top = CosineAnnealingLR(opt_top, T_max=config.num_rounds, eta_min=1e-5)

        # ── Trainer ─────────────────────────────────────────────────────
        # pass dp_config so the trainer wires up DP clip+noise and
        #         the privacy budget accountant when config.dp.enabled=True.
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
            dp_config=config.dp,
        )

        if config.dp.enabled:
            print(
                f"[DP] Differential Privacy ENABLED — "
                f"clip_norm={config.dp.clip_norm}, "
                f"noise_multiplier={config.dp.noise_multiplier}, "
                f"δ={config.dp.delta}"
            )

        # ── Training loop ────────────────────────────────────────────────
        val_metrics: dict = {}
        for rnd in range(1, config.num_rounds + 1):
            train_metrics = trainer.train_one_epoch(loader_a, loader_b, loader_server)
            val_metrics   = trainer.evaluate(val_a, val_b, val_server)

            # Step LR schedulers at the end of each round (T1)
            sched_a.step()
            sched_b.step()
            sched_top.step()

            current_lr = sched_top.get_last_lr()[0]

            print(
                f"[Round {rnd:>3}] "
                f"train loss={train_metrics['loss']:.4f}  "
                f"train acc={train_metrics['accuracy']:.4f} | "
                f"val loss={val_metrics['loss']:.4f}  "
                f"val acc={val_metrics['accuracy']:.4f}  "
                f"lr={current_lr:.2e}"
            )

            # log per-round metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss":     train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss":       val_metrics["loss"],
                    "val_accuracy":   val_metrics["accuracy"],
                    "lr":             current_lr,
                },
                step=rnd,
            )

        print("\nTraining complete.")
        print(f"Final val accuracy: {val_metrics['accuracy']:.4f}")

        # report privacy budget if DP was enabled
        if trainer.dp_accountant is not None:
            eps = trainer.dp_accountant.get_epsilon()
            print(
                f"Privacy budget — ε={eps:.4f} at δ={config.dp.delta}  "
                f"(clip_norm={config.dp.clip_norm}, "
                f"noise_multiplier={config.dp.noise_multiplier}, "
                f"rounds={config.num_rounds})"
            )
            # log final privacy budget
            mlflow.log_metric("dp_epsilon", eps)

        # log trained model artefacts to MLflow
        mlflow.pytorch.log_model(trainer.top,      "top_model")
        mlflow.pytorch.log_model(trainer.bottom_a, "bottom_model_a")
        mlflow.pytorch.log_model(trainer.bottom_b, "bottom_model_b")

        print(f"[MLflow] Run complete — experiment: VFL-MNIST  name: {run_name}")


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
    cfg.dp.enabled = True
    cfg.dp.clip_norm = 1.0
    cfg.dp.noise_multiplier = 1.0

    # ── Choose mode ──────────────────────────────────────────────────────
    run_pure_pytorch(cfg)           # Mode A (default — fast, no Flower overhead)
    # run_flower_simulation(cfg)    # Mode B (uncomment to use Flower framework)
