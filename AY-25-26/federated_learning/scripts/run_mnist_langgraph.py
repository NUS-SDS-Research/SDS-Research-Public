"""
Entry point: MNIST VFL training orchestrated via LangGraph.

This script replaces the manual ``for rnd in range(...)`` loop in
``run_mnist_vfl.py`` with a LangGraph StateGraph.  Training, evaluation,
and round-stopping logic become explicit graph nodes and edges — making
the control flow inspectable, checkpointable, and extensible.

Key differences from run_mnist_vfl.py
--------------------------------------
* The training loop is a compiled StateGraph (not a for-loop).
* VFLState carries the authoritative model checkpoint between nodes.
* Conditional early stopping is a first-class graph edge (not an if-statement).
* MLflow logging spans the full graph.invoke() call.

Quick start
-----------
    cd federated_learning
    python scripts/run_mnist_langgraph.py

MLflow tracking
---------------
Same experiment / run naming as run_mnist_vfl.py ("VFL-MNIST").
Run name: mnist-lg-dp{on|off}-r{num_rounds}-s{seed}
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.pytorch
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.vfl_config import VFLConfig
from src.datasets.dataloader_factory import VFLDataLoaderFactory
from src.models.bottom_models import CNNBottomModel
from src.models.top_model import VFLTopModel
from src.vfl.training_loop import VFLTrainer
from src.langgraph.vfl_graph import build_vfl_graph


def run_langgraph(config: VFLConfig) -> None:
    """Run MNIST VFL training via the LangGraph StateGraph."""
    print("=" * 60)
    print(" MNIST VFL — LangGraph Orchestration Mode")
    print("=" * 60)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    # ── MLflow setup ─────────────────────────────────────────────────────
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("VFL-MNIST")
    run_name = (
        f"mnist-lg-dp{'on' if config.dp.enabled else 'off'}"
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
            "orchestration":       "langgraph",
        })

        # ── DataLoaders ───────────────────────────────────────────────────
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

        # ── Models ────────────────────────────────────────────────────────
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
        top_model = VFLTopModel(
            embedding_dim=config.embedding_dim,
            num_parties=2,
            num_classes=10,
        )

        # ── Optimizers + schedulers ───────────────────────────────────────
        opt_a   = torch.optim.Adam(bottom_a.parameters(), lr=config.learning_rate)
        opt_b   = torch.optim.Adam(bottom_b.parameters(), lr=config.learning_rate)
        opt_top = torch.optim.Adam(top_model.parameters(), lr=config.learning_rate)

        sched_a   = CosineAnnealingLR(opt_a,   T_max=config.num_rounds, eta_min=1e-5)
        sched_b   = CosineAnnealingLR(opt_b,   T_max=config.num_rounds, eta_min=1e-5)
        sched_top = CosineAnnealingLR(opt_top, T_max=config.num_rounds, eta_min=1e-5)

        # ── Trainer ───────────────────────────────────────────────────────
        trainer = VFLTrainer(
            bottom_model_a=bottom_a,
            bottom_model_b=bottom_b,
            top_model=top_model,
            optimizer_a=opt_a,
            optimizer_b=opt_b,
            optimizer_top=opt_top,
            criterion=torch.nn.CrossEntropyLoss(),
            device=device,
            verbose=False,   # tqdm suppressed — LangGraph nodes print their own summary
            dp_config=config.dp,
        )

        if config.dp.enabled:
            print(
                f"[DP] Differential Privacy ENABLED — "
                f"clip_norm={config.dp.clip_norm}, "
                f"noise_multiplier={config.dp.noise_multiplier}, "
                f"δ={config.dp.delta}"
            )

        # ── Build graph ───────────────────────────────────────────────────
        graph = build_vfl_graph(
            trainer=trainer,
            train_loaders=(loader_a, loader_b, loader_server),
            val_loaders=(val_a, val_b, val_server),
            schedulers=[sched_a, sched_b, sched_top],
        )

        # ── Initial state ─────────────────────────────────────────────────
        initial_state = {
            "model_state":      trainer.get_model_state(),
            "round":            0,
            "num_rounds":       config.num_rounds,
            "metrics_history":  [],
            "best_val_acc":     0.0,
            "best_recall":      0.0,   # unused by MNIST graph
            "best_round":       0,
            "best_model_state": {},    # unused by MNIST graph
            "dp_epsilon":       None,
        }

        # ── Run graph ─────────────────────────────────────────────────────
        print(f"\nStarting LangGraph run: {run_name}  ({config.num_rounds} rounds)\n")
        final_state = graph.invoke(initial_state)

        # ── Post-run reporting + MLflow logging ───────────────────────────
        print("\nTraining complete.")
        print(
            f"Best val accuracy: {final_state['best_val_acc']:.4f} "
            f"(round {final_state['best_round']})"
        )

        # Log all per-round metrics from history to MLflow
        for entry in final_state["metrics_history"]:
            rnd = entry["round"]
            phase = entry["phase"]
            prefix = "train" if phase == "train" else "val"
            mlflow.log_metrics(
                {
                    f"{prefix}_loss":     entry["loss"],
                    f"{prefix}_accuracy": entry["accuracy"],
                },
                step=rnd,
            )

        # Log DP epsilon if enabled
        if final_state["dp_epsilon"] is not None:
            eps = final_state["dp_epsilon"]
            print(
                f"Privacy budget — ε={eps:.4f} at δ={config.dp.delta}  "
                f"(clip_norm={config.dp.clip_norm}, "
                f"noise_multiplier={config.dp.noise_multiplier}, "
                f"rounds={config.num_rounds})"
            )
            mlflow.log_metric("dp_epsilon", eps)

        # Restore best-round weights before logging artefacts
        # (final_state["model_state"] holds the last-round weights;
        #  for artefact logging we reload into trainer and save as-is)
        trainer.load_model_state(final_state["model_state"])
        mlflow.pytorch.log_model(trainer.top,      "top_model")
        mlflow.pytorch.log_model(trainer.bottom_a, "bottom_model_a")
        mlflow.pytorch.log_model(trainer.bottom_b, "bottom_model_b")

        mlflow.log_metric("best_val_acc", final_state["best_val_acc"])
        mlflow.log_metric("best_round",   float(final_state["best_round"]))

        print(f"[MLflow] Run complete — experiment: VFL-MNIST  name: {run_name}")


if __name__ == "__main__":
    cfg = VFLConfig(dataset="mnist")
    # DP off by default — toggle below to compare:
    cfg.dp.enabled = True
    cfg.dp.clip_norm = 1.0
    cfg.dp.noise_multiplier = 1.0

    run_langgraph(cfg)
