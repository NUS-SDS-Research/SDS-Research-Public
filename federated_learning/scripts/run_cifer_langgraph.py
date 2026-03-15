"""
Entry point: CiferAI Fraud Detection VFL via LangGraph.

Replaces the manual ``for rnd in range(...)`` loop in ``run_cifer_vfl.py``
with a LangGraph StateGraph, adding explicit round-loop control and a
first-class M4 checkpoint edge stored directly in VFLState.

Key differences from run_cifer_vfl.py
--------------------------------------
* Training loop is a compiled StateGraph; round count drives a conditional edge.
* VFLState carries ``best_model_state`` — a deepcopy checkpoint saved at the
  best-recall round (M4) — eliminating the need for a local ``best_state`` var.
* Fraud metrics (F1/recall/precision) computed inside ``cifer_evaluate_node``,
  not interleaved with the loop body.
* No LR scheduler (mirrors run_cifer_vfl.py — constant LR=1e-3).

Quick start
-----------
    cd federated_learning
    python scripts/run_cifer_langgraph.py

MLflow tracking
---------------
Experiment: VFL-CiferAI
Run name: cifer-lg-dp{on|off}-r{num_rounds}-s{seed}
Identical metrics/params/artefacts to run_cifer_vfl.py runs.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.pytorch
import numpy as np
import torch

from configs.vfl_config import VFLConfig
from src.datasets.dataloader_factory import VFLDataLoaderFactory
from src.models.bottom_models import TabularBottomModel
from src.models.top_model import VFLTopModel
from src.vfl.training_loop import VFLTrainer
from src.langgraph.vfl_graph import build_vfl_graph
from src.langgraph.nodes import make_cifer_evaluate_node, _fraud_metrics


def run_cifer_langgraph(config: VFLConfig) -> None:
    """Run CiferAI VFL fraud detection training via LangGraph StateGraph."""
    print("=" * 60)
    print(" CiferAI Fraud Detection VFL — LangGraph Orchestration Mode")
    print("=" * 60)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    # ── MLflow setup ─────────────────────────────────────────────────────
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    mlflow.set_tracking_uri(os.path.join(_project_root, "mlruns"))
    mlflow.set_experiment("VFL-CiferAI")
    run_name = (
        f"cifer-lg-dp{'on' if config.dp.enabled else 'off'}"
        f"-r{config.num_rounds}-s{config.seed}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "num_rounds":              config.num_rounds,
            "learning_rate":           config.cifer.learning_rate,
            "embedding_dim":           config.embedding_dim,
            "hidden_dim":              config.hidden_dim,
            "batch_size":              config.cifer.batch_size,
            "seed":                    config.seed,
            "max_samples":             config.cifer.max_samples,
            "oversample_minority":     config.cifer.oversample_minority,
            "oversample_target_ratio": config.cifer.oversample_target_ratio,
            "fraud_threshold":         config.cifer.fraud_threshold,
            "dp_enabled":              config.dp.enabled,
            "dp_clip_norm":            config.dp.clip_norm,
            "dp_noise_multiplier":     config.dp.noise_multiplier,
            "dp_delta":                config.dp.delta,
            "orchestration":           "langgraph",
        })

        # ── DataLoaders ───────────────────────────────────────────────────
        (loader_a, loader_b, loader_server), (val_a, val_b, val_server) = (
            VFLDataLoaderFactory.get_cifer_loaders(
                config=config.cifer,
                batch_size=config.cifer.batch_size,
                seed=config.seed,
            )
        )

        dim_a = loader_a.dataset.feature_dim
        dim_b = loader_b.dataset.feature_dim
        print(f"Party A feature dim: {dim_a}  |  Party B feature dim: {dim_b}")

        # ── Class weights ─────────────────────────────────────────────────
        labels_all: np.ndarray = loader_server.dataset._labels
        n_pos = int(labels_all.sum())
        n_neg = int(len(labels_all) - n_pos)
        class_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
        print(
            f"Class weights — non-fraud: {class_weight[0]:.2f} | "
            f"fraud: {class_weight[1]:.1f}  (n_neg={n_neg}, n_pos={n_pos})"
        )
        mlflow.log_params({"n_pos_train": n_pos, "n_neg_train": n_neg})

        # ── Models ────────────────────────────────────────────────────────
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
            num_classes=2,
        )

        # ── Optimizers (constant LR — no scheduler for CiferAI) ──────────
        cifer_lr = config.cifer.learning_rate
        opt_a   = torch.optim.Adam(bottom_a.parameters(), lr=cifer_lr)
        opt_b   = torch.optim.Adam(bottom_b.parameters(), lr=cifer_lr)
        opt_top = torch.optim.Adam(top_model.parameters(), lr=cifer_lr)

        # ── Trainer ───────────────────────────────────────────────────────
        trainer = VFLTrainer(
            bottom_model_a=bottom_a,
            bottom_model_b=bottom_b,
            top_model=top_model,
            optimizer_a=opt_a,
            optimizer_b=opt_b,
            optimizer_top=opt_top,
            criterion=torch.nn.CrossEntropyLoss(weight=class_weight),
            device=device,
            verbose=False,
            dp_config=config.dp,
        )

        if config.dp.enabled:
            print(
                f"[DP] Differential Privacy ENABLED — "
                f"clip_norm={config.dp.clip_norm}, "
                f"noise_multiplier={config.dp.noise_multiplier}, "
                f"δ={config.dp.delta}"
            )

        # ── Build graph with CiferAI evaluate node ────────────────────────
        cifer_eval = make_cifer_evaluate_node(
            trainer=trainer,
            val_a=val_a,
            val_b=val_b,
            val_server=val_server,
            fraud_threshold=config.cifer.fraud_threshold,
        )

        graph = build_vfl_graph(
            trainer=trainer,
            train_loaders=(loader_a, loader_b, loader_server),
            val_loaders=(val_a, val_b, val_server),
            schedulers=None,               # M3: no scheduler for CiferAI
            evaluate_node_fn=cifer_eval,
        )

        # ── Initial state ─────────────────────────────────────────────────
        initial_state = {
            "model_state":      trainer.get_model_state(),
            "round":            0,
            "num_rounds":       config.num_rounds,
            "metrics_history":  [],
            "best_val_acc":     0.0,   # unused by cifer graph
            "best_recall":      0.0,
            "best_round":       0,
            "best_model_state": {},    # filled by M4 checkpoint on first recall > 0
            "dp_epsilon":       None,
        }

        # ── Run graph ─────────────────────────────────────────────────────
        print(f"\nStarting LangGraph run: {run_name}  ({config.num_rounds} rounds)\n")
        final_state = graph.invoke(initial_state)

        # ── Post-run: restore M4 checkpoint and re-evaluate ───────────────
        best_round = final_state["best_round"]
        best_recall = final_state["best_recall"]

        if final_state["best_model_state"]:
            trainer.load_model_state(final_state["best_model_state"])
            best_report = _fraud_metrics(
                trainer, val_a, val_b, val_server,
                fraud_threshold=config.cifer.fraud_threshold,
            )
        else:
            # No positive predictions ever made — all metrics zero
            best_report = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        print("\nTraining complete.")
        print(
            f"Best model  (round {best_round:>2}) — "
            f"recall={best_report['recall']:.4f}  "
            f"F1={best_report['f1']:.4f}  "
            f"precision={best_report['precision']:.4f}"
        )
        print("(Accuracy dominated by majority class; F1/recall are the key signals.)")

        # ── Log per-round metrics to MLflow from history ──────────────────
        train_history = {e["round"]: e for e in final_state["metrics_history"] if e["phase"] == "train"}
        val_history   = {e["round"]: e for e in final_state["metrics_history"] if e["phase"] == "val"}

        for rnd in range(1, config.num_rounds + 1):
            metrics_step: dict[str, float] = {}
            if rnd in train_history:
                t = train_history[rnd]
                metrics_step["train_loss"]     = t["loss"]
                metrics_step["train_accuracy"] = t["accuracy"]
            if rnd in val_history:
                v = val_history[rnd]
                metrics_step["val_loss"]       = v["loss"]
                metrics_step["val_accuracy"]   = v["accuracy"]
                metrics_step["val_f1"]         = v["f1"]
                metrics_step["val_recall"]     = v["recall"]
                metrics_step["val_precision"]  = v["precision"]
            if metrics_step:
                mlflow.log_metrics(metrics_step, step=rnd)

        # ── DP epsilon ────────────────────────────────────────────────────
        if final_state["dp_epsilon"] is not None:
            eps = final_state["dp_epsilon"]
            print(
                f"Privacy budget — ε={eps:.4f} at δ={config.dp.delta}  "
                f"(clip_norm={config.dp.clip_norm}, "
                f"noise_multiplier={config.dp.noise_multiplier}, "
                f"rounds={config.num_rounds})"
            )
            mlflow.log_metric("dp_epsilon", eps)

        # ── Summary metrics ───────────────────────────────────────────────
        # Retrieve best_f1 from val_history for the summary
        best_f1 = max((e["f1"] for e in val_history.values()), default=0.0)
        best_f1_round = max(
            (e["round"] for e in val_history.values() if e["f1"] == best_f1),
            default=0,
        )

        mlflow.log_metrics({
            "best_recall":          best_recall,
            "best_recall_round":    float(best_round),
            "best_f1":              best_f1,
            "best_f1_round":        float(best_f1_round),
            "best_model_recall":    best_report["recall"],
            "best_model_f1":        best_report["f1"],
            "best_model_precision": best_report["precision"],
        })

        # ── Log M4 checkpoint artefacts (best-recall weights) ─────────────
        mlflow.pytorch.log_model(trainer.top,      "top_model")
        mlflow.pytorch.log_model(trainer.bottom_a, "bottom_model_a")
        mlflow.pytorch.log_model(trainer.bottom_b, "bottom_model_b")

        print(f"[MLflow] Run complete — experiment: VFL-CiferAI  name: {run_name}")


if __name__ == "__main__":
    cfg = VFLConfig(dataset="cifer")
    # DP off by default — toggle below to compare:
    cfg.dp.enabled = True
    cfg.dp.clip_norm = 1.0
    cfg.dp.noise_multiplier = 1.0

    run_cifer_langgraph(cfg)
