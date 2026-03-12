"""
VFL Dagster Assets.

Each @asset materialises one VFL training artefact.  Asset lineage:

    mnist_vfl_trained ──► mnist_vfl_eval
    cifer_vfl_trained ──► cifer_vfl_eval

Design rationale
----------------
* DataLoaders are NOT serialisable — they are recreated from config inside each
  asset.  Only serialisable results (model state dicts, metrics dicts) flow
  between assets via Dagster's IO manager.
* Model state dicts (dicts of torch.Tensor) are returned as asset values.
  Dagster's default pickle IO manager handles them correctly.
* Training uses the LangGraph StateGraph (build_vfl_graph) so the full round
  loop, early-stopping logic, and M4 checkpoint work identically to the
  standalone scripts.
* Dagster Output metadata exposes key metrics in the Dagster UI at
  materialisation time — best_val_acc, best_recall, dp_epsilon, etc.
* No MLflow calls inside assets: the standalone scripts (run_mnist_vfl.py,
  run_mnist_langgraph.py, etc.) are the designated MLflow entry points.
  Dagster's native lineage/metadata replace MLflow for the pipeline layer.

Usage
-----
    cd federated_learning
    dagster dev -f scripts/run_dagster.py
"""
import sys
import os

# Ensure federated_learning/ is importable regardless of invocation directory
_project_root = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import copy

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from dagster import asset, Config, Output, MetadataValue

from configs.vfl_config import VFLConfig, DPConfig
from src.datasets.dataloader_factory import VFLDataLoaderFactory
from src.models.bottom_models import CNNBottomModel, TabularBottomModel
from src.models.top_model import VFLTopModel
from src.vfl.training_loop import VFLTrainer
from src.langgraph.vfl_graph import build_vfl_graph
from src.langgraph.nodes import make_cifer_evaluate_node, _fraud_metrics


# ---------------------------------------------------------------------------
# Dagster Config classes (Pydantic-backed; fields must be plain types)
# ---------------------------------------------------------------------------

class MNISTTrainConfig(Config):
    """Hyperparameters for the MNIST VFL training asset."""
    num_rounds: int = 25
    learning_rate: float = 1e-3
    embedding_dim: int = 128
    hidden_dim: int = 256
    seed: int = 42
    batch_size: int = 128
    split_row: int = 14
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    dp_delta: float = 1e-5


class CiferTrainConfig(Config):
    """Hyperparameters for the CiferAI VFL training asset."""
    num_rounds: int = 25
    embedding_dim: int = 128
    hidden_dim: int = 256
    seed: int = 42
    batch_size: int = 256
    max_samples: int = 100_000
    fraud_threshold: float = 0.5
    learning_rate: float = 1e-3
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    dp_delta: float = 1e-5


# ---------------------------------------------------------------------------
# Helper: build DPConfig from Dagster config fields
# ---------------------------------------------------------------------------

def _dp_config(cfg) -> DPConfig:
    return DPConfig(
        enabled=cfg.dp_enabled,
        clip_norm=cfg.dp_clip_norm,
        noise_multiplier=cfg.dp_noise_multiplier,
        delta=cfg.dp_delta,
    )


# ---------------------------------------------------------------------------
# MNIST assets
# ---------------------------------------------------------------------------

@asset(
    group_name="mnist",
    description=(
        "Run MNIST VFL training via LangGraph. "
        "Materialises model state dicts + per-round metrics history."
    ),
)
def mnist_vfl_trained(config: MNISTTrainConfig) -> Output[dict]:
    """
    Full MNIST VFL training pipeline.

    Returns
    -------
    dict with keys:
        model_state     : dict of state_dicts (bottom_a, bottom_b, top)
        metrics_history : list of per-round metric dicts
        best_val_acc    : float
        best_round      : int
        dp_epsilon      : float | None
    """
    device = torch.device("cpu")
    torch.manual_seed(config.seed)

    vfl_cfg = VFLConfig(dataset="mnist")
    vfl_cfg.num_rounds = config.num_rounds
    vfl_cfg.learning_rate = config.learning_rate
    vfl_cfg.embedding_dim = config.embedding_dim
    vfl_cfg.hidden_dim = config.hidden_dim
    vfl_cfg.seed = config.seed
    vfl_cfg.mnist.batch_size = config.batch_size
    vfl_cfg.mnist.split_row = config.split_row
    dp = _dp_config(config)
    vfl_cfg.dp = dp

    loader_a, loader_b, loader_server = VFLDataLoaderFactory.get_mnist_loaders(
        config=vfl_cfg.mnist,
        batch_size=config.batch_size,
        seed=config.seed,
        train=True,
    )
    val_a, val_b, val_server = VFLDataLoaderFactory.get_mnist_loaders(
        config=vfl_cfg.mnist,
        batch_size=config.batch_size,
        seed=config.seed,
        train=False,
    )

    bottom_a = CNNBottomModel(
        in_channels=1,
        input_height=config.split_row,
        input_width=28,
        embedding_dim=config.embedding_dim,
    )
    bottom_b = CNNBottomModel(
        in_channels=1,
        input_height=28 - config.split_row,
        input_width=28,
        embedding_dim=config.embedding_dim,
    )
    top_model = VFLTopModel(
        embedding_dim=config.embedding_dim,
        num_parties=2,
        num_classes=10,
    )

    opt_a   = torch.optim.Adam(bottom_a.parameters(), lr=config.learning_rate)
    opt_b   = torch.optim.Adam(bottom_b.parameters(), lr=config.learning_rate)
    opt_top = torch.optim.Adam(top_model.parameters(), lr=config.learning_rate)

    sched_a   = CosineAnnealingLR(opt_a,   T_max=config.num_rounds, eta_min=1e-5)
    sched_b   = CosineAnnealingLR(opt_b,   T_max=config.num_rounds, eta_min=1e-5)
    sched_top = CosineAnnealingLR(opt_top, T_max=config.num_rounds, eta_min=1e-5)

    trainer = VFLTrainer(
        bottom_model_a=bottom_a,
        bottom_model_b=bottom_b,
        top_model=top_model,
        optimizer_a=opt_a,
        optimizer_b=opt_b,
        optimizer_top=opt_top,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        verbose=False,
        dp_config=dp,
    )

    graph = build_vfl_graph(
        trainer=trainer,
        train_loaders=(loader_a, loader_b, loader_server),
        val_loaders=(val_a, val_b, val_server),
        schedulers=[sched_a, sched_b, sched_top],
    )

    initial_state = {
        "model_state":      trainer.get_model_state(),
        "round":            0,
        "num_rounds":       config.num_rounds,
        "metrics_history":  [],
        "best_val_acc":     0.0,
        "best_recall":      0.0,
        "best_round":       0,
        "best_model_state": {},
        "dp_epsilon":       None,
    }

    print(f"\n[Dagster] mnist_vfl_trained — {config.num_rounds} rounds  "
          f"DP={'on' if config.dp_enabled else 'off'}\n")
    final_state = graph.invoke(initial_state)

    result = {
        "model_state":     final_state["model_state"],
        "metrics_history": final_state["metrics_history"],
        "best_val_acc":    final_state["best_val_acc"],
        "best_round":      final_state["best_round"],
        "dp_epsilon":      final_state["dp_epsilon"],
    }

    meta: dict = {
        "best_val_acc": MetadataValue.float(final_state["best_val_acc"]),
        "best_round":   MetadataValue.int(final_state["best_round"]),
        "num_rounds":   MetadataValue.int(config.num_rounds),
        "dp_enabled":   MetadataValue.bool(config.dp_enabled),
    }
    if final_state["dp_epsilon"] is not None:
        meta["dp_epsilon"] = MetadataValue.float(final_state["dp_epsilon"])

    return Output(result, metadata=meta)


@asset(
    group_name="mnist",
    description=(
        "Evaluate the trained MNIST VFL model on the held-out validation set. "
        "Materialises val_acc, val_loss (and dp_epsilon if DP was used)."
    ),
)
def mnist_vfl_eval(mnist_vfl_trained: dict, config: MNISTTrainConfig) -> Output[dict]:
    """
    Evaluate the trained MNIST model.

    Loads model weights from the upstream ``mnist_vfl_trained`` asset,
    recreates the validation DataLoaders, and runs a single evaluation pass.

    Returns
    -------
    dict with keys: val_acc, val_loss, best_val_acc, best_round, dp_epsilon
    """
    device = torch.device("cpu")
    torch.manual_seed(config.seed)

    vfl_cfg = VFLConfig(dataset="mnist")
    vfl_cfg.mnist.split_row = config.split_row

    _, _, _ = VFLDataLoaderFactory.get_mnist_loaders(   # train loaders (discarded)
        config=vfl_cfg.mnist, batch_size=config.batch_size, seed=config.seed, train=True,
    )
    val_a, val_b, val_server = VFLDataLoaderFactory.get_mnist_loaders(
        config=vfl_cfg.mnist, batch_size=config.batch_size, seed=config.seed, train=False,
    )

    bottom_a = CNNBottomModel(
        in_channels=1, input_height=config.split_row, input_width=28,
        embedding_dim=config.embedding_dim,
    )
    bottom_b = CNNBottomModel(
        in_channels=1, input_height=28 - config.split_row, input_width=28,
        embedding_dim=config.embedding_dim,
    )
    top_model = VFLTopModel(
        embedding_dim=config.embedding_dim, num_parties=2, num_classes=10,
    )

    trainer = VFLTrainer(
        bottom_model_a=bottom_a,
        bottom_model_b=bottom_b,
        top_model=top_model,
        optimizer_a=torch.optim.Adam(bottom_a.parameters()),
        optimizer_b=torch.optim.Adam(bottom_b.parameters()),
        optimizer_top=torch.optim.Adam(top_model.parameters()),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        verbose=False,
    )
    trainer.load_model_state(mnist_vfl_trained["model_state"])

    metrics = trainer.evaluate(val_a, val_b, val_server)

    result = {
        "val_acc":      metrics["accuracy"],
        "val_loss":     metrics["loss"],
        "best_val_acc": mnist_vfl_trained["best_val_acc"],
        "best_round":   mnist_vfl_trained["best_round"],
        "dp_epsilon":   mnist_vfl_trained["dp_epsilon"],
    }

    meta: dict = {
        "val_acc":      MetadataValue.float(metrics["accuracy"]),
        "val_loss":     MetadataValue.float(metrics["loss"]),
        "best_val_acc": MetadataValue.float(mnist_vfl_trained["best_val_acc"]),
        "best_round":   MetadataValue.int(mnist_vfl_trained["best_round"]),
    }
    if mnist_vfl_trained["dp_epsilon"] is not None:
        meta["dp_epsilon"] = MetadataValue.float(mnist_vfl_trained["dp_epsilon"])

    print(
        f"\n[Dagster] mnist_vfl_eval — "
        f"val_acc={metrics['accuracy']:.4f}  val_loss={metrics['loss']:.4f}\n"
    )
    return Output(result, metadata=meta)


# ---------------------------------------------------------------------------
# CiferAI assets
# ---------------------------------------------------------------------------

@asset(
    group_name="cifer",
    description=(
        "Run CiferAI VFL fraud detection training via LangGraph. "
        "Materialises M4 best-recall model state + per-round fraud metrics."
    ),
)
def cifer_vfl_trained(config: CiferTrainConfig) -> Output[dict]:
    """
    Full CiferAI VFL training pipeline.

    Uses ``make_cifer_evaluate_node`` for F1/recall/precision at every eval step
    and stores the M4 best-recall checkpoint in the asset output.

    Returns
    -------
    dict with keys:
        model_state     : M4 best-recall state dict (or final-round if no recall)
        metrics_history : list of per-round metric dicts (includes F1/recall/prec)
        best_recall     : float
        best_round      : int
        best_f1         : float
        best_f1_round   : int
        dp_epsilon      : float | None
        n_pos_train     : int   (fraud samples after oversampling)
        n_neg_train     : int
    """
    device = torch.device("cpu")
    torch.manual_seed(config.seed)

    vfl_cfg = VFLConfig(dataset="cifer")
    vfl_cfg.num_rounds      = config.num_rounds
    vfl_cfg.embedding_dim   = config.embedding_dim
    vfl_cfg.hidden_dim      = config.hidden_dim
    vfl_cfg.seed            = config.seed
    vfl_cfg.cifer.batch_size           = config.batch_size
    vfl_cfg.cifer.max_samples          = config.max_samples
    vfl_cfg.cifer.fraud_threshold      = config.fraud_threshold
    vfl_cfg.cifer.learning_rate        = config.learning_rate
    vfl_cfg.dp = _dp_config(config)

    (loader_a, loader_b, loader_server), (val_a, val_b, val_server) = (
        VFLDataLoaderFactory.get_cifer_loaders(
            config=vfl_cfg.cifer,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    )

    dim_a = loader_a.dataset.feature_dim
    dim_b = loader_b.dataset.feature_dim

    labels_all: np.ndarray = loader_server.dataset._labels
    n_pos = int(labels_all.sum())
    n_neg = int(len(labels_all) - n_pos)
    class_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
    print(
        f"[Dagster] cifer_vfl_trained — "
        f"Party A dim={dim_a}  Party B dim={dim_b}  "
        f"n_pos={n_pos}  n_neg={n_neg}  "
        f"DP={'on' if config.dp_enabled else 'off'}"
    )

    bottom_a = TabularBottomModel(
        input_dim=dim_a, embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
    )
    bottom_b = TabularBottomModel(
        input_dim=dim_b, embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
    )
    top_model = VFLTopModel(
        embedding_dim=config.embedding_dim, num_parties=2, num_classes=2,
    )

    opt_a   = torch.optim.Adam(bottom_a.parameters(), lr=config.learning_rate)
    opt_b   = torch.optim.Adam(bottom_b.parameters(), lr=config.learning_rate)
    opt_top = torch.optim.Adam(top_model.parameters(), lr=config.learning_rate)

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
        dp_config=vfl_cfg.dp,
    )

    cifer_eval = make_cifer_evaluate_node(
        trainer=trainer,
        val_a=val_a,
        val_b=val_b,
        val_server=val_server,
        fraud_threshold=config.fraud_threshold,
    )

    graph = build_vfl_graph(
        trainer=trainer,
        train_loaders=(loader_a, loader_b, loader_server),
        val_loaders=(val_a, val_b, val_server),
        schedulers=None,            # M3: constant LR for CiferAI
        evaluate_node_fn=cifer_eval,
    )

    initial_state = {
        "model_state":      trainer.get_model_state(),
        "round":            0,
        "num_rounds":       config.num_rounds,
        "metrics_history":  [],
        "best_val_acc":     0.0,
        "best_recall":      0.0,
        "best_round":       0,
        "best_model_state": {},
        "dp_epsilon":       None,
    }

    print(f"\n[Dagster] cifer_vfl_trained — starting {config.num_rounds} rounds\n")
    final_state = graph.invoke(initial_state)

    # M4 checkpoint: use best-recall weights; fall back to final if none found
    best_model_state = final_state["best_model_state"] or final_state["model_state"]

    # Compute best_f1 from history
    val_history = [e for e in final_state["metrics_history"] if e["phase"] == "val"]
    best_f1 = max((e.get("f1", 0.0) for e in val_history), default=0.0)
    best_f1_round = max(
        (e["round"] for e in val_history if e.get("f1", 0.0) == best_f1),
        default=0,
    )

    result = {
        "model_state":     best_model_state,
        "metrics_history": final_state["metrics_history"],
        "best_recall":     final_state["best_recall"],
        "best_round":      final_state["best_round"],
        "best_f1":         best_f1,
        "best_f1_round":   best_f1_round,
        "dp_epsilon":      final_state["dp_epsilon"],
        "n_pos_train":     n_pos,
        "n_neg_train":     n_neg,
        "fraud_threshold": config.fraud_threshold,
    }

    meta: dict = {
        "best_recall":   MetadataValue.float(final_state["best_recall"]),
        "best_round":    MetadataValue.int(final_state["best_round"]),
        "best_f1":       MetadataValue.float(best_f1),
        "best_f1_round": MetadataValue.int(best_f1_round),
        "num_rounds":    MetadataValue.int(config.num_rounds),
        "n_pos_train":   MetadataValue.int(n_pos),
        "dp_enabled":    MetadataValue.bool(config.dp_enabled),
    }
    if final_state["dp_epsilon"] is not None:
        meta["dp_epsilon"] = MetadataValue.float(final_state["dp_epsilon"])

    return Output(result, metadata=meta)


@asset(
    group_name="cifer",
    description=(
        "Evaluate the trained CiferAI VFL model (M4 best-recall checkpoint) "
        "with fraud-specific metrics: F1, recall, precision."
    ),
)
def cifer_vfl_eval(cifer_vfl_trained: dict, config: CiferTrainConfig) -> Output[dict]:
    """
    Evaluate the M4 best-recall CiferAI model checkpoint.

    Loads the best-recall model state from ``cifer_vfl_trained``, recreates
    validation DataLoaders, and computes final fraud detection metrics.

    Returns
    -------
    dict with keys: recall, f1, precision, val_acc, val_loss,
                    best_recall, best_round, dp_epsilon
    """
    device = torch.device("cpu")
    torch.manual_seed(config.seed)

    vfl_cfg = VFLConfig(dataset="cifer")
    vfl_cfg.cifer.batch_size  = config.batch_size
    vfl_cfg.cifer.max_samples = config.max_samples

    (_, _, _), (val_a, val_b, val_server) = VFLDataLoaderFactory.get_cifer_loaders(
        config=vfl_cfg.cifer, batch_size=config.batch_size, seed=config.seed,
    )

    dim_a = val_a.dataset.feature_dim
    dim_b = val_b.dataset.feature_dim

    bottom_a = TabularBottomModel(
        input_dim=dim_a, embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
    )
    bottom_b = TabularBottomModel(
        input_dim=dim_b, embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
    )
    top_model = VFLTopModel(
        embedding_dim=config.embedding_dim, num_parties=2, num_classes=2,
    )

    n_pos = cifer_vfl_trained["n_pos_train"]
    n_neg = cifer_vfl_trained["n_neg_train"]
    class_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)

    trainer = VFLTrainer(
        bottom_model_a=bottom_a,
        bottom_model_b=bottom_b,
        top_model=top_model,
        optimizer_a=torch.optim.Adam(bottom_a.parameters()),
        optimizer_b=torch.optim.Adam(bottom_b.parameters()),
        optimizer_top=torch.optim.Adam(top_model.parameters()),
        criterion=torch.nn.CrossEntropyLoss(weight=class_weight),
        device=device,
        verbose=False,
    )
    trainer.load_model_state(cifer_vfl_trained["model_state"])

    val_metrics  = trainer.evaluate(val_a, val_b, val_server)
    fraud_report = _fraud_metrics(
        trainer, val_a, val_b, val_server,
        fraud_threshold=cifer_vfl_trained["fraud_threshold"],
    )

    result = {
        "recall":     fraud_report["recall"],
        "f1":         fraud_report["f1"],
        "precision":  fraud_report["precision"],
        "val_acc":    val_metrics["accuracy"],
        "val_loss":   val_metrics["loss"],
        "best_recall": cifer_vfl_trained["best_recall"],
        "best_round":  cifer_vfl_trained["best_round"],
        "dp_epsilon":  cifer_vfl_trained["dp_epsilon"],
    }

    print(
        f"\n[Dagster] cifer_vfl_eval — "
        f"recall={fraud_report['recall']:.4f}  "
        f"F1={fraud_report['f1']:.4f}  "
        f"precision={fraud_report['precision']:.4f}\n"
    )

    meta: dict = {
        "recall":    MetadataValue.float(fraud_report["recall"]),
        "f1":        MetadataValue.float(fraud_report["f1"]),
        "precision": MetadataValue.float(fraud_report["precision"]),
        "val_acc":   MetadataValue.float(val_metrics["accuracy"]),
    }
    if cifer_vfl_trained["dp_epsilon"] is not None:
        meta["dp_epsilon"] = MetadataValue.float(cifer_vfl_trained["dp_epsilon"])

    return Output(result, metadata=meta)
