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
    [CiferAI] Loaded 80000 train | 20000 val samples | fraud rate: 0.0013 | ...
    [Round  1] train loss=X.XXXX  train acc=X.XXXX | val loss=X.XXXX  val acc=X.XXXX | F1=X.XXXX  recall=X.XXXX
    ...

Notes
-----
- CiferAI is heavily class-imbalanced (~0.1% fraud rate).
  Accuracy is meaningless; F1 and recall for the fraud class are the key signals.
- Training uses minority oversampling (10% fraud) + class-weighted loss.
  The fraud_threshold (default 0.5) can be raised toward 0.9 to trade
  recall for precision if false positives dominate.
- Learning rate is constant 1e-4 throughout (no scheduler). CosineAnnealingLR
  was removed because it decayed LR to 1e-6 by round 25, killing the fraud
  signal in later rounds. Constant LR keeps every round equally capable of
  detecting fraud; best recall across rounds is reported in the summary.
- max_samples=100_000 (default) keeps first runs fast.
  Set config.cifer.max_samples = None for the full 6.3M row dataset.

MLflow tracking 
------------------------
Each run is logged to the "VFL-CiferAI" experiment (./mlruns by default).
View results:
    mlflow ui          # opens http://localhost:5000
Params logged: num_rounds, lr, embedding_dim, dp_*, oversample_*, fraud_threshold
Metrics logged per round: train_loss, train_accuracy, val_loss, val_accuracy,
                          val_f1, val_recall, val_precision
Summary metrics: best_recall, best_recall_round, best_f1, best_f1_round,
                 best_model_recall, best_model_f1, best_model_precision, dp_epsilon
Model artefacts: top_model, bottom_model_a, bottom_model_b (best-recall checkpoint)
"""
from __future__ import annotations

import copy
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from configs.vfl_config import VFLConfig
from src.datasets.dataloader_factory import VFLDataLoaderFactory
from src.models.bottom_models import TabularBottomModel
from src.models.top_model import VFLTopModel
from src.vfl.training_loop import VFLTrainer


# ---------------------------------------------------------------------------
# Fraud-specific metrics helper
# ---------------------------------------------------------------------------

def _fraud_metrics(
    trainer: VFLTrainer,
    loader_a,
    loader_b,
    loader_server,
    fraud_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Run inference and return F1, precision, and recall for the fraud class (label=1).

    Parameters
    ----------
    fraud_threshold : float
        Minimum softmax probability for the fraud class before a sample is
        classified as fraud.  Default 0.5 (equivalent to argmax for 2 classes).

        The model is trained on an oversampled distribution (10% fraud) but
        evaluated on the real distribution (~0.12% fraud).  A threshold above
        0.5 corrects for this mismatch: it reduces false positives (improving
        precision) at some cost to recall.  Tune between 0.5 and 0.95.
    """
    trainer.bottom_a.eval()
    trainer.bottom_b.eval()
    trainer.top.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x_a, x_b, labels in zip(loader_a, loader_b, loader_server):
            x_a = x_a.to(trainer.device)
            x_b = x_b.to(trainer.device)
            emb_a = trainer.bottom_a(x_a)
            emb_b = trainer.bottom_b(x_b)
            logits = trainer.top(emb_a, emb_b)
            # Use softmax probability of the fraud class vs a tunable threshold
            # (C2: prevents over-flagging caused by oversampling calibration gap)
            fraud_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (fraud_probs >= fraud_threshold).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Restore train mode
    trainer.bottom_a.train()
    trainer.bottom_b.train()
    trainer.top.train()

    return {
        "f1":        f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "precision": precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_pure_pytorch(config: VFLConfig) -> None:
    """Run CiferAI VFL training in pure PyTorch mode."""
    print("=" * 60)
    print(" CiferAI Fraud Detection VFL — Pure PyTorch Mode")
    print("=" * 60)

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    # ── open MLflow run ──────────────────────────────────────────
    # Use file-based backend (mlruns/) anchored to the project root.
    # This avoids the SQLite backend that triggers MLflow's auth middleware.
    # View runs with: mlflow ui --backend-store-uri ./mlruns
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlflow.set_tracking_uri("file:///"  + os.path.join(_project_root, "mlruns").replace("\\", "/"))
    mlflow.set_experiment("VFL-CiferAI")
    run_name = (
        f"cifer-dp{'on' if config.dp.enabled else 'off'}"
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
        })

        # ── Load dataset — single call, returns both train and val loaders ──
        # B1: removed the redundant build_aligned_pair() call that was here.
        # B2: get_cifer_loaders() now returns (train_loaders, val_loaders).
        (loader_a, loader_b, loader_server), (val_a, val_b, val_server) = (
            VFLDataLoaderFactory.get_cifer_loaders(
                config=config.cifer,
                batch_size=config.cifer.batch_size,
                seed=config.seed,
            )
        )

        # ── Infer feature dimensions from the train datasets ────────────────
        # B1: feature_dim read from loader.dataset — no redundant factory call.
        dim_a = loader_a.dataset.feature_dim    # e.g. 3 (step, nameOrig_freq, nameDest_freq)
        dim_b = loader_b.dataset.feature_dim    # e.g. 7 (type, amount, 4 balances, isFlaggedFraud)

        print(f"Party A feature dim: {dim_a}  |  Party B feature dim: {dim_b}")

        # ── Class weights for imbalanced fraud detection ─────────────────────
        # B3: CrossEntropyLoss with weight=[1, n_neg/n_pos] balances the 99.87/0.13
        #     class split so the model cannot trivially collapse to 'predict no fraud'.
        labels_all: np.ndarray = loader_server.dataset._labels
        n_pos = int(labels_all.sum())
        n_neg = int(len(labels_all) - n_pos)
        class_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
        print(
            f"Class weights — non-fraud: {class_weight[0]:.2f} | "
            f"fraud: {class_weight[1]:.1f}  (n_neg={n_neg}, n_pos={n_pos})"
        )
        mlflow.log_params({"n_pos_train": n_pos, "n_neg_train": n_neg})

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
        # M2: use CiferAI-specific learning rate (1e-4), lower than MNIST's 1e-3.
        #     The oversampled training landscape causes limit-cycle oscillation at
        #     lr=1e-3; 1e-4 stabilises the gradient steps.
        cifer_lr = config.cifer.learning_rate
        opt_a   = torch.optim.Adam(bottom_a.parameters(), lr=cifer_lr)
        opt_b   = torch.optim.Adam(bottom_b.parameters(), lr=cifer_lr)
        opt_top = torch.optim.Adam(top_model.parameters(), lr=cifer_lr)

        # M3: No LR scheduler for CiferAI. CosineAnnealingLR decayed LR to 1e-6
        #     by round 25, which killed fraud detection in later rounds (recall
        #     dropped from 78% at round 2 to 4% at round 25). Constant LR=1e-4
        #     keeps all rounds equally capable of updating toward fraud detection.

        # ── Trainer ─────────────────────────────────────────────────────────
        # pass dp_config so the trainer wires up DP clip+noise and
        #         the privacy budget accountant when config.dp.enabled=True.
        trainer = VFLTrainer(
            bottom_model_a=bottom_a,
            bottom_model_b=bottom_b,
            top_model=top_model,
            optimizer_a=opt_a,
            optimizer_b=opt_b,
            optimizer_top=opt_top,
            criterion=torch.nn.CrossEntropyLoss(weight=class_weight),   # B3
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

        # ── Training loop ────────────────────────────────────────────────────
        val_metrics  = {"loss": float("nan"), "accuracy": float("nan")}
        fraud_report = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        # Track best-performing round for the final summary.
        # M4: save a deepcopy of model weights at best-recall round so we can
        #     restore and re-evaluate after training ends (models eventually
        #     collapse to "predict no-fraud" as Adam v-hat accumulates).
        best_recall       = 0.0
        best_recall_round = 0
        best_f1           = 0.0
        best_f1_round     = 0
        best_state: dict | None = None

        for rnd in range(1, config.num_rounds + 1):
            train_metrics = trainer.train_one_epoch(loader_a, loader_b, loader_server)

            # B2: evaluate() now uses the real held-out val split
            val_metrics  = trainer.evaluate(val_a, val_b, val_server)
            fraud_report = _fraud_metrics(
                trainer, val_a, val_b, val_server,
                fraud_threshold=config.cifer.fraud_threshold,
            )

            if fraud_report["recall"] > best_recall:
                best_recall       = fraud_report["recall"]
                best_recall_round = rnd
                best_state = {
                    "bottom_a": copy.deepcopy(trainer.bottom_a.state_dict()),
                    "bottom_b": copy.deepcopy(trainer.bottom_b.state_dict()),
                    "top":      copy.deepcopy(trainer.top.state_dict()),
                }
            if fraud_report["f1"] > best_f1:
                best_f1       = fraud_report["f1"]
                best_f1_round = rnd

            print(
                f"[Round {rnd:>3}] "
                f"train loss={train_metrics['loss']:.4f}  "
                f"train acc={train_metrics['accuracy']:.4f} | "
                f"val loss={val_metrics['loss']:.4f}  "
                f"val acc={val_metrics['accuracy']:.4f} | "
                f"F1={fraud_report['f1']:.4f}  "
                f"recall={fraud_report['recall']:.4f}  "
                f"prec={fraud_report['precision']:.4f}  "
                f"lr={cifer_lr:.2e}"
            )

            # log per-round metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss":     train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss":       val_metrics["loss"],
                    "val_accuracy":   val_metrics["accuracy"],
                    "val_f1":         fraud_report["f1"],
                    "val_recall":     fraud_report["recall"],
                    "val_precision":  fraud_report["precision"],
                },
                step=rnd,
            )

        # M4: restore best-recall model weights and re-evaluate for the final report.
        #     Models often collapse to "predict no-fraud" in later rounds; the best
        #     checkpoint reflects the true fraud detection capability of the system.
        if best_state is not None:
            trainer.bottom_a.load_state_dict(best_state["bottom_a"])
            trainer.bottom_b.load_state_dict(best_state["bottom_b"])
            trainer.top.load_state_dict(best_state["top"])
            best_report = _fraud_metrics(
                trainer, val_a, val_b, val_server,
                fraud_threshold=config.cifer.fraud_threshold,
            )
        else:
            best_report = fraud_report

        print("\nTraining complete.")
        print(
            f"Final   (round {config.num_rounds:>2}) — "
            f"val acc: {val_metrics['accuracy']:.4f}  "
            f"F1: {fraud_report['f1']:.4f}  "
            f"recall: {fraud_report['recall']:.4f}  "
            f"precision: {fraud_report['precision']:.4f}"
        )
        print(
            f"Best F1     (round {best_f1_round:>2}) — "
            f"F1={best_f1:.4f}"
        )
        print(
            f"Best model  (round {best_recall_round:>2}) — "
            f"F1={best_report['f1']:.4f}  "
            f"recall={best_report['recall']:.4f}  "
            f"precision={best_report['precision']:.4f}"
        )
        print("(Accuracy is dominated by the majority class; F1/recall are the key signals.)")

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

        # log summary metrics (best-round results) to MLflow
        mlflow.log_metrics({
            "best_recall":           best_recall,
            "best_recall_round":     float(best_recall_round),
            "best_f1":               best_f1,
            "best_f1_round":         float(best_f1_round),
            "best_model_recall":     best_report["recall"],
            "best_model_f1":         best_report["f1"],
            "best_model_precision":  best_report["precision"],
        })

        # log best-recall checkpoint as model artefacts
        with tempfile.TemporaryDirectory() as tmp:
            for name, model in [("top_model", trainer.top), ("bottom_model_a", trainer.bottom_a), ("bottom_model_b", trainer.bottom_b)]:
                path = os.path.join(tmp, f"{name}.pt")
                torch.save(model.state_dict(), path)
                mlflow.log_artifact(path)

        print(f"[MLflow] Run complete — experiment: VFL-CiferAI  name: {run_name}")


if __name__ == "__main__":
    cfg = VFLConfig(dataset="cifer")
    cfg.dp.enabled = True
    cfg.dp.clip_norm = 1.0
    cfg.dp.noise_multiplier = 1.0
    run_pure_pytorch(cfg)
