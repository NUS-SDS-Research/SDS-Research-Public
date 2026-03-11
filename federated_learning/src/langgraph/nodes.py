"""
VFL LangGraph Nodes.

Each public function in this module is a *node factory* — it closes over the
non-serialisable runtime objects (VFLTrainer, DataLoaders, LR schedulers) and
returns a plain callable ``(state: VFLState) -> dict`` that LangGraph can
invoke as a graph node.

Design rationale
----------------
* VFLTrainer and DataLoaders are not JSON-serialisable, so they cannot live
  inside VFLState.  Instead they are captured at graph-build time via closure.
* Each node calls ``trainer.load_model_state(state["model_state"])`` at entry
  and ``trainer.get_model_state()`` at exit.  This ensures VFLState is always
  the authoritative checkpoint — the in-memory trainer is just the compute engine.
* ``metrics_history`` uses ``operator.add`` as its LangGraph reducer, so nodes
  return a list of new entries (not the full history).

Graph flow
----------
  START → train_round → evaluate → [should_stop?] → END
                 ↑_____________________↓  (continue)

Node factories
--------------
  make_train_round_node   — shared by MNIST and CiferAI
  make_evaluate_node      — MNIST: tracks best_val_acc
  make_cifer_evaluate_node — CiferAI: computes fraud metrics, tracks best_recall (M4)
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.langgraph.state import VFLState
from src.vfl.training_loop import VFLTrainer


# ---------------------------------------------------------------------------
# Fraud-specific metrics helper (mirrors run_cifer_vfl._fraud_metrics)
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

    The model is set to eval mode, inference is run with no_grad, then training
    mode is restored.  Mirrors the helper in run_cifer_vfl.py exactly.
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
            fraud_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (fraud_probs >= fraud_threshold).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    trainer.bottom_a.train()
    trainer.bottom_b.train()
    trainer.top.train()

    return {
        "f1":        f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "precision": precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
    }


def make_train_round_node(
    trainer: VFLTrainer,
    loader_a,
    loader_b,
    loader_server,
    schedulers: Optional[list] = None,
):
    """
    Factory for the train_round node.

    The returned node:
    1. Restores model weights from VFLState (load_model_state)
    2. Runs one epoch of VFL training (train_one_epoch)
    3. Steps all LR schedulers (if provided)
    4. Increments the round counter
    5. Saves updated weights back to VFLState (get_model_state)
    """
    def train_round_node(state: VFLState) -> dict:
        trainer.load_model_state(state["model_state"])

        train_metrics = trainer.train_one_epoch(loader_a, loader_b, loader_server)

        if schedulers:
            for sched in schedulers:
                sched.step()

        new_round = state["round"] + 1

        print(
            f"[Round {new_round:>3}] "
            f"train loss={train_metrics['loss']:.4f}  "
            f"train acc={train_metrics['accuracy']:.4f}"
        )

        return {
            "model_state": trainer.get_model_state(),
            "round": new_round,
            "metrics_history": [
                {"round": new_round, "phase": "train", **train_metrics}
            ],
        }

    return train_round_node


def make_evaluate_node(
    trainer: VFLTrainer,
    val_a,
    val_b,
    val_server,
):
    """
    Factory for the evaluate node.

    The returned node:
    1. Restores model weights from VFLState
    2. Runs evaluation (no gradients)
    3. Updates best_val_acc / best_round tracking
    4. Reads current DP epsilon from accountant (if DP enabled)
    """
    def evaluate_node(state: VFLState) -> dict:
        trainer.load_model_state(state["model_state"])

        val_metrics = trainer.evaluate(val_a, val_b, val_server)
        rnd = state["round"]
        val_acc = val_metrics["accuracy"]

        if val_acc > state["best_val_acc"]:
            best_val_acc = val_acc
            best_round = rnd
        else:
            best_val_acc = state["best_val_acc"]
            best_round = state["best_round"]

        dp_eps = state["dp_epsilon"]
        if trainer.dp_accountant is not None:
            dp_eps = trainer.dp_accountant.get_epsilon()

        print(
            f"          "
            f"val  loss={val_metrics['loss']:.4f}  "
            f"val  acc={val_acc:.4f}  "
            f"(best={best_val_acc:.4f} @ round {best_round})"
        )

        return {
            "metrics_history": [
                {"round": rnd, "phase": "val", **val_metrics}
            ],
            "best_val_acc": best_val_acc,
            "best_round": best_round,
            "dp_epsilon": dp_eps,
        }

    return evaluate_node


def make_cifer_evaluate_node(
    trainer: VFLTrainer,
    val_a,
    val_b,
    val_server,
    fraud_threshold: float = 0.5,
):
    """
    Factory for the CiferAI-specific evaluate node.

    The returned node:
    1. Restores model weights from VFLState
    2. Runs loss/accuracy evaluation (trainer.evaluate)
    3. Runs fraud-specific metrics (F1, precision, recall via _fraud_metrics)
    4. Updates best_recall tracking and stores M4 deepcopy checkpoint
       (best_model_state) when a new recall peak is found
    5. Reads current DP epsilon from accountant (if DP enabled)

    M4 checkpoint: best_model_state holds a deepcopy of model weights at the
    best-recall round.  The post-run script restores these weights for final
    re-evaluation and MLflow artefact logging.
    """
    def cifer_evaluate_node(state: VFLState) -> dict:
        trainer.load_model_state(state["model_state"])

        val_metrics  = trainer.evaluate(val_a, val_b, val_server)
        fraud_report = _fraud_metrics(
            trainer, val_a, val_b, val_server, fraud_threshold=fraud_threshold
        )

        rnd = state["round"]
        recall = fraud_report["recall"]

        # M4: deepcopy state at best-recall round
        if recall > state["best_recall"]:
            best_recall = recall
            best_round = rnd
            best_model_state = {
                "bottom_a": copy.deepcopy(trainer.bottom_a.state_dict()),
                "bottom_b": copy.deepcopy(trainer.bottom_b.state_dict()),
                "top":      copy.deepcopy(trainer.top.state_dict()),
            }
        else:
            best_recall = state["best_recall"]
            best_round = state["best_round"]
            best_model_state = state["best_model_state"]

        dp_eps = state["dp_epsilon"]
        if trainer.dp_accountant is not None:
            dp_eps = trainer.dp_accountant.get_epsilon()

        print(
            f"          "
            f"val  loss={val_metrics['loss']:.4f}  "
            f"val  acc={val_metrics['accuracy']:.4f} | "
            f"F1={fraud_report['f1']:.4f}  "
            f"recall={recall:.4f}  "
            f"prec={fraud_report['precision']:.4f}  "
            f"(best recall={best_recall:.4f} @ round {best_round})"
        )

        return {
            "metrics_history": [
                {
                    "round":        rnd,
                    "phase":        "val",
                    "loss":         val_metrics["loss"],
                    "accuracy":     val_metrics["accuracy"],
                    "f1":           fraud_report["f1"],
                    "recall":       fraud_report["recall"],
                    "precision":    fraud_report["precision"],
                }
            ],
            "best_recall":      best_recall,
            "best_round":       best_round,
            "best_model_state": best_model_state,
            "dp_epsilon":       dp_eps,
        }

    return cifer_evaluate_node
