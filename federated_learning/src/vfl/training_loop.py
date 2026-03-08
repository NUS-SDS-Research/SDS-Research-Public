"""
VFL Training Loop.

Implements the full synchronous Vertical Federated Learning training protocol
for two passive parties and one active party (server), running in a single
process (no actual network I/O).

VFL mini-batch protocol (10 steps)
-----------------------------------
 1. Party A: local_emb_a = bottom_model_a(x_a)
 2. Party B: local_emb_b = bottom_model_b(x_b)
 3. Detach both for 'transmission':
      sent_a = local_emb_a.detach().requires_grad_(True)
      sent_b = local_emb_b.detach().requires_grad_(True)
 4. Active party: logits = top_model(sent_a, sent_b)
 5. Active party: loss = criterion(logits, labels)
 6. optimizer_top.zero_grad()
    loss.backward()
    # sent_a.grad = dL/d_sent_a  |  sent_b.grad = dL/d_sent_b
 7. Extract: grad_a = sent_a.grad.clone()
             grad_b = sent_b.grad.clone()
 8. optimizer_top.step()
 9. optimizer_a.zero_grad()
    local_emb_a.backward(grad_a)
    optimizer_a.step()
10. optimizer_b.zero_grad()
    local_emb_b.backward(grad_b)
    optimizer_b.step()

Optimizer ordering notes
------------------------
- top model optimizer steps BEFORE passive party optimizers (step 8 before 9/10).
  This is safe because step() only modifies parameter values; it does not clear
  sent_emb.grad, which is still needed in step 7.
- zero_grad() for each bottom optimizer is called immediately before its own
  backward call (steps 9/10), NOT before loss.backward(). This avoids
  accidentally zeroing stale gradients before they have been used.

MLflow compatibility
------------------------------
train_one_epoch() and evaluate() both return dict[str, float] with keys
{"loss", "accuracy"} — directly passable to mlflow.log_metrics().

LangGraph / Dagster compatibility
-----------------------------------
get_model_state() / load_model_state() provide clean checkpoint interfaces
for agentic control and Dagster job resumption.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.top_model import VFLTopModel
from src.vfl.gradient_bridge import EmbeddingGradientBridge


class VFLTrainer:
    """
    Orchestrates VFL training for two passive parties and one active party.

    Parameters
    ----------
    bottom_model_a : nn.Module
        Party A's bottom model (feature extractor). Produces 128-dim embedding.
    bottom_model_b : nn.Module
        Party B's bottom model. Produces 128-dim embedding.
    top_model : VFLTopModel
        Active party aggregation head. Takes concat([emb_a, emb_b]) → logits.
    optimizer_a : Optimizer
        Optimizer for bottom_model_a.
    optimizer_b : Optimizer
        Optimizer for bottom_model_b.
    optimizer_top : Optimizer
        Optimizer for top_model.
    criterion : nn.Module
        Loss function. CrossEntropyLoss for both MNIST (10-class) and CiferAI (2-class).
    device : torch.device
        Computation device.
    verbose : bool
        Show per-epoch tqdm progress bar if True.
    dp_config : DPConfig or None
        When provided and ``dp_config.enabled=True``, embeddings are clipped and
        noised at the cut-layer before transmission (Epic 3). Provides (ε, δ)-DP
        guarantees against gradient inversion attacks. ``None`` = no DP.
    """

    def __init__(
        self,
        bottom_model_a: nn.Module,
        bottom_model_b: nn.Module,
        top_model: VFLTopModel,
        optimizer_a: torch.optim.Optimizer,
        optimizer_b: torch.optim.Optimizer,
        optimizer_top: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        verbose: bool = True,
        dp_config: Optional[object] = None,
    ) -> None:
        self.bottom_a = bottom_model_a.to(device)
        self.bottom_b = bottom_model_b.to(device)
        self.top = top_model.to(device)
        self.opt_a = optimizer_a
        self.opt_b = optimizer_b
        self.opt_top = optimizer_top
        self.criterion = criterion
        self.device = device
        self.verbose = verbose
        self._bridge = EmbeddingGradientBridge()
        self._n_train: int = 0  # set in train_one_epoch; used for DP sample rate

        # Epic 3 — DP setup
        self._dp_config = dp_config
        if dp_config is not None and getattr(dp_config, "enabled", False):
            from src.vfl.dp_accountant import DPBudgetAccountant
            self.dp_accountant: Optional[object] = DPBudgetAccountant(
                noise_multiplier=dp_config.noise_multiplier,
                delta=dp_config.delta,
            )
        else:
            self.dp_accountant = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_one_epoch(
        self,
        loader_a: DataLoader,
        loader_b: DataLoader,
        loader_server: DataLoader,
    ) -> dict[str, float]:
        """
        Run one full epoch of VFL training.

        Parameters
        ----------
        loader_a, loader_b, loader_server : DataLoader
            Must be strictly index-aligned (created by VFLDataLoaderFactory).

        Returns
        -------
        dict with keys "loss" and "accuracy" (epoch averages).
        """
        self.bottom_a.train()
        self.bottom_b.train()
        self.top.train()

        # Store dataset size so _train_one_batch can compute DP sample rate
        self._n_train = len(loader_a.dataset)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        iterator = zip(loader_a, loader_b, loader_server)
        if self.verbose:
            iterator = tqdm(iterator, total=len(loader_a), desc="Train", leave=False)

        for x_a, x_b, labels in iterator:
            loss, n_correct, n_samples = self._train_one_batch(x_a, x_b, labels)
            total_loss += loss * n_samples
            total_correct += n_correct
            total_samples += n_samples

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def evaluate(
        self,
        loader_a: DataLoader,
        loader_b: DataLoader,
        loader_server: DataLoader,
    ) -> dict[str, float]:
        """
        Evaluation-only forward pass (no gradient tracking).

        Returns
        -------
        dict with keys "loss" and "accuracy".
        """
        self.bottom_a.eval()
        self.bottom_b.eval()
        self.top.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_a, x_b, labels in zip(loader_a, loader_b, loader_server):
                x_a = x_a.to(self.device)
                x_b = x_b.to(self.device)
                labels = labels.to(self.device)

                emb_a = self.bottom_a(x_a)
                emb_b = self.bottom_b(x_b)
                logits = self.top(emb_a, emb_b)
                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def get_model_state(self) -> dict:
        """
        Return state dicts for all three models.

        Used by Flower integration, LangGraph checkpointing (Epic 5),
        and Dagster asset materialisation (Epic 6).
        """
        return {
            "bottom_a": self.bottom_a.state_dict(),
            "bottom_b": self.bottom_b.state_dict(),
            "top": self.top.state_dict(),
        }

    def load_model_state(self, state: dict) -> None:
        """Restore model weights from a state dict produced by get_model_state()."""
        self.bottom_a.load_state_dict(state["bottom_a"])
        self.bottom_b.load_state_dict(state["bottom_b"])
        self.top.load_state_dict(state["top"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_one_batch(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, int, int]:
        """
        Execute the full 10-step VFL protocol for a single mini-batch.

        Returns
        -------
        (loss_value, num_correct, batch_size)
        """
        x_a = x_a.to(self.device)
        x_b = x_b.to(self.device)
        labels = labels.to(self.device)

        # ── Steps 1 & 2: Passive parties compute local embeddings ──────
        local_emb_a = self.bottom_a(x_a)   # grad_fn connected to bottom_a params
        local_emb_b = self.bottom_b(x_b)   # grad_fn connected to bottom_b params

        # ── Step 3: Detach for 'transmission' (+ DP clip+noise if enabled) ──
        dp = self._dp_config
        clip  = dp.clip_norm         if (dp and dp.enabled) else None
        sigma = dp.noise_multiplier  if (dp and dp.enabled) else None
        sent_a = self._bridge.detach_for_transmission(local_emb_a, clip_norm=clip, noise_multiplier=sigma)
        sent_b = self._bridge.detach_for_transmission(local_emb_b, clip_norm=clip, noise_multiplier=sigma)

        # ── Epic 3: Increment DP privacy budget accountant ─────────────
        if self.dp_accountant is not None and self._n_train > 0:
            sample_rate = local_emb_a.size(0) / self._n_train
            self.dp_accountant.step(sample_rate=sample_rate)  # party A noise
            self.dp_accountant.step(sample_rate=sample_rate)  # party B noise

        # ── Steps 4 & 5: Active party forward + loss ───────────────────
        logits = self.top(sent_a, sent_b)
        loss = self.criterion(logits, labels)

        # ── Step 6: Active party backward ──────────────────────────────
        # Zero only the top-model gradients here.  Bottom-model grads are
        # zeroed independently (steps 9/10) right before their own backward.
        self.opt_top.zero_grad()
        loss.backward()
        # sent_a.grad = dL/d_sent_a  |  sent_b.grad = dL/d_sent_b

        # ── Step 7: Extract gradients for passive parties ───────────────
        grad_a = self._bridge.extract_gradient(sent_a)
        grad_b = self._bridge.extract_gradient(sent_b)

        # ── Step 8: Update top model ────────────────────────────────────
        self.opt_top.step()

        # ── Steps 9 & 10: Update passive party bottom models ───────────
        self.opt_a.zero_grad()
        self._bridge.apply_gradient(local_emb_a, grad_a)
        self.opt_a.step()

        self.opt_b.zero_grad()
        self._bridge.apply_gradient(local_emb_b, grad_b)
        self.opt_b.step()

        # ── Metrics ─────────────────────────────────────────────────────
        batch_size = labels.size(0)
        n_correct = (logits.detach().argmax(dim=1) == labels).sum().item()
        return loss.item(), int(n_correct), batch_size