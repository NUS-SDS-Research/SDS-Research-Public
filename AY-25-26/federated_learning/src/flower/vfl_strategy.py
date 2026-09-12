"""
Custom Flower Strategy for Vertical Federated Learning.

VFL differs fundamentally from Horizontal FL (HFL):
  HFL : clients send MODEL WEIGHTS → server averages them (FedAvg).
  VFL : clients send EMBEDDINGS    → server runs top model, sends GRADIENTS back.

Flower's Parameters type is repurposed here:
  FitIns.parameters  = gradient arrays (server → client, per party)
  FitRes.parameters  = embedding arrays (client → server, per party)

Round protocol
--------------
configure_fit()   → sends stored gradient (from previous aggregate_fit) to each client
                    (round 1: sends empty signal → client skips backward, computes embedding)
aggregate_fit()   → receives embeddings from all clients
                    → runs top-model forward + backward
                    → stores per-client gradients for next configure_fit()
                    → returns empty Parameters (gradients are cached, not sent here)
configure_evaluate() / aggregate_evaluate()  → standard evaluation pass

Client identification
---------------------
Each client includes party_id (0 or 1) in its FitRes.metrics dict.
The strategy reads metrics["party_id"] to assign the correct gradient.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from src.models.top_model import VFLTopModel


class VFLStrategy(Strategy):
    """
    Flower Strategy for two-party VFL simulation.

    Parameters
    ----------
    top_model : VFLTopModel
        Server-side aggregation head.
    optimizer_top : torch.optim.Optimizer
        Optimizer for the top model.
    criterion : nn.Module
        Loss function (CrossEntropyLoss).
    label_loader : DataLoader
        Server's label DataLoader. Must be aligned with client loaders
        (same seed, same batch size, drop_last=True).
    device : torch.device
    embedding_dim : int
        Per-party embedding size (default 128).
    num_parties : int
        Number of passive parties (default 2).
    fraction_fit : float
        Fraction of clients sampled per round (1.0 = all clients).
    min_fit_clients : int
        Minimum clients required to start a fit round.
    min_available_clients : int
        Minimum clients that must be connected before training starts.
    """

    def __init__(
        self,
        top_model: VFLTopModel,
        optimizer_top: torch.optim.Optimizer,
        criterion: nn.Module,
        label_loader: DataLoader,
        device: torch.device,
        embedding_dim: int = 128,
        num_parties: int = 2,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        self.top_model = top_model.to(device)
        self.optimizer_top = optimizer_top
        self.criterion = criterion
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_parties = num_parties
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

        # Label iterator — advances one batch per aggregate_fit call
        self._label_iter = iter(label_loader)
        self._label_loader = label_loader  # kept for re-iteration at epoch boundaries

        # Gradient cache: {party_id (int): np.ndarray or None}
        # None signals "round 1 — skip backward on client side"
        self._gradient_cache: dict[int, Optional[np.ndarray]] = {
            i: None for i in range(num_parties)
        }

        self._round = 0

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Clients initialise their own bottom models; server sends nothing."""
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Send each client its gradient from the previous round.
        Round 1: send empty Parameters (client detects this and skips backward).
        """
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        fit_configs = []
        for proxy in clients:
            party_id = int(proxy.cid)
            cached_grad = self._gradient_cache.get(party_id)

            if cached_grad is None:
                # Round 1: send empty array as signal
                params = ndarrays_to_parameters([np.array([])])
            else:
                params = ndarrays_to_parameters([cached_grad])

            fit_configs.append((proxy, FitIns(parameters=params, config={"round": server_round})))

        return fit_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Receive embeddings → top model forward/backward → store gradients.

        Returns empty Parameters (gradients are cached for next configure_fit).
        """
        if failures:
            print(f"[VFLStrategy] Round {server_round}: {len(failures)} client failures.")

        # Collect embeddings keyed by party_id (from metrics)
        embeddings: dict[int, np.ndarray] = {}
        num_samples = 0
        for _proxy, fitres in results:
            pid = int(fitres.metrics.get("party_id", int(_proxy.cid)))
            arrays = parameters_to_ndarrays(fitres.parameters)
            if arrays and arrays[0].size > 0:
                embeddings[pid] = arrays[0]   # shape (batch, embedding_dim)
                num_samples = fitres.num_examples

        if len(embeddings) < self.num_parties:
            print(f"[VFLStrategy] Skipping round {server_round}: missing party embeddings.")
            return ndarrays_to_parameters([]), {}

        # Stack embeddings in party order: [Party 0, Party 1, ...]
        emb_tensors = []
        for pid in range(self.num_parties):
            arr = embeddings[pid]
            t = torch.from_numpy(arr).float().to(self.device)
            t = t.detach().requires_grad_(True)   # leaf node for gradient extraction
            emb_tensors.append(t)

        # Get matching label batch from server DataLoader
        try:
            labels = next(self._label_iter)
        except StopIteration:
            self._label_iter = iter(self._label_loader)
            labels = next(self._label_iter)
        labels = labels.to(self.device)

        # Top model forward + backward
        self.top_model.train()
        concat_emb = torch.cat(emb_tensors, dim=1)   # (batch, num_parties * embedding_dim)
        logits = self.top_model.forward_from_concat(concat_emb)

        # Trim label batch to match embedding batch size (drop_last may differ across loaders)
        batch_size = logits.size(0)
        labels = labels[:batch_size]

        loss = self.criterion(logits, labels)
        self.optimizer_top.zero_grad()
        loss.backward()
        self.optimizer_top.step()

        # Extract and store per-party gradients
        accuracy = (logits.detach().argmax(dim=1) == labels).float().mean().item()
        for pid, emb_t in enumerate(emb_tensors):
            if emb_t.grad is not None:
                self._gradient_cache[pid] = emb_t.grad.clone().cpu().numpy()

        metrics: dict[str, Scalar] = {
            "loss": float(loss.item()),
            "accuracy": float(accuracy),
            "round": server_round,
        }
        print(
            f"[Server] Round {server_round:>3} | "
            f"loss={loss.item():.4f} | acc={accuracy:.4f}"
        )
        self._round = server_round
        return ndarrays_to_parameters([]), metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Tell clients to compute evaluation embeddings."""
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        eval_ins = EvaluateIns(parameters=ndarrays_to_parameters([]), config={"eval": True})
        return [(proxy, eval_ins) for proxy in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate loss values reported by clients (placeholder — eval is server-side)."""
        if not results:
            return None, {}
        # Clients return placeholder 0.0; real evaluation happens in aggregate_fit
        return 0.0, {}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Server-side evaluation is handled inside aggregate_fit."""
        return None