"""
Flower Passive-Party Client for VFL Simulation.

Each client represents one passive party (Party A = cid "0", Party B = cid "1").
It owns a bottom model and a vertical slice of the training data.

Round protocol (client side)
-----------------------------
fit(parameters, config):
  1. Decode received gradient NDArray from `parameters`.
  2. If gradient is non-empty and a local embedding is cached (not round 1):
       optimizer.zero_grad()
       cached_local_emb.backward(gradient_tensor)
       optimizer.step()
  3. Fetch next batch from the cycling data iterator.
  4. local_emb = bottom_model(x_batch)      [cache for next round]
  5. sent_emb  = local_emb.detach().requires_grad_(True)
  6. Return [sent_emb.numpy()], batch_size, {"party_id": int(cid)}

State across Flower rounds
---------------------------
Flower's start_simulation() invokes client_fn(cid) every round, which would
normally discard all client state. To persist the bottom model weights and the
data iterator, two module-level dicts (keyed by cid string) are used.
Call register_client() for each party before starting the simulation.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import NDArrays

from src.vfl.gradient_bridge import EmbeddingGradientBridge


# ---------------------------------------------------------------------------
# Module-level persistent state (survives across Flower rounds in one run)
# ---------------------------------------------------------------------------

_client_models: dict[str, nn.Module] = {}       # cid -> bottom model
_client_state: dict[str, dict] = {}             # cid -> {optimizer, data_iter, local_emb, ...}


def register_client(
    cid: str,
    bottom_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Register a passive party before the Flower simulation starts.

    Parameters
    ----------
    cid : str
        "0" for Party A, "1" for Party B.
    bottom_model : nn.Module
        The party's feature extractor (TabularBottomModel or CNNBottomModel).
    optimizer : Optimizer
    data_loader : DataLoader
        Vertical feature slice loader (aligned, drop_last=True).
    device : torch.device
    """
    _client_models[cid] = bottom_model.to(device)
    _client_state[cid] = {
        "optimizer": optimizer,
        "data_iter": cycle(data_loader),   # never exhausted mid-simulation
        "local_emb": None,                 # cached pre-detach tensor for backward
        "device": device,
        "party_id": int(cid),
    }


# ---------------------------------------------------------------------------
# NumPyClient
# ---------------------------------------------------------------------------

class VFLPassiveClient(NumPyClient):
    """Flower client representing a single VFL passive party."""

    def __init__(self, cid: str) -> None:
        self.cid = cid
        self._bridge = EmbeddingGradientBridge()

    def get_parameters(self, config: dict) -> NDArrays:
        """Return current bottom model weights (for optional checkpointing)."""
        model = _client_models[self.cid]
        return [p.detach().cpu().numpy() for p in model.parameters()]

    def fit(self, parameters: NDArrays, config: dict) -> tuple[NDArrays, int, dict]:
        """
        Delayed-gradient VFL step:
          apply previous gradient → compute new embedding → return embedding.
        """
        state = _client_state[self.cid]
        model = _client_models[self.cid]
        device: torch.device = state["device"]
        optimizer: torch.optim.Optimizer = state["optimizer"]

        model.train()

        # ── Apply received gradient (skip on round 1) ───────────────────
        has_gradient = (
            len(parameters) > 0
            and parameters[0].size > 0
            and state["local_emb"] is not None
        )
        if has_gradient:
            grad_tensor = torch.from_numpy(parameters[0]).float().to(device)
            optimizer.zero_grad()
            self._bridge.apply_gradient(state["local_emb"], grad_tensor)
            optimizer.step()

        # ── Fetch next batch and compute embedding ──────────────────────
        x_batch = next(state["data_iter"])
        x_batch = x_batch.to(device)

        local_emb = model(x_batch)
        state["local_emb"] = local_emb                          # cache for backward

        sent_emb = self._bridge.detach_for_transmission(local_emb)
        emb_array = sent_emb.detach().cpu().numpy()

        return [emb_array], x_batch.size(0), {"party_id": state["party_id"]}

    def evaluate(self, parameters: NDArrays, config: dict) -> tuple[float, int, dict]:
        """Evaluation is done server-side; return placeholder."""
        state = _client_state[self.cid]
        return 0.0, 1, {"party_id": state["party_id"]}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client_fn():
    """
    Return a client_fn compatible with fl.simulation.start_simulation().

    Call register_client() for every cid before starting simulation.

    Example
    -------
    register_client("0", bottom_a, opt_a, loader_a, device)
    register_client("1", bottom_b, opt_b, loader_b, device)
    fl.simulation.start_simulation(client_fn=make_client_fn(), ...)
    """
    def client_fn(cid: str) -> VFLPassiveClient:
        if cid not in _client_models:
            raise RuntimeError(
                f"Client cid='{cid}' not registered. "
                "Call register_client() before starting simulation."
            )
        return VFLPassiveClient(cid)

    return client_fn