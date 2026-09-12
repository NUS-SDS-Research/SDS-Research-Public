"""
Flower Server-Side Factory for VFL Simulation.

Assembles the VFLStrategy (top model + optimizer + label loader) and
returns the objects needed by fl.simulation.start_simulation().

Usage
-----
    from src.flower.server_app import make_vfl_server_components

    strategy, server_config = make_vfl_server_components(
        config=vfl_config,
        label_loader=loader_server,
        num_classes=10,
    )
    fl.simulation.start_simulation(
        client_fn=make_client_fn(),
        num_clients=2,
        config=server_config,
        strategy=strategy,
    )
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl

from configs.vfl_config import VFLConfig
from src.models.top_model import VFLTopModel
from src.flower.vfl_strategy import VFLStrategy


def make_vfl_server_components(
    config: VFLConfig,
    label_loader: DataLoader,
    num_classes: int,
) -> tuple[VFLStrategy, fl.server.ServerConfig]:
    """
    Build the VFLStrategy and ServerConfig for Flower simulation.

    Parameters
    ----------
    config : VFLConfig
        Master training configuration.
    label_loader : DataLoader
        Server-side label DataLoader. Must be aligned with client loaders.
    num_classes : int
        10 for MNIST, 2 for CiferAI.

    Returns
    -------
    (strategy, server_config)
        Pass directly to fl.simulation.start_simulation().
    """
    device = torch.device(config.device)

    top_model = VFLTopModel(
        embedding_dim=config.embedding_dim,
        num_parties=2,
        num_classes=num_classes,
    )

    optimizer_top = torch.optim.Adam(
        top_model.parameters(),
        lr=config.learning_rate,
    )

    strategy = VFLStrategy(
        top_model=top_model,
        optimizer_top=optimizer_top,
        criterion=nn.CrossEntropyLoss(),
        label_loader=label_loader,
        device=device,
        embedding_dim=config.embedding_dim,
        num_parties=2,
        min_fit_clients=2,
        min_available_clients=2,
    )

    server_config = fl.server.ServerConfig(num_rounds=config.num_rounds)

    return strategy, server_config