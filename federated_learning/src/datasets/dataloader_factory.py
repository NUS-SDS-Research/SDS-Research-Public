"""
Strictly Index-Aligned VFL DataLoaders.

The core alignment guarantee:
  A single torch.Generator is seeded once and shared across all DataLoaders.
  Because every DataLoader uses the same RNG state sequence for shuffling,
  batch i from Party A always corresponds to batch i from Party B and the server.

Usage
-----
    from src.datasets.dataloader_factory import VFLDataLoaderFactory
    from configs.vfl_config import VFLConfig

    config = VFLConfig(dataset="mnist")
    loader_a, loader_b, loader_server = VFLDataLoaderFactory.get_mnist_loaders(config.mnist)

    for x_a, x_b, labels in zip(loader_a, loader_b, loader_server):
        ...  # guaranteed to be the same sample index
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from configs.vfl_config import CiferConfig, MNISTConfig, VFLConfig
from src.datasets.mnist_dataset import MNISTVerticalDataset
from src.datasets.cifer_dataset import CiferVerticalDataset


class VFLDataLoaderFactory:
    """
    Factory for creating strictly index-aligned DataLoaders for VFL.

    All loaders produced by a single factory call share one torch.Generator,
    ensuring that the shuffle order is identical across all parties.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_mnist_loaders(
        config: MNISTConfig,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 42,
        train: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build aligned DataLoaders for the MNIST vertical split.

        Parameters
        ----------
        config : MNISTConfig
        batch_size : int, optional
            Overrides config.batch_size when provided.
        num_workers : int
            Number of worker processes. Keep 0 unless you add worker_init_fn.
        seed : int
            RNG seed for the shared generator.
        train : bool
            Whether to use the training or test split.

        Returns
        -------
        (loader_a, loader_b, loader_server)
            Three DataLoaders aligned by index.
        """
        ds_a, ds_b, ds_server = MNISTVerticalDataset.build_aligned_pair(config, train=train)
        bs = batch_size if batch_size is not None else config.batch_size
        return tuple(
            VFLDataLoaderFactory._make_aligned_loaders(
                datasets=[ds_a, ds_b, ds_server],
                batch_size=bs,
                shuffle=train,
                seed=seed,
                num_workers=num_workers,
                drop_last=True,     # avoid single-sample batches (breaks BatchNorm)
            )
        )

    @staticmethod
    def get_cifer_loaders(
        config: CiferConfig,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 42,
    ) -> tuple[
        tuple[DataLoader, DataLoader, DataLoader],
        tuple[DataLoader, DataLoader, DataLoader],
    ]:
        """
        Build aligned DataLoaders for the CiferAI vertical split.

        Returns
        -------
        (train_loaders, val_loaders)
            Each is a 3-tuple (loader_a, loader_b, loader_server).
            Train loaders shuffle with drop_last=True.
            Val loaders are sequential with drop_last=False.
        """
        (ds_a_train, ds_b_train, ds_server_train), (ds_a_val, ds_b_val, ds_server_val) = (
            CiferVerticalDataset.build_aligned_pair(config)
        )
        bs = batch_size if batch_size is not None else config.batch_size

        train_loaders = tuple(
            VFLDataLoaderFactory._make_aligned_loaders(
                datasets=[ds_a_train, ds_b_train, ds_server_train],
                batch_size=bs,
                shuffle=True,
                seed=seed,
                num_workers=num_workers,
                drop_last=True,
            )
        )
        val_loaders = tuple(
            VFLDataLoaderFactory._make_aligned_loaders(
                datasets=[ds_a_val, ds_b_val, ds_server_val],
                batch_size=bs,
                shuffle=False,
                seed=seed,
                num_workers=num_workers,
                drop_last=False,
            )
        )
        return train_loaders, val_loaders

    # ------------------------------------------------------------------
    # Core alignment mechanism
    # ------------------------------------------------------------------

    @staticmethod
    def _make_aligned_loaders(
        datasets: list[Dataset],
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_workers: int = 0,
        drop_last: bool = True,
    ) -> list[DataLoader]:
        """
        Create one DataLoader per dataset, all sharing the same RNG seed
        so their shuffle orders are identical.

        Parameters
        ----------
        datasets : list[Dataset]
            Must all have the same __len__. Assertion is enforced.
        batch_size : int
        shuffle : bool
            True for training, False for evaluation.
        seed : int
            Seed for the shared torch.Generator.
        num_workers : int
            WARNING: for num_workers > 0 you must set worker_init_fn to
            re-seed each worker identically, or alignment breaks.
        drop_last : bool
            Drop the last incomplete batch. Recommended True to avoid
            single-sample batches that break BatchNorm1d.

        Returns
        -------
        list[DataLoader]
            One DataLoader per input dataset, guaranteed to be aligned.
        """
        lengths = [len(ds) for ds in datasets]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All datasets must have equal length for VFL alignment. "
                f"Got lengths: {lengths}"
            )

        loaders = []
        for ds in datasets:
            # Each DataLoader gets its own generator initialised with the same seed.
            # Because Generator state is independent per object but starts from the
            # same seed, every DataLoader produces the same shuffle permutation.
            g = torch.Generator()
            g.manual_seed(seed)

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=g if shuffle else None,
                sampler=None,           # shuffle=True uses RandomSampler internally
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=False,       # safe default; set True for GPU training
            )
            loaders.append(loader)

        return loaders
