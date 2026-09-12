"""
MNIST Vertical Spatial Split Dataset.

Party A receives the top half of each image:  tensor[:, :14, :]  -> (1, 14, 28)
Party B receives the bottom half:             tensor[:, 14:, :]  -> (1, 14, 28)
Server ('active party') holds the labels.

All three Dataset objects share a single underlying torchvision.datasets.MNIST
instance so the dataset is downloaded only once and memory is not tripled.
"""
from __future__ import annotations

import sys
import os

# Allow running from project root with `python scripts/...`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Literal, Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from configs.vfl_config import MNISTConfig


class MNISTVerticalDataset(Dataset):
    """
    Vertical spatial split of MNIST for two-party VFL.

    Parameters
    ----------
    config : MNISTConfig
        Dataset configuration (data_dir, download, split_row).
    party : {'A', 'B', 'server'}
        Which party this dataset instance represents.
    train : bool
        Use training split if True, test split if False.
    base_dataset : torchvision.datasets.MNIST, optional
        Pre-loaded MNIST dataset to share across party instances.
        If None, a new instance is created (triggers download if needed).
    """

    def __init__(
        self,
        config: MNISTConfig,
        party: Literal["A", "B", "server"],
        train: bool = True,
        base_dataset: Optional[datasets.MNIST] = None,
    ) -> None:
        if party not in ("A", "B", "server"):
            raise ValueError(f"party must be 'A', 'B', or 'server'; got {party!r}")

        self.party = party
        self.split_row = config.split_row
        self._transform = transforms.ToTensor()

        if base_dataset is not None:
            self._base = base_dataset
        else:
            self._base = datasets.MNIST(
                root=config.data_dir,
                train=train,
                download=config.download,
                transform=self._transform,
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_aligned_pair(
        cls,
        config: MNISTConfig,
        train: bool = True,
    ) -> tuple["MNISTVerticalDataset", "MNISTVerticalDataset", "MNISTVerticalDataset"]:
        """
        Build three index-aligned Dataset objects from a single MNIST load.

        Returns
        -------
        (dataset_a, dataset_b, dataset_server)
            All three share `base_dataset` so they are perfectly aligned by index.
        """
        base = datasets.MNIST(
            root=config.data_dir,
            train=train,
            download=config.download,
            transform=transforms.ToTensor(),
        )
        dataset_a = cls(config, party="A", train=train, base_dataset=base)
        dataset_b = cls(config, party="B", train=train, base_dataset=base)
        dataset_server = cls(config, party="server", train=train, base_dataset=base)
        return dataset_a, dataset_b, dataset_server

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Party A : image slice (1, 14, 28), float32
            Party B : image slice (1, 14, 28), float32
            server  : label scalar, torch.long
        """
        image, label = self._base[idx]  # image: (1, 28, 28), label: int

        if self.party == "A":
            return image[:, : self.split_row, :]          # (1, 14, 28)
        elif self.party == "B":
            return image[:, self.split_row :, :]          # (1, 14, 28)
        else:  # server
            return torch.tensor(label, dtype=torch.long)
