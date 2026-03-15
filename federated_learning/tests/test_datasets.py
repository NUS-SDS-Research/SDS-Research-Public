"""
Tests for Dual-Modality Vertical Split Datasets.

All tests use synthetic data or mocked MNIST.
The MNIST tests that require actual data are marked with @pytest.mark.slow
and skipped by default (run with: pytest -m slow).
"""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    """Simple dataset of sequential integers — used for alignment tests."""
    def __init__(self, n: int):
        self._data = torch.arange(n, dtype=torch.float32)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


# ---------------------------------------------------------------------------
# DataLoader alignment tests (no downloads required)
# ---------------------------------------------------------------------------

class TestDataLoaderAlignment:

    def test_aligned_loaders_same_shuffle_order(self):
        """
        Three loaders built with the same seed must produce identical index orders.
        Verified by checking that corresponding batches are numerically equal
        when the dataset maps index → value.
        """
        from src.datasets.dataloader_factory import VFLDataLoaderFactory

        n = 200
        ds_a = _SyntheticDataset(n)
        ds_b = _SyntheticDataset(n)
        ds_s = _SyntheticDataset(n)

        loaders = VFLDataLoaderFactory._make_aligned_loaders(
            datasets=[ds_a, ds_b, ds_s],
            batch_size=32,
            shuffle=True,
            seed=42,
            num_workers=0,
        )
        loader_a, loader_b, loader_s = loaders

        # Collect full epoch
        batches_a = [b for b in loader_a]
        batches_b = [b for b in loader_b]
        batches_s = [b for b in loader_s]

        assert len(batches_a) == len(batches_b) == len(batches_s)

        for ba, bb, bs in zip(batches_a, batches_b, batches_s):
            assert torch.allclose(ba, bb), "loader_a and loader_b batches differ"
            assert torch.allclose(ba, bs), "loader_a and loader_s batches differ"

    def test_mismatched_lengths_raises(self):
        """DataLoaderFactory must raise ValueError if dataset lengths differ."""
        from src.datasets.dataloader_factory import VFLDataLoaderFactory

        ds_a = _SyntheticDataset(100)
        ds_b = _SyntheticDataset(99)   # different length

        with pytest.raises(ValueError, match="equal length"):
            VFLDataLoaderFactory._make_aligned_loaders(
                datasets=[ds_a, ds_b],
                batch_size=16,
                shuffle=False,
                seed=0,
            )

    def test_drop_last_prevents_small_batches(self):
        """With drop_last=True no batch should be smaller than batch_size."""
        from src.datasets.dataloader_factory import VFLDataLoaderFactory

        n = 105        # not divisible by batch_size=32 → last batch would be 9
        ds = _SyntheticDataset(n)

        loaders = VFLDataLoaderFactory._make_aligned_loaders(
            datasets=[ds],
            batch_size=32,
            shuffle=False,
            seed=0,
            drop_last=True,
        )
        for batch in loaders[0]:
            assert batch.size(0) == 32, f"Expected batch size 32, got {batch.size(0)}"


# ---------------------------------------------------------------------------
# MNIST Dataset tests (using unittest.mock)
# ---------------------------------------------------------------------------

class TestMNISTVerticalDataset:

    def _make_fake_mnist(self, n: int = 50):
        """Return a mock object that behaves like torchvision.datasets.MNIST."""
        import unittest.mock as mock

        images = torch.rand(n, 1, 28, 28)
        labels = torch.randint(0, 10, (n,))

        fake = mock.MagicMock()
        fake.__len__ = mock.Mock(return_value=n)
        fake.__getitem__ = mock.Mock(side_effect=lambda i: (images[i], labels[i].item()))
        return fake

    def test_party_a_shape(self):
        from configs.vfl_config import MNISTConfig
        from src.datasets.mnist_dataset import MNISTVerticalDataset

        cfg = MNISTConfig(split_row=14, download=False)
        fake_mnist = self._make_fake_mnist()

        ds_a = MNISTVerticalDataset(cfg, party="A", base_dataset=fake_mnist)
        sample = ds_a[0]

        assert sample.shape == (1, 14, 28), f"Expected (1,14,28), got {sample.shape}"
        assert sample.dtype == torch.float32

    def test_party_b_shape(self):
        from configs.vfl_config import MNISTConfig
        from src.datasets.mnist_dataset import MNISTVerticalDataset

        cfg = MNISTConfig(split_row=14, download=False)
        fake_mnist = self._make_fake_mnist()

        ds_b = MNISTVerticalDataset(cfg, party="B", base_dataset=fake_mnist)
        sample = ds_b[0]

        assert sample.shape == (1, 14, 28), f"Expected (1,14,28), got {sample.shape}"

    def test_server_returns_label_scalar(self):
        from configs.vfl_config import MNISTConfig
        from src.datasets.mnist_dataset import MNISTVerticalDataset

        cfg = MNISTConfig(split_row=14, download=False)
        fake_mnist = self._make_fake_mnist()

        ds_s = MNISTVerticalDataset(cfg, party="server", base_dataset=fake_mnist)
        label = ds_s[0]

        assert label.ndim == 0, "Server label should be a scalar tensor"
        assert label.dtype == torch.long

    def test_build_aligned_pair_lengths_equal(self):
        from configs.vfl_config import MNISTConfig
        from src.datasets.mnist_dataset import MNISTVerticalDataset

        cfg = MNISTConfig(split_row=14, download=False)
        fake_mnist = self._make_fake_mnist(n=60)

        ds_a = MNISTVerticalDataset(cfg, party="A", base_dataset=fake_mnist)
        ds_b = MNISTVerticalDataset(cfg, party="B", base_dataset=fake_mnist)
        ds_s = MNISTVerticalDataset(cfg, party="server", base_dataset=fake_mnist)

        assert len(ds_a) == len(ds_b) == len(ds_s) == 60

    def test_no_label_leakage_in_party_views(self):
        """
        Party A and B should return float tensors (features), not long scalars.
        If they accidentally returned labels, a classification model would trivially overfit.
        """
        from configs.vfl_config import MNISTConfig
        from src.datasets.mnist_dataset import MNISTVerticalDataset

        cfg = MNISTConfig(split_row=14, download=False)
        fake_mnist = self._make_fake_mnist()

        for party in ("A", "B"):
            ds = MNISTVerticalDataset(cfg, party=party, base_dataset=fake_mnist)
            sample = ds[0]
            assert sample.dtype != torch.long, (
                f"Party {party} returned a long tensor — possible label leakage"
            )
            assert sample.ndim > 0, f"Party {party} returned a scalar"

    def test_invalid_party_raises(self):
        from configs.vfl_config import MNISTConfig
        from src.datasets.mnist_dataset import MNISTVerticalDataset

        cfg = MNISTConfig(download=False)
        fake_mnist = self._make_fake_mnist()

        with pytest.raises(ValueError):
            MNISTVerticalDataset(cfg, party="C", base_dataset=fake_mnist)


# ---------------------------------------------------------------------------
# CiferAI Dataset tests (no HuggingFace download — test components in isolation)
# ---------------------------------------------------------------------------

class TestCiferDatasetComponents:
    """
    Tests for encoding/scaling helpers in cifer_dataset.py.
    """

    def test_frequency_encoding_fit_and_transform(self):
        import pandas as pd
        from src.datasets.cifer_dataset import _frequency_encode

        series = pd.Series(["A", "A", "B", "C", "A", "B"])
        encoded, freq_map = _frequency_encode(series, fit=True)

        # A appears 3/6 = 0.5
        assert abs(freq_map["A"] - 0.5) < 1e-6
        # B appears 2/6 ≈ 0.333
        assert abs(freq_map["B"] - 1/3) < 1e-6
        assert encoded.shape == (6,)

    def test_frequency_encoding_unseen_category_is_zero(self):
        import pandas as pd
        from src.datasets.cifer_dataset import _frequency_encode

        train = pd.Series(["X", "Y", "X"])
        _, freq_map = _frequency_encode(train, fit=True)

        test = pd.Series(["X", "Z"])          # "Z" is unseen
        encoded, _ = _frequency_encode(test, freq_map=freq_map, fit=False)

        assert encoded[0] > 0            # "X" has known freq
        assert encoded[1] == 0.0         # "Z" maps to 0.0

    def test_cifer_dataset_getitem_returns_float_tensor(self):
        """CiferVerticalDataset (party A or B) must return float32 tensors."""
        from src.datasets.cifer_dataset import CiferVerticalDataset
        import numpy as np

        features = np.random.randn(50, 3).astype(np.float32)
        ds = CiferVerticalDataset(features=features, labels=None, party="A")

        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32
        assert sample.shape == (3,)

    def test_cifer_server_dataset_returns_long_tensor(self):
        """Server dataset must return integer class labels."""
        from src.datasets.cifer_dataset import CiferVerticalDataset
        import numpy as np

        labels = np.array([0, 1, 0, 0, 1], dtype=np.int64)
        ds = CiferVerticalDataset(features=None, labels=labels, party="server")

        label = ds[1]
        assert label.dtype == torch.long
        assert label.item() == 1

    def test_cifer_feature_dim_property(self):
        from src.datasets.cifer_dataset import CiferVerticalDataset
        import numpy as np

        features = np.random.randn(20, 7).astype(np.float32)
        ds = CiferVerticalDataset(features=features, labels=None, party="B")
        assert ds.feature_dim == 7