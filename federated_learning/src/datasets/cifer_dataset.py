"""
CiferAI Fraud Detection Vertical Split Dataset.

Loads CiferAI/Cifer-Fraud-Detection-Dataset-AF from HuggingFace and
partitions features vertically across two parties:

  Party A (identity/temporal):   step, nameOrig, nameDest
  Party B (transaction/behavioral): type, amount, oldbalanceOrg,
                                     newbalanceOrig, oldbalanceDest,
                                     newbalanceDest, isFlaggedFraud
  Server (active party):          isFraud  (labels only)

Encoding strategy:
  - 'type' -> LabelEncoder (6 unique transaction types)
  - 'nameOrig' / 'nameDest' -> frequency encoding (millions of unique account IDs; one-hot would blow up memory)
  - numeric columns -> StandardScaler (fit on train, transform on test)

Use CiferVerticalDataset.build_aligned_pair() as the primary entry point.
It returns three Dataset objects that are strictly index-aligned.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from configs.vfl_config import CiferConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_raw_dataframe(config: CiferConfig) -> pd.DataFrame:
    """Stream a subsample (or full dataset) from HuggingFace as a DataFrame.

    When max_samples is set, streams only the required rows (~30 MB) instead
    of downloading all 14 HuggingFace parts (~1.8 GB).  The shuffle buffer
    gives a random sample without loading the full dataset into memory.
    """
    if config.max_samples is not None:
        hf_dataset = load_dataset(config.dataset_id, split="train", streaming=True)
        hf_dataset = hf_dataset.shuffle(
            seed=config.random_state,
            buffer_size=min(10_000, config.max_samples),
        )
        hf_dataset = hf_dataset.take(config.max_samples)
        df = pd.DataFrame(hf_dataset)
    else:
        # User opted into the full 6.3 M-row dataset — download all parts
        hf_dataset = load_dataset(config.dataset_id, split="train")
        df = hf_dataset.to_pandas()

    return df.reset_index(drop=True)


def _frequency_encode(
    series: pd.Series,
    freq_map: Optional[dict] = None,
    fit: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Replace each category with its relative frequency in the training set.
    Unknown values (test set unseen categories) → 0.0.

    Returns (encoded_array, freq_map)
    """
    if fit:
        freq_map = (series.value_counts() / len(series)).to_dict()
    assert freq_map is not None
    encoded = series.map(freq_map).fillna(0.0).to_numpy(dtype=np.float32)
    return encoded, freq_map


def _build_encoders(
    df: pd.DataFrame,
    config: CiferConfig,
) -> tuple[LabelEncoder, dict, dict]:
    """
    Fit encoders on a training DataFrame.

    Returns
    -------
    (type_encoder, nameOrig_freq_map, nameDest_freq_map)
    """
    type_enc = LabelEncoder().fit(df["type"].astype(str))
    _, orig_freq = _frequency_encode(df["nameOrig"].astype(str), fit=True)
    _, dest_freq = _frequency_encode(df["nameDest"].astype(str), fit=True)
    return type_enc, orig_freq, dest_freq


def _encode_dataframe(
    df: pd.DataFrame,
    type_enc: LabelEncoder,
    orig_freq: dict,
    dest_freq: dict,
) -> pd.DataFrame:
    """
    Apply fitted encoders in-place and return a fully numeric DataFrame.
    Unknown 'type' values fallback to 0 (first encoded class).
    """
    df = df.copy()
    known_classes = set(type_enc.classes_)
    df["type"] = df["type"].astype(str).apply(
        lambda v: v if v in known_classes else type_enc.classes_[0]
    )
    df["type"] = type_enc.transform(df["type"]).astype(np.float32)

    df["nameOrig"], _ = _frequency_encode(df["nameOrig"].astype(str), freq_map=orig_freq, fit=False)
    df["nameDest"], _ = _frequency_encode(df["nameDest"].astype(str), freq_map=dest_freq, fit=False)
    return df


def _fit_scaler(df: pd.DataFrame, columns: list[str]) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[columns].to_numpy(dtype=np.float32))
    return scaler, arr


def _oversample_minority(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    labels: np.ndarray,
    target_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly repeat minority-class (fraud, label=1) rows until the positive
    class makes up ``target_ratio`` of the combined training set.

    All three arrays are shuffled together so fraud samples are interleaved
    throughout the epoch rather than clumped at the end.

    Parameters
    ----------
    target_ratio : float
        Desired minority/(minority+majority) ratio after oversampling.
        E.g. 0.1 → 10 % of training rows are fraud.

    Returns the three arrays with the same dtype and column ordering.
    """
    rng = np.random.default_rng(random_state)

    pos_idx = np.where(labels == 1)[0]
    n_pos_orig = len(pos_idx)
    n_neg = int((labels == 0).sum())

    # How many positive samples are needed to hit the target ratio?
    n_pos_target = int(n_neg * target_ratio / (1.0 - target_ratio))
    if n_pos_target <= n_pos_orig:
        return arr_a, arr_b, labels     # already at or above target ratio

    # Randomly sample extra fraud indices (with replacement)
    extra_idx = rng.choice(pos_idx, size=n_pos_target - n_pos_orig, replace=True)

    # Build the combined index and shuffle so fraud rows are spread evenly
    all_idx = np.concatenate([np.arange(len(labels)), extra_idx])
    rng.shuffle(all_idx)

    print(
        f"[CiferAI] Oversampled minority class: {n_pos_orig} → {n_pos_target} fraud rows "
        f"({n_pos_target / (n_pos_target + n_neg) * 100:.1f}% of training set)"
    )
    return arr_a[all_idx], arr_b[all_idx], labels[all_idx]


def _apply_scaler(scaler: StandardScaler, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return scaler.transform(df[columns].to_numpy(dtype=np.float32))


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CiferVerticalDataset(Dataset):
    """
    PyTorch Dataset for one party's view of the CiferAI fraud dataset.

    Do not instantiate directly — use ``build_aligned_pair`` instead.

    Parameters
    ----------
    features : np.ndarray
        Encoded and scaled feature matrix for this party, or None (server).
    labels : np.ndarray | None
        Integer fraud labels, only populated for party='server'.
    party : str
        'A', 'B', or 'server' (informational only after construction).
    """

    def __init__(
        self,
        features: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        party: Literal["A", "B", "server"],
    ) -> None:
        self.party = party
        self._features = features   # (N, D) float32 or None
        self._labels = labels       # (N,) int64 or None

        if party in ("A", "B"):
            assert features is not None, "Party A/B must receive feature array"
            self._n = len(features)
        else:
            assert labels is not None, "Server party must receive label array"
            self._n = len(labels)

    # ------------------------------------------------------------------
    # Factory — the only public way to create aligned split datasets
    # ------------------------------------------------------------------

    @classmethod
    def build_aligned_pair(
        cls,
        config: CiferConfig,
    ) -> tuple[
        tuple["CiferVerticalDataset", "CiferVerticalDataset", "CiferVerticalDataset"],
        tuple["CiferVerticalDataset", "CiferVerticalDataset", "CiferVerticalDataset"],
    ]:
        """
        Load, encode, scale, and split the CiferAI dataset.

        Encoders and scalers are fit on the training split only; the val split
        is transformed with the same fitted objects to prevent data leakage.

        Returns
        -------
        (train_datasets, val_datasets)
            Each is a 3-tuple (ds_a, ds_b, ds_server) strictly index-aligned.
        """
        print("[CiferAI] Loading dataset from HuggingFace... (first run may take a moment)")
        df = _load_raw_dataframe(config)

        # Train / val split (stratified on label)
        df_train, df_val = train_test_split(
            df,
            train_size=config.train_split,
            stratify=df[config.label_column],
            random_state=config.random_state,
        )
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        # Fit encoders on train only — then apply to both splits
        type_enc, orig_freq, dest_freq = _build_encoders(df_train, config)
        df_train_enc = _encode_dataframe(df_train, type_enc, orig_freq, dest_freq)
        df_val_enc   = _encode_dataframe(df_val,   type_enc, orig_freq, dest_freq)

        a_cols = config.party_a_columns    # ["step", "nameOrig", "nameDest"]
        b_cols = config.party_b_columns    # ["type", "amount", ...]

        # Fit scalers on train; apply to both splits
        scaler_a, arr_a_train = _fit_scaler(df_train_enc, a_cols)
        scaler_b, arr_b_train = _fit_scaler(df_train_enc, b_cols)
        arr_a_val = _apply_scaler(scaler_a, df_val_enc, a_cols)
        arr_b_val = _apply_scaler(scaler_b, df_val_enc, b_cols)

        labels_train = df_train_enc[config.label_column].to_numpy(dtype=np.int64)
        labels_val   = df_val_enc[config.label_column].to_numpy(dtype=np.int64)

        print(
            f"[CiferAI] Loaded {len(df_train)} train | {len(df_val)} val samples | "
            f"fraud rate: {labels_train.mean():.4f} | "
            f"Party A features: {arr_a_train.shape[1]}, "
            f"Party B features: {arr_b_train.shape[1]}"
        )

        # C1: oversample the minority fraud class so the model receives enough
        # positive signal. Oversampling is applied to training data only;
        # val data is never modified (real-world distribution must be preserved).
        if getattr(config, "oversample_minority", False):
            arr_a_train, arr_b_train, labels_train = _oversample_minority(
                arr_a_train, arr_b_train, labels_train,
                target_ratio=config.oversample_target_ratio,
                random_state=config.random_state,
            )

        train_datasets = (
            cls(features=arr_a_train, labels=None, party="A"),
            cls(features=arr_b_train, labels=None, party="B"),
            cls(features=None, labels=labels_train, party="server"),
        )
        val_datasets = (
            cls(features=arr_a_val, labels=None, party="A"),
            cls(features=arr_b_val, labels=None, party="B"),
            cls(features=None, labels=labels_val, party="server"),
        )
        return train_datasets, val_datasets

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Party A/B : feature vector, float32
            server    : label scalar, torch.long
        """
        if self.party in ("A", "B"):
            return torch.from_numpy(self._features[idx])            # float32
        else:
            return torch.tensor(self._labels[idx], dtype=torch.long)

    @property
    def feature_dim(self) -> int:
        """Number of features for this party. Needed to instantiate bottom model."""
        if self._features is None:
            raise AttributeError("Server party has no features")
        return self._features.shape[1]
