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
    """Download (or use cache) and return a stratified subsample as DataFrame."""
    hf_dataset = load_dataset(config.dataset_id, split="train")

    if config.max_samples is not None and config.max_samples < len(hf_dataset):
        # Stratified subsample preserving fraud ratio
        df_full = hf_dataset.to_pandas()
        df, _ = train_test_split(
            df_full,
            train_size=config.max_samples,
            stratify=df_full[config.label_column],
            random_state=config.random_state,
        )
        df = df.reset_index(drop=True)
    else:
        df = hf_dataset.to_pandas()

    return df


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
    ) -> tuple["CiferVerticalDataset", "CiferVerticalDataset", "CiferVerticalDataset"]:
        """
        Load, encode, scale, and split the CiferAI dataset.

        Returns
        -------
        (train_a, train_b, train_server)
            Three strictly index-aligned Dataset objects for training.
            (Test datasets can be built similarly by passing a separate config.)
        """
        print("[CiferAI] Loading dataset from HuggingFace... (first run may take a moment)")
        df = _load_raw_dataframe(config)

        # Train / test split (stratified on label)
        df_train, _df_test = train_test_split(
            df,
            train_size=config.train_split,
            stratify=df[config.label_column],
            random_state=config.random_state,
        )
        df_train = df_train.reset_index(drop=True)

        # Fit encoders on train
        type_enc, orig_freq, dest_freq = _build_encoders(df_train, config)

        # Encode both splits
        df_train_enc = _encode_dataframe(df_train, type_enc, orig_freq, dest_freq)

        # Extract party column arrays
        a_cols = config.party_a_columns    # ["step", "nameOrig", "nameDest"]
        b_cols = config.party_b_columns    # ["type", "amount", ...]

        # Fit scalers on train numeric columns (nameOrig/nameDest already float)
        scaler_a, arr_a_train = _fit_scaler(df_train_enc, a_cols)
        scaler_b, arr_b_train = _fit_scaler(df_train_enc, b_cols)

        labels_train = df_train_enc[config.label_column].to_numpy(dtype=np.int64)

        print(
            f"[CiferAI] Loaded {len(df_train)} train samples | "
            f"fraud rate: {labels_train.mean():.4f} | "
            f"Party A features: {arr_a_train.shape[1]}, "
            f"Party B features: {arr_b_train.shape[1]}"
        )

        return (
            cls(features=arr_a_train, labels=None, party="A"),
            cls(features=arr_b_train, labels=None, party="B"),
            cls(features=None, labels=labels_train, party="server"),
        )

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
