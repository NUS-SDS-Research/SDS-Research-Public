"""
Privacy budget accountant for VFL embedding-level differential privacy.

Wraps Opacus RDPAccountant to track (ε, δ) as noise is applied to embeddings
during training. One "step" corresponds to one batch of noise applications for
one party; two steps are recorded per training batch (one per passive party).

Usage
-----
    from src.vfl.dp_accountant import DPBudgetAccountant

    accountant = DPBudgetAccountant(noise_multiplier=1.0, delta=1e-5)

    # Inside the training loop, after noise is applied:
    accountant.step(sample_rate=batch_size / n_train)   # party A
    accountant.step(sample_rate=batch_size / n_train)   # party B

    # At the end of training:
    eps = accountant.get_epsilon()
    print(f"Privacy budget: ε={eps:.4f} at δ=1e-5")
"""
from __future__ import annotations

from opacus.accountants import RDPAccountant


class DPBudgetAccountant:
    """
    Tracks (ε, δ) privacy budget using Rényi Differential Privacy composition.

    Parameters
    ----------
    noise_multiplier : float
        σ in the Gaussian mechanism. noise_std = noise_multiplier * clip_norm.
        Higher σ → more noise → lower ε (stronger privacy, worse utility).
    delta : float
        δ for (ε, δ)-DP. Typically 1e-5 (probability of privacy failure).
    """

    def __init__(self, noise_multiplier: float, delta: float) -> None:
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self._accountant = RDPAccountant()
        self._steps = 0

    def step(self, *, sample_rate: float) -> None:
        """
        Record one noise application (one batch for one party).

        Parameters
        ----------
        sample_rate : float
            Fraction of training data used in this batch (batch_size / n_train).
            Used by the RDP accountant for subsampling amplification.
        """
        self._accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=sample_rate,
        )
        self._steps += 1

    def get_epsilon(self) -> float:
        """
        Return the current ε for the configured δ.

        Returns ``float('inf')`` if no steps have been recorded yet.
        """
        if self._steps == 0:
            return float("inf")
        return self._accountant.get_epsilon(delta=self.delta)

    @property
    def steps(self) -> int:
        """Total number of noise-application steps recorded."""
        return self._steps
