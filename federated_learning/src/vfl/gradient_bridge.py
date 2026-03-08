"""
VFL Cut-Layer Gradient Bridge.

Implements the three-step gradient protocol that enables passive parties
(bottom models) to receive training signal from the active party (top model)
without exposing raw labels or top-model parameters.

Protocol overview
-----------------
Step 1  [Passive party, before 'transmission']:
    local_emb = bottom_model(x)          # connected to bottom model params via grad_fn
    sent_emb  = detach_for_transmission(local_emb)   # NEW leaf tensor, no grad_fn

Step 2  [Active party]:
    logits = top_model(sent_emb_a, sent_emb_b)
    loss   = criterion(logits, labels)
    optimizer_top.zero_grad()
    loss.backward()                      # populates sent_emb.grad = dL/d_sent_emb
    optimizer_top.step()
    grad = extract_gradient(sent_emb)    # read dL/d_sent_emb

Step 3  [Passive party, after receiving grad]:
    optimizer_bottom.zero_grad()
    apply_gradient(local_emb, grad)      # backprop through grad_fn into bottom params
    optimizer_bottom.step()

Why .detach().requires_grad_(True)?
------------------------------------
  - .detach()           : severs the tensor from the computation graph so
                          loss.backward() does NOT reach bottom model params
                          directly (simulating the network boundary).
  - .requires_grad_(True): registers the detached tensor as a NEW leaf node
                            so PyTorch accumulates gradients on it, giving us
                            dL/d_sent_emb after loss.backward().
  - local_emb.backward(grad): uses the saved grad_fn on the original local_emb
                               to propagate dL/d_sent_emb further back into the
                               bottom model parameters.

Privacy Model
-------------
This module sits at the privacy boundary of the VFL system.

CURRENT — Structural / Partitioned Privacy:
  - Raw features are NEVER transmitted. Only the 128-dim embedding vector
    crosses the simulated network boundary.
  - Labels are held exclusively by the active party (server).
    Passive parties receive only dL/d_embedding — not logits, loss values,
    or top-model weights.
  - Cross-party gradient isolation: Party A receives only grad_a = dL/d_emb_a;
    it never sees grad_b. Confirmed by the test suite (test_no_cross_party_leakage).

LIMITATION — No formal Differential Privacy:
  Without noise injection, an adversary who intercepts the gradient stream
  could attempt a gradient inversion attack (Zhu et al., NeurIPS 2019) to
  approximately reconstruct the passive party's input features from the
  gradients. This is the primary remaining privacy risk in the current system.

IMPLEMENTED — Opacus Differential Privacy:
  detach_for_transmission(local_emb, clip_norm=C, noise_multiplier=σ)
  clips each embedding row to L2 norm ≤ C, then adds Gaussian noise N(0,(σC)²).
  Privacy budget (ε, δ) is tracked by DPBudgetAccountant (Opacus RDPAccountant).
  Enable via VFLConfig.dp.enabled=True; see DPConfig in configs/vfl_config.py.
"""
from __future__ import annotations

import torch


class EmbeddingGradientBridge:
    """Static helper methods implementing the VFL cut-layer gradient protocol."""

    @staticmethod
    def detach_for_transmission(
        local_emb: torch.Tensor,
        clip_norm: float | None = None,
        noise_multiplier: float | None = None,
    ) -> torch.Tensor:
        """
        Prepare a local embedding for 'transmission' to the active party.

        Creates a new leaf tensor with the same values as ``local_emb`` but
        detached from the computation graph, while still accumulating gradients.

        When ``clip_norm`` is supplied (Epic 3 — DP mode), the embedding is
        additionally clipped and noised before transmission:

          1. Per-sample L2-norm clipping: each row is scaled so its norm ≤ C.
             This bounds the *sensitivity* of the transmitted value.
          2. Gaussian noise N(0, (σ·C)²) is added to every element.
             This masks individual sample information.

        Parameters
        ----------
        local_emb : Tensor
            Raw output of the passive party's bottom model. Must have
            ``requires_grad=True`` (i.e. bottom model must be in train mode).
        clip_norm : float or None
            C — maximum L2 norm allowed per embedding row.
            Pass ``None`` (default) to skip clipping and noise (no-DP mode).
        noise_multiplier : float or None
            σ — noise scale relative to clip_norm.
            noise_std = noise_multiplier * clip_norm.
            Only used when ``clip_norm`` is also provided.

        Returns
        -------
        sent_emb : Tensor
            Detached leaf tensor. Shape identical to ``local_emb``.
            ``sent_emb.grad_fn`` is None; ``sent_emb.requires_grad`` is True.
            In DP mode, values are clipped and noised.
        """
        sent = local_emb.detach()

        if clip_norm is not None:
            # Per-sample L2 norm clipping: divide each row by max(1, ||row||/C)
            norms = sent.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = (clip_norm / norms).clamp(max=1.0)
            sent = sent * scale

            if noise_multiplier is not None:
                # Gaussian mechanism: noise_std = σ * C
                noise_std = noise_multiplier * clip_norm
                sent = sent + torch.randn_like(sent) * noise_std

        return sent.requires_grad_(True)

    @staticmethod
    def extract_gradient(sent_emb: torch.Tensor) -> torch.Tensor:
        """
        Read the gradient that the active party accumulated on the sent embedding.

        Must be called AFTER ``loss.backward()`` on the active party side and
        BEFORE ``optimizer_top.zero_grad()`` (or any operation that clears grads).

        Parameters
        ----------
        sent_emb : Tensor
            The same tensor returned by ``detach_for_transmission``, after the
            active party has called ``loss.backward()``.

        Returns
        -------
        grad : Tensor
            Clone of ``sent_emb.grad``. Cloning ensures the gradient is not
            invalidated by subsequent ``zero_grad()`` calls.

        Raises
        ------
        RuntimeError
            If ``sent_emb.grad`` is None (backward was not called, or the
            tensor was not used in the forward pass).
        """
        if sent_emb.grad is None:
            raise RuntimeError(
                "sent_emb.grad is None. Ensure that:\n"
                "  1. loss.backward() was called on the active party.\n"
                "  2. sent_emb was actually used in the top-model forward pass.\n"
                "  3. extract_gradient() is called before any zero_grad()."
            )
        return sent_emb.grad.clone()

    @staticmethod
    def apply_gradient(
        local_emb: torch.Tensor,
        received_grad: torch.Tensor,
        retain_graph: bool = False,
    ) -> None:
        """
        Backpropagate the active party's gradient into the passive party's
        bottom model parameters.

        Parameters
        ----------
        local_emb : Tensor
            The original (pre-detach) output of the bottom model. Must still
            be in scope (not garbage-collected). Its ``grad_fn`` connects it
            back to the bottom model weights.
        received_grad : Tensor
            The gradient tensor returned by ``extract_gradient`` on the active
            party. Represents dL/d(embedding).
        retain_graph : bool
            Set True only if the same bottom model forward pass is reused in
            multiple backward calls (not needed for standard single-pass VFL).

        Notes
        -----
        Call ``optimizer_bottom.zero_grad()`` BEFORE this method, not before
        ``loss.backward()`` on the active party side.
        """
        local_emb.backward(gradient=received_grad, retain_graph=retain_graph)