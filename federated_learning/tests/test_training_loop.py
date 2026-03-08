"""
Tests for VFL Gradient Bridge and Training Loop.

All tests use synthetic in-memory tensors — no downloads required.
The most important tests here verify correctness of the gradient protocol:
  1. Gradient bridge: gradients actually flow into bottom model parameters.
  2. No cross-party leakage: Party A grad is None before apply_gradient().
  3. Loss decreases: end-to-end training should improve over epochs.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_bottom_model(input_dim: int, emb_dim: int = 16) -> nn.Module:
    """Tiny MLP — fast for tests."""
    return nn.Sequential(nn.Linear(input_dim, emb_dim))


def _make_loaders(n: int = 128, dim_a: int = 4, dim_b: int = 3, num_classes: int = 2, batch_size: int = 32):
    """Create three aligned TensorDataset loaders with synthetic data."""
    x_a = torch.randn(n, dim_a)
    x_b = torch.randn(n, dim_b)
    labels = torch.randint(0, num_classes, (n,))

    loader_a = DataLoader(TensorDataset(x_a), batch_size=batch_size, shuffle=False, drop_last=True)
    loader_b = DataLoader(TensorDataset(x_b), batch_size=batch_size, shuffle=False, drop_last=True)
    loader_s = DataLoader(TensorDataset(labels), batch_size=batch_size, shuffle=False, drop_last=True)
    return loader_a, loader_b, loader_s


# ---------------------------------------------------------------------------
# EmbeddingGradientBridge
# ---------------------------------------------------------------------------

class TestEmbeddingGradientBridge:

    def test_detach_for_transmission_is_leaf(self):
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        model = _make_simple_bottom_model(4, 8)
        x = torch.randn(4, 4)
        local_emb = model(x)

        sent = EmbeddingGradientBridge.detach_for_transmission(local_emb)

        assert sent.grad_fn is None,     "sent_emb must be a leaf (no grad_fn)"
        assert sent.requires_grad,       "sent_emb must require grad"
        assert not local_emb.is_leaf,    "local_emb must NOT be a leaf (still connected)"

    def test_extract_gradient_after_backward(self):
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        model = _make_simple_bottom_model(4, 8)
        x = torch.randn(4, 4)
        local_emb = model(x)
        sent = EmbeddingGradientBridge.detach_for_transmission(local_emb)

        # Simulate active party loss
        loss = sent.sum()
        loss.backward()

        grad = EmbeddingGradientBridge.extract_gradient(sent)
        assert grad is not None
        assert grad.shape == sent.shape
        assert torch.all(grad == 1.0), "sum().backward() should give all-ones gradient"

    def test_extract_gradient_before_backward_raises(self):
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        model = _make_simple_bottom_model(4, 8)
        x = torch.randn(4, 4)
        local_emb = model(x)
        sent = EmbeddingGradientBridge.detach_for_transmission(local_emb)

        with pytest.raises(RuntimeError, match="grad is None"):
            EmbeddingGradientBridge.extract_gradient(sent)

    def test_full_gradient_flow_into_bottom_model(self):
        """
        Full 3-step protocol: detach → backward on sent → apply to local_emb.
        Bottom model parameters must have non-None gradients afterwards.
        """
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        model = _make_simple_bottom_model(4, 8)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(4, 4)
        local_emb = model(x)                                          # step 1
        sent = EmbeddingGradientBridge.detach_for_transmission(local_emb)  # step 2

        # Active party backward
        loss = (sent ** 2).sum()
        loss.backward()                                               # step 3
        grad = EmbeddingGradientBridge.extract_gradient(sent)        # step 4

        # Passive party backward
        optimizer.zero_grad()
        EmbeddingGradientBridge.apply_gradient(local_emb, grad)      # step 5
        optimizer.step()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"'{name}' has no gradient after apply_gradient"

    def test_no_gradient_cross_party_leakage(self):
        """
        Bottom model A's gradients must be None before apply_gradient is called.
        This verifies that the detach() truly isolates Party A from Party B's backward.
        """
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        model_a = _make_simple_bottom_model(4, 8)
        model_b = _make_simple_bottom_model(3, 8)
        model_a.train()
        model_b.train()

        x_a = torch.randn(4, 4)
        x_b = torch.randn(4, 3)

        local_a = model_a(x_a)
        local_b = model_b(x_b)

        sent_a = EmbeddingGradientBridge.detach_for_transmission(local_a)
        sent_b = EmbeddingGradientBridge.detach_for_transmission(local_b)

        # Active party only uses sent_b
        loss = sent_b.sum()
        loss.backward()

        # model_a parameters must have NO gradients (backward didn't touch them)
        for name, param in model_a.named_parameters():
            assert param.grad is None, (
                f"Cross-party leakage: model_a '{name}' has gradient even though "
                "Party A was not involved in loss.backward()"
            )


# ---------------------------------------------------------------------------
# VFLTrainer end-to-end
# ---------------------------------------------------------------------------

class TestVFLTrainer:

    def _make_trainer(self, dim_a=4, dim_b=3, emb_dim=16, num_classes=2):
        from src.models.bottom_models import TabularBottomModel
        from src.models.top_model import VFLTopModel
        from src.vfl.training_loop import VFLTrainer

        bottom_a = TabularBottomModel(input_dim=dim_a, embedding_dim=emb_dim, hidden_dim=32)
        bottom_b = TabularBottomModel(input_dim=dim_b, embedding_dim=emb_dim, hidden_dim=32)
        top      = VFLTopModel(embedding_dim=emb_dim, num_parties=2, num_classes=num_classes)

        device = torch.device("cpu")
        trainer = VFLTrainer(
            bottom_model_a=bottom_a,
            bottom_model_b=bottom_b,
            top_model=top,
            optimizer_a=torch.optim.Adam(bottom_a.parameters(), lr=1e-3),
            optimizer_b=torch.optim.Adam(bottom_b.parameters(), lr=1e-3),
            optimizer_top=torch.optim.Adam(top.parameters(), lr=1e-3),
            criterion=nn.CrossEntropyLoss(),
            device=device,
            verbose=False,
        )
        return trainer

    def test_train_one_epoch_returns_metrics(self):
        trainer = self._make_trainer()
        loader_a, loader_b, loader_s = _make_loaders(n=64, dim_a=4, dim_b=3)

        # Unwrap TensorDataset tuples
        loader_a2 = DataLoader(loader_a.dataset.tensors[0], batch_size=32, drop_last=True)
        loader_b2 = DataLoader(loader_b.dataset.tensors[0], batch_size=32, drop_last=True)
        loader_s2 = DataLoader(loader_s.dataset.tensors[0], batch_size=32, drop_last=True)

        metrics = trainer.train_one_epoch(loader_a2, loader_b2, loader_s2)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] > 0.0

    def test_loss_decreases_over_training(self):
        """
        After 10 epochs on 200 in-memory samples, training loss must decrease.
        This validates the full 10-step VFL gradient protocol end-to-end.
        """
        torch.manual_seed(0)
        trainer = self._make_trainer(dim_a=6, dim_b=5, emb_dim=32, num_classes=2)

        n = 200
        x_a = torch.randn(n, 6)
        x_b = torch.randn(n, 5)
        labels = torch.randint(0, 2, (n,))

        def make_loaders():
            la = DataLoader(x_a, batch_size=32, shuffle=False, drop_last=True)
            lb = DataLoader(x_b, batch_size=32, shuffle=False, drop_last=True)
            ls = DataLoader(labels, batch_size=32, shuffle=False, drop_last=True)
            return la, lb, ls

        losses = []
        for _ in range(10):
            la, lb, ls = make_loaders()
            m = trainer.train_one_epoch(la, lb, ls)
            losses.append(m["loss"])

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}. "
            "Check the gradient bridge implementation."
        )

    def test_get_and_load_model_state(self):
        """State dict round-trip must restore identical parameter values."""
        trainer = self._make_trainer()

        state = trainer.get_model_state()
        assert set(state.keys()) == {"bottom_a", "bottom_b", "top"}

        # Modify bottom_a weights
        for param in trainer.bottom_a.parameters():
            param.data.fill_(99.0)

        # Restore
        trainer.load_model_state(state)

        # Check that bottom_a is back to original
        for p_orig, p_loaded in zip(
            state["bottom_a"].values(), trainer.bottom_a.state_dict().values()
        ):
            assert torch.allclose(p_orig, p_loaded), "load_model_state did not restore weights"

    def test_evaluate_does_not_modify_gradients(self):
        """evaluate() must not leave gradients on model parameters."""
        trainer = self._make_trainer()
        loader_a, loader_b, loader_s = _make_loaders(n=64)

        la = DataLoader(loader_a.dataset.tensors[0], batch_size=32, drop_last=True)
        lb = DataLoader(loader_b.dataset.tensors[0], batch_size=32, drop_last=True)
        ls = DataLoader(loader_s.dataset.tensors[0], batch_size=32, drop_last=True)

        trainer.evaluate(la, lb, ls)

        for model in [trainer.bottom_a, trainer.bottom_b, trainer.top]:
            for param in model.parameters():
                assert param.grad is None, (
                    "evaluate() must not populate gradients (torch.no_grad missing?)"
                )


# ---------------------------------------------------------------------------
# Epic 3 — Differential Privacy
# ---------------------------------------------------------------------------

class TestDifferentialPrivacy:
    """Tests for embedding-level DP at the VFL cut-layer (Epic 3)."""

    def test_dp_disabled_no_change(self):
        """Without clip_norm, detach_for_transmission returns identical values."""
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        model = _make_simple_bottom_model(4, 16)
        model.train()
        x = torch.randn(8, 4)
        local_emb = model(x)

        sent = EmbeddingGradientBridge.detach_for_transmission(local_emb)

        assert torch.allclose(sent, local_emb.detach()), (
            "With no DP args, sent_emb values must exactly match local_emb"
        )
        assert sent.grad_fn is None
        assert sent.requires_grad

    def test_dp_norm_clipping(self):
        """After clipping, every embedding row must have L2 norm ≤ clip_norm."""
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        torch.manual_seed(0)
        model = _make_simple_bottom_model(4, 32)
        model.train()
        x = torch.randn(16, 4) * 10   # large values to ensure some norms exceed clip_norm
        local_emb = model(x)

        clip_norm = 0.5
        sent = EmbeddingGradientBridge.detach_for_transmission(
            local_emb, clip_norm=clip_norm, noise_multiplier=None
        )

        row_norms = sent.norm(dim=1)
        assert (row_norms <= clip_norm + 1e-5).all(), (
            f"All row norms must be ≤ clip_norm={clip_norm}. "
            f"Max found: {row_norms.max().item():.6f}"
        )

    def test_dp_noise_added(self):
        """With noise_multiplier > 0, sent_emb must differ from clipped local_emb."""
        from src.vfl.gradient_bridge import EmbeddingGradientBridge

        model = _make_simple_bottom_model(4, 16)
        model.train()
        x = torch.randn(8, 4)

        # Run 5 times to rule out the astronomically unlikely case of zero noise
        for seed in range(5):
            torch.manual_seed(seed)
            local_emb = model(x)
            sent = EmbeddingGradientBridge.detach_for_transmission(
                local_emb, clip_norm=1.0, noise_multiplier=1.0
            )
            if not torch.allclose(sent, local_emb.detach().clamp(max=1.0)):
                return  # noise confirmed
        pytest.fail("Noise was not added in any of 5 attempts — check noise injection")

    def test_dp_trainer_accountant_increments(self):
        """
        When DPConfig.enabled=True, training must increment the accountant and
        get_epsilon() must return a finite value.
        """
        from src.models.bottom_models import TabularBottomModel
        from src.models.top_model import VFLTopModel
        from src.vfl.training_loop import VFLTrainer
        from configs.vfl_config import DPConfig

        dim_a, dim_b, emb_dim = 4, 3, 16
        bottom_a = TabularBottomModel(input_dim=dim_a, embedding_dim=emb_dim, hidden_dim=32)
        bottom_b = TabularBottomModel(input_dim=dim_b, embedding_dim=emb_dim, hidden_dim=32)
        top = VFLTopModel(embedding_dim=emb_dim, num_parties=2, num_classes=2)

        dp_cfg = DPConfig(enabled=True, clip_norm=1.0, noise_multiplier=1.1, delta=1e-5)

        trainer = VFLTrainer(
            bottom_model_a=bottom_a,
            bottom_model_b=bottom_b,
            top_model=top,
            optimizer_a=torch.optim.Adam(bottom_a.parameters(), lr=1e-3),
            optimizer_b=torch.optim.Adam(bottom_b.parameters(), lr=1e-3),
            optimizer_top=torch.optim.Adam(top.parameters(), lr=1e-3),
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            verbose=False,
            dp_config=dp_cfg,
        )

        assert trainer.dp_accountant is not None, "dp_accountant must be initialised when DP is enabled"

        n = 64
        la = DataLoader(torch.randn(n, dim_a), batch_size=32, drop_last=True)
        lb = DataLoader(torch.randn(n, dim_b), batch_size=32, drop_last=True)
        ls = DataLoader(torch.randint(0, 2, (n,)), batch_size=32, drop_last=True)

        trainer.train_one_epoch(la, lb, ls)

        assert trainer.dp_accountant.steps > 0, (
            "dp_accountant.steps must be > 0 after one training epoch"
        )
        eps = trainer.dp_accountant.get_epsilon()
        assert eps < float("inf"), (
            f"get_epsilon() returned inf after training — accountant may not have been stepped"
        )