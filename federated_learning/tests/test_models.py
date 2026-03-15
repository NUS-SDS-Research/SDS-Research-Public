"""
Tests for Bottom Models and Top Model.

All tests use random tensors — no data downloads required.
"""
import pytest
import torch


class TestTabularBottomModel:

    def test_output_shape(self):
        from src.models.bottom_models import TabularBottomModel

        model = TabularBottomModel(input_dim=7, embedding_dim=128)
        x = torch.randn(16, 7)
        out = model(x)

        assert out.shape == (16, 128), f"Expected (16, 128), got {out.shape}"

    def test_output_is_float32(self):
        from src.models.bottom_models import TabularBottomModel

        model = TabularBottomModel(input_dim=5)
        x = torch.randn(8, 5)
        out = model(x)
        assert out.dtype == torch.float32

    def test_different_input_dims(self):
        from src.models.bottom_models import TabularBottomModel

        for dim in (3, 7, 20, 64):
            model = TabularBottomModel(input_dim=dim, embedding_dim=128)
            x = torch.randn(4, dim)
            out = model(x)
            assert out.shape == (4, 128)

    def test_output_has_gradients(self):
        """Output tensor must carry a grad_fn for the VFL backward pass."""
        from src.models.bottom_models import TabularBottomModel

        model = TabularBottomModel(input_dim=7)
        model.train()
        x = torch.randn(8, 7)
        out = model(x)
        assert out.grad_fn is not None, "Output must have grad_fn (model in train mode)"

    def test_parameters_update_after_backward(self):
        """Verify gradients flow into bottom model parameters."""
        from src.models.bottom_models import TabularBottomModel

        model = TabularBottomModel(input_dim=4, embedding_dim=16, hidden_dim=32)
        model.train()
        x = torch.randn(8, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Param '{name}' has no gradient"


class TestCNNBottomModel:

    def test_output_shape_standard_mnist_half(self):
        """Standard MNIST half: (batch, 1, 14, 28) → (batch, 128)."""
        from src.models.bottom_models import CNNBottomModel

        model = CNNBottomModel(in_channels=1, input_height=14, input_width=28, embedding_dim=128)
        x = torch.randn(8, 1, 14, 28)
        out = model(x)
        assert out.shape == (8, 128), f"Expected (8, 128), got {out.shape}"

    def test_output_shape_bottom_half(self):
        """Bottom half of MNIST (rows 14-28): (batch, 1, 14, 28) → (batch, 128)."""
        from src.models.bottom_models import CNNBottomModel

        model = CNNBottomModel(in_channels=1, input_height=14, input_width=28)
        x = torch.randn(4, 1, 14, 28)
        out = model(x)
        assert out.shape == (4, 128)

    def test_flat_size_computed_dynamically(self):
        """_compute_flat_size should adapt to any (H, W) input."""
        from src.models.bottom_models import CNNBottomModel

        # Non-standard input resolution
        model = CNNBottomModel(in_channels=1, input_height=16, input_width=32)
        x = torch.randn(2, 1, 16, 32)
        out = model(x)
        assert out.shape == (2, 128)

    def test_output_has_gradients(self):
        from src.models.bottom_models import CNNBottomModel

        model = CNNBottomModel()
        model.train()
        x = torch.randn(4, 1, 14, 28)
        out = model(x)
        assert out.grad_fn is not None


class TestVFLTopModel:

    def test_output_shape_binary(self):
        """CiferAI binary classification: (batch, 256) → (batch, 2)."""
        from src.models.top_model import VFLTopModel

        model = VFLTopModel(embedding_dim=128, num_parties=2, num_classes=2)
        emb_a = torch.randn(8, 128)
        emb_b = torch.randn(8, 128)
        logits = model(emb_a, emb_b)
        assert logits.shape == (8, 2), f"Expected (8, 2), got {logits.shape}"

    def test_output_shape_multiclass(self):
        """MNIST 10-class: (batch, 256) → (batch, 10)."""
        from src.models.top_model import VFLTopModel

        model = VFLTopModel(embedding_dim=128, num_parties=2, num_classes=10)
        emb_a = torch.randn(16, 128)
        emb_b = torch.randn(16, 128)
        logits = model(emb_a, emb_b)
        assert logits.shape == (16, 10)

    def test_forward_from_concat_equivalent(self):
        """forward_from_concat(cat([a,b])) must produce same result as forward(a,b)."""
        from src.models.top_model import VFLTopModel

        model = VFLTopModel(embedding_dim=64, num_parties=2, num_classes=5)
        model.eval()

        emb_a = torch.randn(4, 64)
        emb_b = torch.randn(4, 64)

        out1 = model(emb_a, emb_b)
        out2 = model.forward_from_concat(torch.cat([emb_a, emb_b], dim=1))

        assert torch.allclose(out1, out2), (
            "forward() and forward_from_concat() must produce identical outputs"
        )

    def test_logits_have_gradients(self):
        from src.models.top_model import VFLTopModel

        model = VFLTopModel(num_classes=10)
        model.train()
        emb_a = torch.randn(4, 128, requires_grad=True)
        emb_b = torch.randn(4, 128, requires_grad=True)
        logits = model(emb_a, emb_b)
        assert logits.grad_fn is not None