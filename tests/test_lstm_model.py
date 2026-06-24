"""Unit tests for core/models/lstm_model.py — LSTM model architectures."""

from __future__ import annotations

import torch
import pytest

from core.models.lstm_model import AttentionLayer, LSTMModel, TransformerLSTMModel, build_lstm_model


BATCH = 4
SEQ_LEN = 30
INPUT_SIZE = 11
HIDDEN = 64
NUM_CLASSES = 8
DEVICE = torch.device("cpu")


def _make_input(batch: int = BATCH, seq: int = SEQ_LEN, feats: int = INPUT_SIZE) -> torch.Tensor:
    return torch.randn(batch, seq, feats, device=DEVICE)


class TestAttentionLayer:
    """Tests for AttentionLayer."""

    def test_forward_shape(self) -> None:
        layer = AttentionLayer(HIDDEN)
        x = torch.randn(2, 10, HIDDEN, device=DEVICE)
        context, weights = layer(x)
        assert context.shape == (2, HIDDEN)
        assert weights.shape == (2, 10)

    def test_weights_sum_to_one(self) -> None:
        layer = AttentionLayer(HIDDEN)
        x = torch.randn(1, 5, HIDDEN, device=DEVICE)
        _, weights = layer(x)
        assert torch.allclose(weights.sum(dim=1), torch.ones(1, device=DEVICE), atol=1e-6)


class TestLSTMModel:
    """Tests for LSTMModel (enhanced)."""

    @pytest.fixture
    def model(self) -> LSTMModel:
        return LSTMModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout_rate=0.4,
            bidirectional=True,
            use_attention=True,
            use_batch_norm=True,
        ).to(DEVICE)

    def test_forward_output_shape(self, model: LSTMModel) -> None:
        model.eval()
        out = model(_make_input())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_no_attention(self) -> None:
        model = LSTMModel(INPUT_SIZE, HIDDEN, 2, NUM_CLASSES, 0.4, use_attention=False).to(DEVICE)
        model.eval()
        out = model(_make_input())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_no_batch_norm(self) -> None:
        model = LSTMModel(INPUT_SIZE, HIDDEN, 2, NUM_CLASSES, 0.4, use_batch_norm=False).to(DEVICE)
        model.eval()
        out = model(_make_input())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_unidirectional(self) -> None:
        model = LSTMModel(INPUT_SIZE, HIDDEN, 2, NUM_CLASSES, 0.4, bidirectional=False).to(DEVICE)
        model.eval()
        out = model(_make_input())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_single_batch(self) -> None:
        model = LSTMModel(INPUT_SIZE, HIDDEN, 1, NUM_CLASSES, 0.3, use_batch_norm=False).to(DEVICE)
        model.eval()
        out = model(_make_input(batch=1))
        assert out.shape == (1, NUM_CLASSES)


class TestTransformerLSTMModel:
    """Tests for TransformerLSTMModel (advanced)."""

    @pytest.fixture
    def model(self) -> TransformerLSTMModel:
        return TransformerLSTMModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout_rate=0.3,
            bidirectional=True,
        ).to(DEVICE)

    def test_forward_output_shape(self, model: TransformerLSTMModel) -> None:
        model.eval()
        out = model(_make_input())
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_single_batch(self) -> None:
        model = TransformerLSTMModel(INPUT_SIZE, HIDDEN, 1, NUM_CLASSES, 0.3).to(DEVICE)
        model.eval()
        out = model(_make_input(batch=1))
        assert out.shape == (1, NUM_CLASSES)


class TestBuildLSTMModel:
    """Tests for build_lstm_model() factory function."""

    def test_builds_enhanced(self) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            model_type="enhanced", device=DEVICE,
        )
        assert isinstance(model, LSTMModel)
        model.eval()
        out = model(_make_input(batch=2))
        assert out.shape == (2, NUM_CLASSES)

    def test_builds_advanced(self) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            model_type="advanced", device=DEVICE,
        )
        assert isinstance(model, TransformerLSTMModel)
        model.eval()
        out = model(_make_input(batch=2))
        assert out.shape == (2, NUM_CLASSES)

    def test_builds_basic(self) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            model_type="basic", device=DEVICE,
        )
        assert isinstance(model, LSTMModel)
        assert model.bidirectional is False
        assert model.use_attention is False

    def test_builds_with_unknown_type_falls_back_to_basic(self) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            model_type="garbage", device=DEVICE,
        )
        assert isinstance(model, LSTMModel)

    def test_model_moved_to_cpu(self) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        param_device = next(model.parameters()).device
        assert param_device.type == "cpu"

    def test_save_load_roundtrip(self, tmp_path) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        path = str(tmp_path / "model.pth")
        torch.save({"state_dict": model.state_dict(), "arch_params": {
            "input_size": INPUT_SIZE, "num_classes": NUM_CLASSES,
            "hidden_size": HIDDEN, "num_layers": 2, "dropout_rate": 0.4,
            "model_type": "enhanced",
        }}, path)

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model2 = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        model2.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model2.eval()

        x = _make_input(batch=1)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6)
