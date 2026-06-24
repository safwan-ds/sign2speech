"""Unit tests for core/models/model_factory.py — checkpoint loading."""

from __future__ import annotations

import pytest
import torch

from core.models.lstm_model import build_lstm_model
from core.models.model_factory import build_model_from_checkpoint
from core.models.model_factory import load_model_checkpoint

INPUT_SIZE = 11
HIDDEN = 64
NUM_CLASSES = 8
DEVICE = torch.device("cpu")


def _save_dummy_checkpoint(path: str, model_type: str = "enhanced") -> None:
    model = build_lstm_model(
        INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
        model_type=model_type, device=DEVICE,
    )
    torch.save({
        "state_dict": model.state_dict(),
        "arch_params": {
            "input_size": INPUT_SIZE,
            "num_classes": NUM_CLASSES,
            "hidden_size": HIDDEN,
            "num_layers": 2,
            "dropout_rate": 0.4,
            "model_type": model_type,
            "bidirectional": True,
            "use_attention": True,
            "use_batch_norm": True,
        },
    }, path)


class TestLoadModelCheckpoint:
    """Tests for load_model_checkpoint()."""

    def test_loads_modern_format(self, tmp_path) -> None:
        path = str(tmp_path / "model.pth")
        _save_dummy_checkpoint(path)
        state_dict, arch_params, in_size, n_classes = load_model_checkpoint(
            path, DEVICE
        )
        assert isinstance(state_dict, dict)
        assert "lstm.weight_ih_l0" in state_dict
        assert arch_params["model_type"] == "enhanced"
        assert in_size == INPUT_SIZE
        assert n_classes == NUM_CLASSES

    def test_loads_legacy_format(self, tmp_path) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        path = str(tmp_path / "legacy.pth")
        torch.save(model.state_dict(), path)
        state_dict, arch_params, in_size, n_classes = load_model_checkpoint(
            path, DEVICE
        )
        assert arch_params == {}
        assert in_size == INPUT_SIZE
        assert n_classes is not None

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_checkpoint("nonexistent.pth", DEVICE)

    def test_infers_num_classes_from_fc_out(self, tmp_path) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        path = str(tmp_path / "no_meta.pth")
        torch.save({"state_dict": model.state_dict()}, path)
        _, _, _, n_classes = load_model_checkpoint(path, DEVICE)
        assert n_classes == NUM_CLASSES

    def test_infers_input_size_from_lstm_weight(self, tmp_path) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        path = str(tmp_path / "no_meta.pth")
        torch.save({"state_dict": model.state_dict(), "arch_params": {}}, path)
        _, _, in_size, _ = load_model_checkpoint(path, DEVICE)
        assert in_size == INPUT_SIZE


class TestBuildModelFromCheckpoint:
    """Tests for build_model_from_checkpoint()."""

    def test_builds_and_loads_model(self, tmp_path) -> None:
        path = str(tmp_path / "model.pth")
        _save_dummy_checkpoint(path)
        model, arch_params, in_size, n_classes = build_model_from_checkpoint(
            path, DEVICE
        )
        assert model.training is False
        assert in_size == INPUT_SIZE
        assert n_classes == NUM_CLASSES
        x = torch.randn(1, 30, INPUT_SIZE, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_encoder_fallback_num_classes(self, tmp_path) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        path = str(tmp_path / "model.pth")
        torch.save({
            "state_dict": model.state_dict(),
            "arch_params": {"input_size": INPUT_SIZE, "model_type": "enhanced"},
        }, path)
        model2, _, _, _ = build_model_from_checkpoint(
            path, DEVICE, encoder_num_classes=8
        )
        out = model2(torch.randn(1, 30, INPUT_SIZE, device=DEVICE))
        assert out.shape == (1, 8)

    def test_missing_num_classes_raises(self, tmp_path) -> None:
        model = build_lstm_model(
            INPUT_SIZE, NUM_CLASSES, HIDDEN, num_layers=2, dropout_rate=0.4,
            device=DEVICE,
        )
        path = str(tmp_path / "model.pth")
        sd = {k: v for k, v in model.state_dict().items()
              if not k.startswith("fc_out.") and not k.startswith("classifier.")}
        torch.save({
            "state_dict": sd,
            "arch_params": {"input_size": INPUT_SIZE, "model_type": "enhanced"},
        }, path)
        with pytest.raises(ValueError, match="Cannot determine number of classes"):
            build_model_from_checkpoint(path, DEVICE)
