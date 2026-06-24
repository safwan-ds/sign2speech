"""Tests for core/training/model_training.py — training loop in isolation."""

from __future__ import annotations

import pytest
import torch

from core.models.lstm_model import build_lstm_model

DEVICE = torch.device("cpu")
INPUT_SIZE = 11
SEQ_LEN = 30
NUM_CLASSES = 5


@pytest.fixture
def tiny_model():
    """A minimal LSTM model for training step tests."""
    return build_lstm_model(
        INPUT_SIZE, NUM_CLASSES, 64,
        num_layers=1, dropout_rate=0.3,
        device=DEVICE,
    )


@pytest.fixture
def tiny_batch():
    """Synthetic (X, y) batch — 4 samples, 30 timesteps, 11 features."""
    X = torch.randn(4, SEQ_LEN, INPUT_SIZE, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (4,), device=DEVICE)
    return X, y


class TestModelForward:
    """Tests for model forward pass behavior."""

    def test_forward_output_shape(self, tiny_model: torch.nn.Module, tiny_batch: tuple) -> None:
        X, _ = tiny_batch
        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(X)
        assert out.shape == (4, NUM_CLASSES)

    def test_training_mode_forward(self, tiny_model: torch.nn.Module, tiny_batch: tuple) -> None:
        X, _ = tiny_batch
        tiny_model.train()
        out = tiny_model(X)
        assert out.shape == (4, NUM_CLASSES)

    def test_loss_computation(self, tiny_model: torch.nn.Module, tiny_batch: tuple) -> None:
        X, y = tiny_batch
        tiny_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        out = tiny_model(X)
        loss = criterion(out, y)
        assert loss.item() > 0.0
        assert loss.requires_grad

    def test_weighted_loss(self, tiny_model: torch.nn.Module, tiny_batch: tuple) -> None:
        X, y = tiny_batch
        weights = torch.tensor([0.5, 1.0, 2.0, 1.0, 0.8], device=DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        out = tiny_model(X)
        loss = criterion(out, y)
        assert loss.item() > 0.0

    def test_label_smoothing_does_not_crash(self, tiny_model: torch.nn.Module, tiny_batch: tuple) -> None:
        X, y = tiny_batch
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        out = tiny_model(X)
        loss = criterion(out, y)
        assert loss.item() > 0.0

    def test_backward_pass(self, tiny_model: torch.nn.Module, tiny_batch: tuple) -> None:
        X, y = tiny_batch
        tiny_model.train()
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = criterion(tiny_model(X), y)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0.0

    def test_model_parameters_update(self, tiny_model: torch.nn.Module, tiny_batch: tuple) -> None:
        X, y = tiny_batch
        before = next(tiny_model.parameters()).clone().detach()
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        loss = criterion(tiny_model(X), y)
        loss.backward()
        optimizer.step()
        after = next(tiny_model.parameters()).detach()
        assert not torch.equal(before, after)
