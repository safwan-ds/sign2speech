"""Shared pytest fixtures for the sign2speech test suite."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from core.models.lstm_model import build_lstm_model

INPUT_SIZE = 11
HIDDEN = 64
NUM_CLASSES = 5
DEVICE = torch.device("cpu")


@pytest.fixture
def synthetic_model_checkpoint(tmp_path):
    """Minimal LSTM checkpoint saved to a temp directory.

    Returns the directory path (model_dir), not the .pth path directly,
    so tests can look up model.pth, encoder.npy, normalization.npz from it.
    """
    model_dir = str(tmp_path / "latest")
    os.makedirs(model_dir, exist_ok=True)

    model = build_lstm_model(
        INPUT_SIZE, NUM_CLASSES, HIDDEN,
        num_layers=1, dropout_rate=0.3,
        device=DEVICE,
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "arch_params": {
            "input_size": INPUT_SIZE,
            "num_classes": NUM_CLASSES,
            "hidden_size": HIDDEN,
            "num_layers": 1,
            "dropout_rate": 0.3,
            "model_type": "enhanced",
        },
    }
    torch.save(checkpoint, os.path.join(model_dir, "model.pth"))
    return model_dir


@pytest.fixture
def synthetic_encoder(tmp_path):
    """Fake LabelEncoder classes saved as .npy file."""
    classes = np.array(["REST", "A", "B", "C", "D"], dtype=object)
    path = str(tmp_path / "encoder.npy")
    np.save(path, classes)
    return path


@pytest.fixture
def synthetic_norm_params(tmp_path):
    """Fake normalization parameters (mean=0, std=1 for 11 features)."""
    mean = np.zeros(INPUT_SIZE, dtype=np.float32)
    std = np.ones(INPUT_SIZE, dtype=np.float32)
    path = str(tmp_path / "normalization.npz")
    np.savez(path, mean=mean, std=std)
    return path


@pytest.fixture
def sample_sequences():
    """Return (X, y) — synthetic gesture sequences ready for training.

    X shape: (20, 30, 11) — 20 samples, 30 timesteps, 11 features
    y: 20 string labels
    """
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (20, 30, INPUT_SIZE)).astype(np.float32)
    labels = ["REST", "A", "B", "C", "D"] * 4
    y = np.array(labels[:20], dtype=str)
    return X, y


@pytest.fixture
def sample_sensor_csv(tmp_path):
    """Create a minimal 10-row CSV file matching the sensor log format.

    Returns the path to the CSV file.
    """
    import pandas as pd

    rng = np.random.default_rng(42)
    data: dict[str, np.ndarray] = {}
    for i in range(5):
        data[f"flex{i}"] = rng.integers(25, 300, 10).astype(float)
    for axis in ["X", "Y", "Z"]:
        data[f"accel{axis}"] = rng.normal(0, 1000, 10)
        data[f"gyro{axis}"] = rng.normal(0, 500, 10)

    data["gesture"] = ["REST"] * 5 + ["A"] * 5

    df = pd.DataFrame(data)
    path = str(tmp_path / "recording.csv")
    df.to_csv(path, index=False)
    return path
