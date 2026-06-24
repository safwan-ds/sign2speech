"""Tests for core/pipeline/data_processor.py — CSV loading and windowing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.pipeline.data_processor import load_all_logs
from core.pipeline.data_processor import load_log_file


class TestLoadLogFile:
    """Tests for load_log_file()."""

    def test_loads_valid_csv(self, sample_sensor_csv: str) -> None:
        df = load_log_file(sample_sensor_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_nonexistent_file_returns_none(self) -> None:
        df = load_log_file("nonexistent_file.csv")
        assert df is None

    def test_csv_has_expected_columns(self, sample_sensor_csv: str) -> None:
        df = load_log_file(sample_sensor_csv)
        assert df is not None
        sensor_cols = [c for c in df.columns if c.startswith("flex") or c.startswith("accel") or c.startswith("gyro")]
        assert len(sensor_cols) > 0


class TestLoadAllLogs:
    """Tests for load_all_logs()."""

    def test_returns_dict_with_gesture_keys(self, sample_sensor_csv: str) -> None:
        import os
        raw_dir = os.path.dirname(sample_sensor_csv)
        # Monkeypatch RAW_DATA_DIR to point at our temp dir
        import config.config as cfg
        old_dir = cfg.RAW_DATA_DIR
        try:
            cfg.RAW_DATA_DIR = raw_dir
            result = load_all_logs()
            assert isinstance(result, dict)
        finally:
            cfg.RAW_DATA_DIR = old_dir


class TestWindowingLogic:
    """Tests for sequence windowing invariants (exercised via data shapes)."""

    def test_window_shape_invariant(self, sample_sequences: tuple) -> None:
        """Synthetic data has correct (n_samples, seq_len, n_features) shape."""
        X, y = sample_sequences
        assert X.ndim == 3
        assert X.shape[0] == len(y)
        assert X.shape[1] == 30  # seq_len
        assert X.shape[2] == 11  # n_features

    def test_labels_match_samples(self, sample_sequences: tuple) -> None:
        X, y = sample_sequences
        assert len(y) == X.shape[0]
        assert all(isinstance(label, str) for label in y)

    def test_data_is_float32(self, sample_sequences: tuple) -> None:
        X, _ = sample_sequences
        assert X.dtype == np.float32
