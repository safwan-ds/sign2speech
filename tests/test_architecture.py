"""Unit tests for config/architecture.py — YAML config loader."""

from __future__ import annotations

import os
import tempfile

import pytest

from config.architecture import (
    ArchitectureConfig,
    HardwareConfig,
    ModelConfig,
    TrainingConfig,
    load_architecture,
)


class TestLoadArchitecture:
    """Tests for load_architecture()."""

    def test_loads_default_yaml(self) -> None:
        cfg = load_architecture()
        assert isinstance(cfg, ArchitectureConfig)
        assert isinstance(cfg.hardware, HardwareConfig)
        assert cfg.model.model_type == "enhanced"

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Architecture YAML not found"):
            load_architecture("nonexistent/path.yaml")

    def test_loads_minimal_yaml(self) -> None:
        yaml_content = """\
hardware:
  com_port: COM5
  num_flex_sensors: 3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            cfg = load_architecture(tmp_path)
            assert cfg.hardware.com_port == "COM5"
            assert cfg.hardware.num_flex_sensors == 3
            assert cfg.model.model_type == "enhanced"
        finally:
            os.unlink(tmp_path)

    def test_missing_sections_use_defaults(self) -> None:
        yaml_content = "hardware:\n  com_port: COM1\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            cfg = load_architecture(tmp_path)
            assert cfg.training.batch_size == 32
            assert cfg.model.lstm_units == 64
        finally:
            os.unlink(tmp_path)

    def test_invalid_top_level_raises(self) -> None:
        yaml_content = "not_a_list: [1, 2, 3]\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unknown section"):
                load_architecture(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_invalid_scalar_top_level_raises(self) -> None:
        yaml_content = "42\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="top-level mapping"):
                load_architecture(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_all_sections_present(self) -> None:
        cfg = load_architecture()
        expected_sections = [
            "hardware", "motion_detection", "model", "training",
            "augmentation", "prediction", "normalization", "general",
            "llm", "gui", "evaluation", "plot",
        ]
        for section in expected_sections:
            assert hasattr(cfg, section), f"Missing section: {section}"


class TestDataclassValidation:
    """Tests for __post_init__ validation across config dataclasses."""

    def test_hardware_invalid_flex_sensor_count(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            HardwareConfig(num_flex_sensors=0)

    def test_hardware_invalid_baud_rate(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            HardwareConfig(baud_rate=0)

    def test_model_invalid_lstm_units(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            ModelConfig(lstm_units=0)

    def test_model_invalid_dropout(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            ModelConfig(dropout_rate=1.5)

    def test_model_invalid_negative_dropout(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            ModelConfig(dropout_rate=-0.1)

    def test_training_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            TrainingConfig(batch_size=0)

    def test_training_invalid_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            TrainingConfig(learning_rate=0.0)

    def test_training_invalid_learning_rate_high(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            TrainingConfig(learning_rate=1.0)
