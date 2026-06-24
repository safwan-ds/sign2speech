from __future__ import annotations

import logging
from queue import Queue

import pytest

from gui.services.serial_service import SerialSettings
from gui.services.stream_service import StreamConfig
from gui.services.stream_service import StreamWorker


class _FakePredictor:
    def add_sensor_dict(self, sensor):
        return None

    def can_predict(self):
        return False

    def predict(self):
        return None, None, None, None


class _FakeModelService:
    def require_predictor(self):
        return _FakePredictor()


class _FakeSerialService:
    def connect(self, settings):
        self.connected_with = settings
        return False

    def read_sensor_row(self, timeout=0.2):
        return None


def test_stream_worker_uses_repeated_zero_warning_threshold(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeMonitor:
        def __init__(
            self,
            logger,
            *,
            min_consecutive_samples: int = 0,
            min_interval_seconds: float = 0.0,
            emit=None,
        ) -> None:
            captured["min_consecutive_samples"] = min_consecutive_samples
            captured["emit"] = emit

        def check(self, sensor_row):
            return ()

    monkeypatch.setattr(
        "gui.services.stream_service.FlexZeroWarningMonitor",
        _FakeMonitor,
    )

    event_queue: Queue[dict] = Queue()
    worker = StreamWorker(
        model_service=_FakeModelService(),
        serial_service=_FakeSerialService(),
        event_queue=event_queue,
        logger=logging.getLogger("test.stream.monitor"),
        config=StreamConfig(
            serial_settings=SerialSettings(port="COM11"),
            confidence_threshold=0.5,
            smoothing_window=3,
        ),
    )
    worker._stop_event.set()

    worker.run()

    assert captured["min_consecutive_samples"] == 2


def test_confidence_gap_calculation() -> None:
    """_confidence_gap_for_token computes correct gap between top and runner-up."""
    from gui.services.motion_detection import _confidence_gap_for_token

    probs = {"A": 0.7, "B": 0.2, "C": 0.1}
    gap = _confidence_gap_for_token(probs, "A")
    assert gap == pytest.approx(0.5)

    probs_close = {"X": 0.45, "Y": 0.44, "Z": 0.11}
    gap_close = _confidence_gap_for_token(probs_close, "X")
    assert gap_close == pytest.approx(0.01)


def test_extract_class_list() -> None:
    """_extract_class_list works with list and numpy array."""
    from gui.services.motion_detection import _extract_class_list

    class MockPredictor:
        classes = ["REST", "A", "B"]

    result = _extract_class_list(MockPredictor())
    assert result == ["REST", "A", "B"]


def test_per_class_threshold_loading_empty_dir(tmp_path) -> None:
    """_load_per_class_thresholds returns empty dict for empty dir."""
    from gui.services.motion_detection import _load_per_class_thresholds
    from pathlib import Path

    result = _load_per_class_thresholds(Path(tmp_path))
    assert result == {}


def test_stream_config_defaults() -> None:
    """StreamConfig holds expected default values."""
    config = StreamConfig(
        serial_settings=SerialSettings(port="COM1"),
        confidence_threshold=0.5,
        smoothing_window=3,
    )
    assert config.confidence_threshold == 0.5
    assert config.smoothing_window == 3
    assert config.serial_settings.port == "COM1"
