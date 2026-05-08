from __future__ import annotations

import logging
from queue import Queue

from gui.services.serial_service import SerialSettings
from gui.services.stream_service import StreamConfig, StreamWorker


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
