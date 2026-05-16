from __future__ import annotations

import logging
from queue import Queue

from config.config import COM_PORT
from gui.services.recording_service import RecordingConfig, RecordingService


class _FakeSerialService:
    def __init__(self) -> None:
        self.connected_with = None

    def connect(self, settings) -> bool:
        self.connected_with = settings
        return False

    def reset_input_buffer(self) -> None:
        return None

    def read_sensor_row(self):
        return None


def test_recording_uses_repeated_zero_warning_threshold(monkeypatch) -> None:
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
        "gui.services.recording_service.FlexZeroWarningMonitor",
        _FakeMonitor,
    )
    monkeypatch.setattr(
        "gui.services.recording_service.select_serial_port",
        lambda preferred_port: "COM11",
    )

    event_queue: Queue[dict] = Queue()
    serial_service = _FakeSerialService()
    service = RecordingService(
        logger=logging.getLogger("test.recording.monitor"),
        event_queue=event_queue,
        serial_service=serial_service,
    )
    service._stop_event.set()

    service._run(RecordingConfig(gesture_label="HELLO"))

    assert captured["min_consecutive_samples"] == 2


def test_recording_prefers_selected_port(monkeypatch) -> None:
    preferred: list[str | None] = []

    def fake_select_serial_port(preferred_port: str | None):
        preferred.append(preferred_port)
        return "COM11"

    monkeypatch.setattr(
        "gui.services.recording_service.select_serial_port",
        fake_select_serial_port,
    )

    event_queue: Queue[dict] = Queue()
    serial_service = _FakeSerialService()
    service = RecordingService(
        logger=logging.getLogger("test.recording.selected"),
        event_queue=event_queue,
        serial_service=serial_service,
    )
    service._stop_event.set()

    service._run(RecordingConfig(gesture_label="HELLO", port="COM9"))

    assert preferred == ["COM9"]
    assert serial_service.connected_with is not None
    assert serial_service.connected_with.port == "COM11"


def test_recording_falls_back_to_config_default_port(monkeypatch) -> None:
    preferred: list[str | None] = []

    def fake_select_serial_port(preferred_port: str | None):
        preferred.append(preferred_port)
        return "COM8"

    monkeypatch.setattr(
        "gui.services.recording_service.select_serial_port",
        fake_select_serial_port,
    )

    event_queue: Queue[dict] = Queue()
    serial_service = _FakeSerialService()
    service = RecordingService(
        logger=logging.getLogger("test.recording.default"),
        event_queue=event_queue,
        serial_service=serial_service,
    )
    service._stop_event.set()

    service._run(RecordingConfig(gesture_label="HELLO"))

    assert preferred == [COM_PORT]
    assert serial_service.connected_with is not None
    assert serial_service.connected_with.port == "COM8"


def test_recording_progress_event_includes_live_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.services.recording_service.select_serial_port",
        lambda preferred_port: "COM11",
    )

    perf_counter_values = iter([0.0, 0.2, 0.3])
    monkeypatch.setattr(
        "gui.services.recording_service.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    event_queue: Queue[dict] = Queue()
    serial_service = _FakeSerialService()
    service = RecordingService(
        logger=logging.getLogger("test.recording.live_preview"),
        event_queue=event_queue,
        serial_service=serial_service,
    )

    sensor_row = {
        "flex0": 1.0,
        "flex1": 2.0,
        "flex2": 3.0,
        "flex3": 4.0,
        "flex4": 5.0,
        "accelX": 6.0,
        "accelY": 7.0,
        "accelZ": 8.0,
        "gyroX": 9.0,
        "gyroY": 10.0,
        "gyroZ": 11.0,
    }

    def read_sensor_row():
        service._stop_event.set()
        return dict(sensor_row)

    serial_service.read_sensor_row = read_sensor_row

    service._run(RecordingConfig(gesture_label="HELLO"))

    events: list[dict] = []
    while not event_queue.empty():
        events.append(event_queue.get())

    progress_event = next(
        event for event in events if event["type"] == "record_progress"
    )
    review_event = next(
        event for event in events if event["type"] == "record_ready_for_review"
    )
    expected_row = {"t_ms": 200, **sensor_row}

    assert progress_event["rows"] == [expected_row]
    assert review_event["rows"] == [expected_row]
    assert progress_event["rows"] is not review_event["rows"]
