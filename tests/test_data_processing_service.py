from __future__ import annotations

import logging
import time
from queue import Queue

from gui.services.data_processing_service import DataProcessingService


class _FakeDataFrame:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


def _wait_until_done(service: DataProcessingService, timeout: float = 1.0) -> None:
    start = time.time()
    while service.is_running and (time.time() - start) < timeout:
        time.sleep(0.01)


def _drain_events(event_queue: Queue[dict]) -> list[dict]:
    events: list[dict] = []
    while not event_queue.empty():
        events.append(event_queue.get())
    return events


def test_data_processing_service_emits_happy_path_events(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.services.data_processing_service.clear_previous_sequence_files",
        lambda: None,
    )
    monkeypatch.setattr(
        "gui.services.data_processing_service.load_all_logs",
        lambda: {"HELLO": [_FakeDataFrame(12), _FakeDataFrame(18)]},
    )

    class _FakeSequence:
        shape = (3, 30, 11)

        def __len__(self) -> int:
            return int(self.shape[0]) if self.shape else 0

    monkeypatch.setattr(
        "gui.services.data_processing_service.prepare_lstm_dataset",
        lambda dataframes, gesture: (_FakeSequence(), [gesture, gesture, gesture]),
    )
    monkeypatch.setattr(
        "gui.services.data_processing_service.save_processed_data_lstm",
        lambda X, y, gesture, output_dir: f"{output_dir}/sequences_{gesture}.npz",
    )

    queue: Queue[dict] = Queue()
    service = DataProcessingService(
        logger=logging.getLogger("test.process.happy"),
        event_queue=queue,
    )

    assert service.start() is True
    _wait_until_done(service)
    events = _drain_events(queue)
    event_types = [event["type"] for event in events]

    assert "process_started" in event_types
    assert "process_total_gestures" in event_types
    assert event_types.count("process_gesture_summary") == 1
    assert event_types.count("process_gesture_current") == 1
    assert "process_train_sequences" in event_types
    assert "process_progress" in event_types
    assert "process_completed" in event_types


def test_data_processing_service_emits_failed_event_on_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.services.data_processing_service.clear_previous_sequence_files",
        lambda: None,
    )

    def _raise() -> dict[str, list[_FakeDataFrame]]:
        raise RuntimeError("boom")

    monkeypatch.setattr("gui.services.data_processing_service.load_all_logs", _raise)

    queue: Queue[dict] = Queue()
    service = DataProcessingService(
        logger=logging.getLogger("test.process.failed"),
        event_queue=queue,
    )

    assert service.start() is True
    _wait_until_done(service)
    events = _drain_events(queue)
    failed_event = next(event for event in events if event["type"] == "process_failed")
    assert "boom" in failed_event["message"]


def test_data_processing_service_rejects_concurrent_start(monkeypatch) -> None:
    def _slow_run(self: DataProcessingService) -> None:
        time.sleep(0.1)
        with self._lock:
            self._thread = None

    monkeypatch.setattr(DataProcessingService, "_run", _slow_run)

    queue: Queue[dict] = Queue()
    service = DataProcessingService(
        logger=logging.getLogger("test.process.busy"),
        event_queue=queue,
    )

    assert service.start() is True
    assert service.start() is False
    _wait_until_done(service)


def test_data_processing_service_stop_reports_running_state(monkeypatch) -> None:
    def _slow_run(self: DataProcessingService) -> None:
        while not self._cancel_event.is_set():
            time.sleep(0.01)
        with self._lock:
            self._thread = None

    monkeypatch.setattr(DataProcessingService, "_run", _slow_run)

    queue: Queue[dict] = Queue()
    service = DataProcessingService(
        logger=logging.getLogger("test.process.stop"),
        event_queue=queue,
    )

    assert service.stop() is False
    assert service.start() is True
    assert service.stop() is True
    _wait_until_done(service)
