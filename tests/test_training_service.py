from __future__ import annotations

import logging
import time
from queue import Queue

from core.pipeline.training_pipeline import TrainingPipelineResult
from gui.services.training_service import TrainingOverrides
from gui.services.training_service import TrainingService


def _wait_until_done(service: TrainingService, timeout: float = 1.5) -> None:
    start = time.time()
    while service.is_running and (time.time() - start) < timeout:
        time.sleep(0.01)


def _drain_events(event_queue: Queue[dict]) -> list[dict]:
    events: list[dict] = []
    while not event_queue.empty():
        events.append(event_queue.get())
    return events


def test_training_service_emits_happy_path_events(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_training_pipeline(**kwargs):
        captured.update(kwargs)
        kwargs["epoch_callback"](1, 3, 0.5, 80.0, 0.6, 75.0, 0.001)
        return TrainingPipelineResult(
            status="completed",
            model_dir="/tmp/lstm_20260101_000000",
        )

    monkeypatch.setattr(
        "gui.services.training_service.run_training_pipeline",
        _fake_run_training_pipeline,
    )

    queue: Queue[dict] = Queue()
    service = TrainingService(
        logger=logging.getLogger("test.train.happy"),
        event_queue=queue,
    )

    overrides = TrainingOverrides(
        epochs=3,
        learning_rate=0.001,
        batch_size=16,
        early_stopping_patience=2,
    )
    assert service.start(overrides) is True
    _wait_until_done(service)
    events = _drain_events(queue)
    event_types = [event["type"] for event in events]

    assert "train_started" in event_types
    assert "train_epoch" in event_types
    assert "train_model_dir" in event_types
    assert "train_completed" in event_types
    assert captured["n_epochs"] == 3
    assert captured["learning_rate"] == 0.001
    assert captured["batch_size"] == 16
    assert captured["patience"] == 2


def test_training_service_emits_failed_event(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.services.training_service.run_training_pipeline",
        lambda **kwargs: TrainingPipelineResult(status="failed", message="bad data"),
    )

    queue: Queue[dict] = Queue()
    service = TrainingService(
        logger=logging.getLogger("test.train.failed"),
        event_queue=queue,
    )

    assert service.start() is True
    _wait_until_done(service)
    events = _drain_events(queue)
    failed = next(event for event in events if event["type"] == "train_failed")
    assert failed["message"] == "bad data"


def test_training_service_rejects_concurrent_start(monkeypatch) -> None:
    def _slow_run_training_pipeline(**kwargs):
        time.sleep(0.2)
        return TrainingPipelineResult(status="completed")

    monkeypatch.setattr(
        "gui.services.training_service.run_training_pipeline",
        _slow_run_training_pipeline,
    )

    queue: Queue[dict] = Queue()
    service = TrainingService(
        logger=logging.getLogger("test.train.busy"),
        event_queue=queue,
    )

    assert service.start() is True
    assert service.start() is False
    _wait_until_done(service)


def test_training_service_emits_cancelled_event(monkeypatch) -> None:
    def _cancel_aware_run_training_pipeline(**kwargs):
        cancel_event = kwargs["cancel_event"]
        while not cancel_event.is_set():
            time.sleep(0.01)
        return TrainingPipelineResult(status="cancelled")

    monkeypatch.setattr(
        "gui.services.training_service.run_training_pipeline",
        _cancel_aware_run_training_pipeline,
    )

    queue: Queue[dict] = Queue()
    service = TrainingService(
        logger=logging.getLogger("test.train.cancel"),
        event_queue=queue,
    )

    assert service.start() is True
    service.cancel()
    _wait_until_done(service)
    events = _drain_events(queue)
    event_types = [event["type"] for event in events]
    assert "train_cancelled" in event_types
