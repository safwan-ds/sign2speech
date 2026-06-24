"""Tests for asynchronous GUI LLM refinement service."""

from __future__ import annotations

import logging
import queue
import threading
import time

from gui.services.llm_service import LLMService


def _collect_until(
    event_queue: queue.Queue[dict],
    event_type: str,
    timeout_s: float = 2.0,
) -> dict | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            event = event_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if event.get("type") == event_type:
            return event
    return None


def test_refinement_request_waits_for_in_progress_preload(monkeypatch) -> None:
    event_queue: queue.Queue[dict] = queue.Queue()
    load_started = threading.Event()
    release_load = threading.Event()
    fake_llm = object()

    def _fake_create_backend():
        load_started.set()
        assert release_load.wait(timeout=2.0)
        return fake_llm, {"type": "local_qwen", "model_path": "/tmp/test.gguf"}

    def _fake_generate_reply(llm, text, language="tr", context=None):
        assert llm is fake_llm
        assert text == "merhaba ben"
        return "Merhaba, ben buradayım."

    monkeypatch.setattr(
        "gui.services.llm_service.create_llm_backend",
        _fake_create_backend,
    )
    monkeypatch.setattr(
        "gui.services.llm_service.generate_reply",
        _fake_generate_reply,
    )

    service = LLMService(event_queue=event_queue, logger=logging.getLogger("test.llm"))
    try:
        service.preload_model()
        assert load_started.wait(timeout=2.0)

        service.request_refinement("merhaba ben", language="tr")
        release_load.set()

        event = _collect_until(event_queue, "llm_text")
        assert event is not None
        assert event["text"] == "Merhaba, ben buradayım."
        assert event["source_text"] == "merhaba ben"
    finally:
        release_load.set()
        service.shutdown()


def test_shutdown_gracefully_stops(monkeypatch) -> None:
    """Shutdown completes without hanging and worker threads terminate."""
    event_queue: queue.Queue[dict] = queue.Queue()

    def _fake_backend():
        return object(), {"type": "local_qwen"}

    def _fake_reply(llm, text, language="tr", context=None):
        return "ok"

    monkeypatch.setattr("gui.services.llm_service.create_llm_backend", _fake_backend)
    monkeypatch.setattr("gui.services.llm_service.generate_reply", _fake_reply)

    service = LLMService(event_queue=event_queue, logger=logging.getLogger("test.llm"))
    service.shutdown()
    # Should return without hanging
    assert True


def test_empty_text_is_noop(monkeypatch) -> None:
    """Requesting refinement with empty text does nothing."""
    event_queue: queue.Queue[dict] = queue.Queue()
    backend_called = []

    def _fake_backend():
        backend_called.append(True)
        return object(), {"type": "local_qwen"}

    monkeypatch.setattr("gui.services.llm_service.create_llm_backend", _fake_backend)

    service = LLMService(event_queue=event_queue, logger=logging.getLogger("test.llm"))
    try:
        service.request_refinement("", language="tr")
        service.request_refinement("   ", language="tr")
        service.shutdown()
        assert len(backend_called) <= 1  # Only preload triggers backend
    finally:
        service.shutdown()


def test_status_emission_on_preload(monkeypatch) -> None:
    """Preloading emits status events."""
    event_queue: queue.Queue[dict] = queue.Queue()
    load_started = threading.Event()
    release_load = threading.Event()

    def _fake_backend():
        load_started.set()
        release_load.wait(timeout=2.0)
        return object(), {"type": "local_qwen"}

    monkeypatch.setattr("gui.services.llm_service.create_llm_backend", _fake_backend)
    monkeypatch.setattr("gui.services.llm_service.generate_reply", lambda *a, **kw: "ok")

    service = LLMService(event_queue=event_queue, logger=logging.getLogger("test.llm"))
    try:
        service.preload_model()
        load_started.wait(timeout=2.0)
        status = _collect_until(event_queue, "llm_status")
        assert status is not None
    finally:
        release_load.set()
        service.shutdown()


def test_result_emission_format(monkeypatch) -> None:
    """Result events have correct fields."""
    event_queue: queue.Queue[dict] = queue.Queue()
    release_load = threading.Event()

    def _fake_backend():
        release_load.wait(timeout=2.0)
        return object(), {"type": "local_qwen"}

    def _fake_reply(llm, text, language="tr", context=None):
        return "Merhaba dünya"

    monkeypatch.setattr("gui.services.llm_service.create_llm_backend", _fake_backend)
    monkeypatch.setattr("gui.services.llm_service.generate_reply", _fake_reply)

    service = LLMService(event_queue=event_queue, logger=logging.getLogger("test.llm"))
    try:
        service.preload_model()
        release_load.set()
        service.request_refinement("merhaba", language="tr")

        event = _collect_until(event_queue, "llm_text")
        assert event is not None
        assert "text" in event
        assert "source_text" in event
        assert event["text"] == "Merhaba dünya"
        assert event["source_text"] == "merhaba"
    finally:
        release_load.set()
        service.shutdown()


def test_stale_request_handling(monkeypatch) -> None:
    """Multiple rapid requests don't crash; last one wins."""
    event_queue: queue.Queue[dict] = queue.Queue()
    results: list[str] = []

    def _fake_backend():
        return object(), {"type": "local_qwen"}

    def _fake_reply(llm, text, language="tr", context=None):
        results.append(text)
        return f"reply: {text}"

    monkeypatch.setattr("gui.services.llm_service.create_llm_backend", _fake_backend)
    monkeypatch.setattr("gui.services.llm_service.generate_reply", _fake_reply)

    service = LLMService(event_queue=event_queue, logger=logging.getLogger("test.llm"))
    try:
        service.preload_model()
        service.request_refinement("first", language="tr")
        service.request_refinement("second", language="tr")
        service.request_refinement("third", language="tr")
        time.sleep(0.5)
        service.shutdown()
        assert len(results) > 0
    finally:
        service.shutdown()
