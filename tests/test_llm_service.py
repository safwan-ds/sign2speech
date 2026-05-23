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

    def _fake_load_qwen_model():
        load_started.set()
        assert release_load.wait(timeout=2.0)
        return fake_llm

    def _fake_generate_reply(llm, text, language="tr", context=None):
        assert llm is fake_llm
        assert text == "merhaba ben"
        return "Merhaba, ben buradayım."

    monkeypatch.setattr(
        "gui.services.llm_service.load_qwen_model",
        _fake_load_qwen_model,
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
