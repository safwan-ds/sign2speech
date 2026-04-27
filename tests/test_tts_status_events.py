"""Tests for TTS status events and UI routing."""

from __future__ import annotations

import queue
import time

from gui.services.tts_service import TTSService
from gui.ui.app_window_events import AppWindowEventMixin


class _DummyWindow(AppWindowEventMixin):
    def __init__(self) -> None:
        self.event_queue: queue.Queue[dict] = queue.Queue()
        self.tts_updates: list[tuple[str, str, str]] = []

    def _set_tts_status_state(
        self, state: str, backend: str, message: str = ""
    ) -> None:
        self.tts_updates.append((state, backend, message))


def _collect_events(
    event_queue: queue.Queue[dict], duration_s: float = 0.7
) -> list[dict]:
    events: list[dict] = []
    deadline = time.time() + duration_s
    while time.time() < deadline:
        try:
            events.append(event_queue.get_nowait())
        except queue.Empty:
            time.sleep(0.01)
    return events


def test_poll_events_routes_tts_status_events() -> None:
    window = _DummyWindow()
    window.event_queue.put(
        {
            "type": "tts_status",
            "state": "working",
            "backend": "edge",
            "message": "Working now",
        }
    )

    window._poll_events()

    assert window.tts_updates == [("working", "edge", "Working now")]


def test_tts_service_emits_edge_status_when_local_worker_unavailable() -> None:
    event_queue: queue.Queue[dict] = queue.Queue()
    service = TTSService(event_queue=event_queue)
    try:
        service._local_worker.is_alive = lambda: False  # type: ignore[method-assign]
        service.speak("merhaba", "tr", backend="local")

        events = _collect_events(event_queue)
        status_events = [e for e in events if e.get("type") == "tts_status"]

        assert status_events
        assert any(
            e.get("state") == "working" and e.get("backend") == "edge"
            for e in status_events
        )
    finally:
        service.stop()
