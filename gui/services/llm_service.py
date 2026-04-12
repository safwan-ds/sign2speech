"""Asynchronous QWEN sentence refinement service for GUI."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from queue import Queue

from utils.llm_utils import generate_turkish_reply, load_qwen_model


@dataclass(slots=True)
class LLMResultEvent:
    """Data emitted back to UI when a refinement completes."""

    text: str
    source_text: str


class LLMService:
    """Queue-based non-blocking service that wraps QWEN inference."""

    def __init__(self, event_queue: Queue[dict], logger: logging.Logger) -> None:
        self._event_queue = event_queue
        self._logger = logger
        self._requests: queue.Queue[str] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._llm = None
        self._load_lock = threading.Lock()
        self._is_loading = False
        self._worker.start()

    def preload_model(self) -> None:
        """Start QWEN model loading in background before first request."""
        threading.Thread(target=self._ensure_loaded, daemon=True).start()

    def request_refinement(self, sentence: str) -> None:
        """Schedule refinement for latest sentence and drop stale work."""
        text = sentence.strip()
        if not text:
            return

        try:
            # Keep latest-only behavior under high update rates.
            while True:
                self._requests.get_nowait()
        except queue.Empty:
            pass

        try:
            self._requests.put_nowait(text)
        except queue.Full:
            pass

    def shutdown(self) -> None:
        """Stop worker thread gracefully."""
        self._stop_event.set()
        try:
            self._requests.put_nowait("")
        except queue.Full:
            pass
        self._worker.join(timeout=1.0)

    def _emit_status(self, message: str) -> None:
        self._event_queue.put({"type": "llm_status", "message": message})

    def _emit_result(self, result: LLMResultEvent) -> None:
        self._event_queue.put(
            {
                "type": "llm_text",
                "text": result.text,
                "source_text": result.source_text,
            }
        )

    def _ensure_loaded(self) -> bool:
        if self._llm is not None:
            return True

        with self._load_lock:
            if self._llm is not None:
                return True
            if self._is_loading:
                return False
            self._is_loading = True

        try:
            self._emit_status("QWEN modeli yükleniyor...")
            self._logger.info("Loading QWEN model for GUI refinement")
            llm = load_qwen_model()
            if llm is None:
                self._logger.warning("QWEN unavailable; refinement disabled")
                self._emit_status(
                    "QWEN kullanılamıyor. Model yolunu/bağımlılıkları kontrol edin."
                )
                return False

            self._llm = llm
            self._logger.info("QWEN model loaded")
            self._emit_status("QWEN hazır")
            return True
        finally:
            with self._load_lock:
                self._is_loading = False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._requests.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            if not text:
                continue

            if not self._ensure_loaded():
                continue

            try:
                refined = generate_turkish_reply(self._llm, text)
                if refined:
                    self._emit_result(LLMResultEvent(text=refined, source_text=text))
                    self._logger.info("QWEN refinement generated")
            except Exception as exc:  # pragma: no cover
                self._logger.exception("QWEN refinement failed: %s", exc)
                self._emit_status("QWEN düzenlemesi başarısız")
