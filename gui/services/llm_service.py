"""Asynchronous QWEN sentence refinement service for GUI."""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from queue import Queue

from config.config import QWEN_N_GPU_LAYERS
from utils.llm_utils import generate_reply, load_qwen_model


@dataclass(slots=True)
class LLMResultEvent:
    """Data emitted back to UI when a refinement completes."""

    text: str
    source_text: str


@dataclass(slots=True)
class LLMRequest:
    """Request payload for refinement queue."""

    text: str
    language: str


class LLMService:
    """Queue-based non-blocking service that wraps QWEN inference."""

    def __init__(self, event_queue: Queue[dict], logger: logging.Logger) -> None:
        self._event_queue = event_queue
        self._logger = logger
        self._requests: queue.Queue[LLMRequest] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._llm = None
        self._load_lock = threading.Lock()
        self._is_loading = False
        self._history: deque[str] = deque(maxlen=2)
        self._worker.start()

    def preload_model(self) -> None:
        """Start QWEN model loading in background before first request."""
        threading.Thread(target=self._ensure_loaded, daemon=True).start()

    def request_refinement(self, sentence: str, language: str = "tr") -> None:
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
            self._requests.put_nowait(LLMRequest(text=text, language=language))
        except queue.Full:
            pass

    def shutdown(self) -> None:
        """Stop worker thread gracefully."""
        self._stop_event.set()
        try:
            self._requests.put_nowait(LLMRequest(text="", language="tr"))
        except queue.Full:
            pass
        self._worker.join(timeout=1.0)

    def _emit_status(
        self,
        message: str,
        progress: str | None = None,
        backend: str | None = None,
    ) -> None:
        payload: dict[str, str] = {"type": "llm_status", "message": message}
        if progress:
            payload["progress"] = progress
        if backend:
            payload["backend"] = backend
        self._event_queue.put(payload)

    def _emit_result(self, result: LLMResultEvent) -> None:
        self._event_queue.put(
            {
                "type": "llm_text",
                "text": result.text,
                "source_text": result.source_text,
            }
        )

    def _ensure_loaded(self) -> bool:
        while not self._stop_event.is_set():
            if self._llm is not None:
                return True

            should_load = False
            with self._load_lock:
                if self._llm is not None:
                    return True
                if not self._is_loading:
                    self._is_loading = True
                    should_load = True

            if should_load:
                break

            # Another thread is preloading the model. Keep the queued request
            # alive until that load finishes instead of dropping it.
            self._stop_event.wait(0.05)

        if self._stop_event.is_set():
            return False

        try:
            if self._llm is not None:
                return True
            self._emit_status(
                "QWEN model is loading...",
                progress="loading",
                backend="unknown",
            )
            self._logger.info("Loading QWEN model for GUI refinement")
            llm = load_qwen_model()
            if llm is None:
                self._logger.warning("QWEN unavailable; refinement disabled")
                self._emit_status(
                    "QWEN unavailable. Check model path and dependencies.",
                    progress="unavailable",
                    backend="unknown",
                )
                return False

            self._llm = llm
            backend = self._detect_backend()
            self._logger.info("QWEN model loaded")
            self._emit_status("QWEN ready", progress="ready", backend=backend)
            return True
        finally:
            with self._load_lock:
                self._is_loading = False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._requests.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            if not request.text:
                continue

            if not self._ensure_loaded():
                continue

            try:
                backend = self._detect_backend()
                self._emit_status(
                    "QWEN is generating refined text...",
                    progress="generating",
                    backend=backend,
                )
                refined = generate_reply(
                    self._llm,
                    request.text,
                    language=request.language,
                    context=list(self._history),
                )
                if refined:
                    self._emit_result(
                        LLMResultEvent(text=refined, source_text=request.text)
                    )
                    self._history.append(refined)
                    self._logger.info("QWEN refinement generated")
                    self._emit_status(
                        "QWEN refinement complete",
                        progress="ready",
                        backend=backend,
                    )
            except Exception as exc:  # pragma: no cover
                self._logger.exception("QWEN refinement failed: %s", exc)
                self._emit_status(
                    "QWEN refinement failed",
                    progress="error",
                    backend=self._detect_backend(),
                )

    def _detect_backend(self) -> str:
        # Backend mode is inferred from offload support and requested GPU layers.
        if self._llm is None:
            return "unknown"
        try:
            from llama_cpp import llama_cpp as llama_lib

            supports_gpu = bool(llama_lib.llama_supports_gpu_offload())
            if supports_gpu and QWEN_N_GPU_LAYERS != 0:
                return "gpu"
        except Exception:
            return "unknown"
        return "cpu"
