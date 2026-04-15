"""Event handling helpers for the dashboard window."""

from __future__ import annotations

import queue
import time

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QTableWidgetItem

from core.inference.gesture_translations import translate_gesture
from gui.services.serial_service import SerialService


class AppWindowEventMixin:
    """Process worker events and keep the live UI in sync."""

    def _append_log(
        self,
        level: str,
        message: str,
        timestamp: str = "--:--:--",
        source: str = "gui",
    ) -> None:
        formatted = f"[{timestamp}] {level:<7} {source} | {message}"
        self.log_box.appendPlainText(formatted)

    def _on_prediction(self, event: dict) -> None:
        unknown = self._t("unknown")
        raw_gesture = str(event.get("raw_gesture", event.get("gesture", unknown)))
        event_gesture = str(event.get("gesture", unknown))
        confidence = float(event.get("confidence", 0.0))

        if event_gesture in {"Belirsiz", "Uncertain"}:
            display_gesture = self._t("uncertain")
        else:
            display_gesture = translate_gesture(
                raw_gesture,
                self._gesture_translations,
                target_language=self.ui_language,
            )
        history_gesture = display_gesture

        self.prediction_card.setText(display_gesture.upper())
        self.confidence_bar.setValue(int(max(0.0, min(1.0, confidence)) * 100))
        self._last_confidence = confidence
        self.confidence_label.setText(self._format_confidence(self._last_confidence))

        self.history_table.insertRow(0)
        self.history_table.setItem(
            0, 0, QTableWidgetItem(str(event.get("timestamp", "--:--:--")))
        )
        self.history_table.setItem(0, 1, QTableWidgetItem(history_gesture))
        self.history_table.setItem(0, 2, QTableWidgetItem(f"{confidence:.2%}"))

        while self.history_table.rowCount() > 20:
            self.history_table.removeRow(self.history_table.rowCount() - 1)

    def _on_sentence(self, event: dict) -> None:
        token = str(event.get("token", "")).strip()
        if not token:
            return

        self.current_sentence_tokens.append(token)
        if self.tts_enabled and self.tts_mode == "instant":
            self.tts_service.speak(token, self.ui_language)
        sentence_text = " ".join(self.current_sentence_tokens)
        self.word_count_label.setText(
            self._format_word_count(len(self.current_sentence_tokens))
        )
        self.sentence_box.setPlainText(sentence_text)

    def _on_llm_request(self, event: dict) -> None:
        if not self.llm_enabled:
            return

        text = str(event.get("text", "")).strip()
        if not text:
            return
        self.llm_service.request_refinement(text, self._effective_llm_language())

    def _on_llm_text(self, event: dict) -> None:
        text = str(event.get("text", "")).strip()
        if not text:
            return
        self.refined_box.setPlainText(text)
        self._set_llm_progress_state("ready")

        if self.tts_enabled and self.tts_mode == "llm":
            self.tts_service.speak(text, self._effective_llm_language())

    def _on_llm_status(self, event: dict) -> None:
        message = str(event.get("message", "")).strip()
        progress = str(event.get("progress", "")).strip().lower()
        backend = str(event.get("backend", "")).strip().lower()

        if progress:
            self._set_llm_progress_state(progress)
        if backend in {"gpu", "cpu", "unknown"}:
            self._set_llm_backend_state(backend)

        if message:
            self._set_status(message, "INFO")

    def _on_connected(self, value: bool) -> None:
        self._set_connection_badge(value)

    def _refresh_runtime_status(self) -> None:
        if self.worker and self.worker.is_alive() and self._active_stream_port:
            if self._active_stream_port not in SerialService.list_ports():
                self.stop_stream(
                    self._tf(
                        "selected_port_no_longer_available",
                        port=self._active_stream_port,
                    ),
                    "WARNING",
                )
                self._set_connection_badge(False)
                return

        if (
            self.worker
            and self.worker.is_alive()
            and self._stream_start_timeout_at is not None
            and not self._stream_input_detected
            and time.monotonic() >= self._stream_start_timeout_at
        ):
            self.stop_stream(
                self._t("selected_port_no_valid_data_stream_stopped"),
                "WARNING",
            )
            self._set_connection_badge(False)
            return

        if self._stream_started_at is None:
            self.runtime_label.setText(self._format_runtime("00:00"))
            return

        elapsed = int(time.time() - self._stream_started_at)
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            text = self._format_runtime(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            text = self._format_runtime(f"{minutes:02d}:{seconds:02d}")
        self.runtime_label.setText(text)

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                event_type = event.get("type")

                if event_type == "prediction":
                    self._on_prediction(event)
                elif event_type == "sentence":
                    self._on_sentence(event)
                elif event_type == "connected":
                    connected = bool(event.get("value", False))
                    self._on_connected(connected)
                    if connected:
                        self._stream_connected = True
                        self.start_stop_btn.setText(self._t("stop_stream"))
                        if not self._stream_input_detected:
                            self._set_status(
                                self._t("port_connected_waiting_data"), "INFO"
                            )
                elif event_type == "stream_input_detected":
                    self._stream_input_detected = True
                    self._stream_start_timeout_at = None
                    self._set_status(self._t("port_validated_preparing_stream"), "INFO")
                elif event_type == "stream_started":
                    self._stream_started_at = time.time()
                    self._set_status(self._t("stream_started"), "INFO")
                elif event_type == "model_loaded":
                    self._update_model_meta(event["metadata"])
                    self._set_status(self._t("model_loaded_successfully"), "INFO")
                elif event_type == "stopped":
                    was_connected = self._stream_connected
                    had_error = self._stream_had_error
                    stop_requested = self._stream_stop_requested
                    self._stream_started_at = None
                    self._stream_connected = False
                    self._active_stream_port = None
                    self.start_stop_btn.setText(self._t("start_stream"))
                    if stop_requested:
                        if self._stream_stop_message:
                            self._set_status(
                                self._stream_stop_message, self._stream_stop_level
                            )
                        else:
                            self._set_status(self._t("stream_stopped"), "INFO")
                    elif not had_error:
                        if was_connected:
                            self._set_status(
                                self._t("stream_stopped_unexpectedly"), "WARNING"
                            )
                        else:
                            self._set_status(self._t("stream_failed_to_start"), "ERROR")
                    self._refresh_action_states()
                elif event_type == "error":
                    message = str(event.get("message", self._t("unknown_error")))
                    self._stream_had_error = True
                    self._set_status(message, "ERROR")
                    if "Model" in message:
                        self._model_loaded = False
                        self._set_model_badge(self._t("model_error"), "error")
                        self.load_btn.setEnabled(True)
                        self._refresh_action_states()
                    self._append_log("ERROR", message, source="ui")
                elif event_type == "log":
                    self._append_log(
                        str(event.get("level", "INFO")),
                        str(event.get("message", "")),
                        str(event.get("timestamp", "--:--:--")),
                        str(event.get("source", "gui")),
                    )
                elif event_type == "llm_text":
                    self._on_llm_text(event)
                elif event_type == "llm_status":
                    self._on_llm_status(event)
                elif event_type == "llm_request":
                    self._on_llm_request(event)
        except queue.Empty:
            return

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop_stream()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.5)
        self.llm_service.shutdown()
        self.tts_service.stop()
        event.accept()
