"""Action and workflow helpers for the dashboard window."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLineEdit,
    QSlider,
)

from config import LOGS_OUTPUT_DIR, MODELS_DIR
from gui.services.model_service import ModelMetadata
from gui.services.serial_service import SerialService, SerialSettings
from gui.services.stream_service import StreamConfig, StreamWorker
from gui.utils.exporter import export_sentence
from gui.utils.formatting import now_stamp
from utils.serial_utils import select_serial_port


class AppWindowActionsMixin:
    """Handle model, streaming, recording, and file operations."""

    def _clear_logs_view(self) -> None:
        self.log_box.clear()
        self._set_status(self._t("log_view_cleared"), "INFO")

    def _export_logs_view(self) -> None:
        text = self.log_box.toPlainText().strip()
        if not text:
            self._set_status(self._t("no_logs_to_export"), "WARNING")
            return
        target = Path(LOGS_OUTPUT_DIR) / f"gui_view_{now_stamp()}.log"
        target.write_text(text + "\n", encoding="utf-8")
        self._set_status(self._tf("logs_exported", name=target.name), "INFO")

    def refresh_ports(self) -> None:
        current = self._selected_port()
        entries = SerialService.list_port_entries()
        self.port_combo.clear()
        if not entries:
            self.port_combo.addItem(self._t("no_ports"))
            self._set_status(self._t("no_serial_ports_found"), "WARNING")
            self._refresh_action_states()
            return

        ports = [device for _label, device in entries]
        for label, device in entries:
            self.port_combo.addItem(label, device)

        preferred_port = ""
        for label, device in entries:
            if "USB-SERIAL CH340" in label.upper():
                preferred_port = device
                break

        if not preferred_port:
            preferred_port = select_serial_port(current if current in ports else None)

        if preferred_port and preferred_port in ports:
            idx = self.port_combo.findData(preferred_port)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)
        else:
            idx = self.port_combo.findData(current)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)
            else:
                self.port_combo.setCurrentIndex(0)

        self._set_status(self._t("serial_ports_refreshed"), "INFO")
        self._refresh_action_states()

    def select_model_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            self._t("select_model_directory"),
            str(self.project_root),
        )
        if chosen:
            self.model_path_edit.setText(chosen)

    def use_latest_model(self) -> None:
        self.model_path_edit.setText(str(Path(MODELS_DIR) / "latest"))

    def load_model_async(self) -> None:
        model_dir = Path(self.model_path_edit.text().strip())
        self._set_status(self._t("model_loading_progress"), "INFO")
        self._set_model_badge(self._t("model_loading"), "loading")
        self._model_loaded = False
        self.load_btn.setEnabled(False)
        self._refresh_action_states()

        def _load() -> None:
            try:
                metadata = self.model_service.load(model_dir)
                self.event_queue.put({"type": "model_loaded", "metadata": metadata})
                self.logger.info("Model loaded from %s", model_dir)
            except Exception as exc:
                self.logger.exception("Model load failed")
                self.event_queue.put(
                    {"type": "error", "message": f"Model load failed: {exc}"}
                )

        threading.Thread(target=_load, daemon=True).start()

    def _update_model_meta(self, metadata: ModelMetadata) -> None:
        classes_preview = ", ".join(metadata.classes[:10])
        self.model_meta_label.setText(
            (
                f"{self._t('model_classes')} ({len(metadata.classes)}): {classes_preview}\n"
                f"{self._t('sequence_length')}: {metadata.sequence_length}\n"
                f"{self._t('input_shape')}: {metadata.input_shape}\n"
                f"{self._t('loaded_at')}: {metadata.loaded_at}"
            )
        )
        self._model_loaded = True
        self._set_model_badge(self._t("model_ready"), "ready")
        self.load_btn.setEnabled(True)
        self._refresh_action_states()

    def _get_stream_config(self) -> StreamConfig:
        port = self._selected_port()
        if not port:
            raise ValueError(self._t("serial_port_not_selected"))

        baud = int(self.baud_edit.text().strip())
        serial_settings = SerialSettings(port=port, baud_rate=baud)
        return StreamConfig(
            serial_settings=serial_settings,
            confidence_threshold=float(self.threshold_slider.value()) / 100.0,
            smoothing_window=int(self.smoothing_slider.value()),
            language=self.ui_language,
        )

    def toggle_stream(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self) -> None:
        if self.model_service.predictor is None:
            self._set_status(self._t("load_model_before_stream"), "WARNING")
            return

        try:
            config = self._get_stream_config()
        except Exception as exc:
            self._set_status(str(exc), "WARNING")
            return

        self.worker = StreamWorker(
            model_service=self.model_service,
            event_queue=self.event_queue,
            logger=self.logger,
            config=config,
        )
        self._active_stream_port = config.serial_settings.port
        self._stream_stop_requested = False
        self._stream_had_error = False
        self._stream_connected = False
        self._stream_stop_message = None
        self._stream_input_detected = False
        self._stream_start_timeout_at = (
            time.monotonic() + self.STREAM_STARTUP_TIMEOUT_SECONDS
        )
        if self.llm_enabled:
            self.llm_service.preload_model()
        self.worker.start()
        self.start_stop_btn.setText(self._t("stop_stream"))
        self._set_status(self._t("stream_starting_waiting_data"), "INFO")

    def stop_stream(
        self, feedback_message: str | None = None, level: str = "INFO"
    ) -> None:
        self._stream_stop_requested = True
        self._stream_stop_message = feedback_message
        self._stream_stop_level = level
        if self.worker and self.worker.is_alive():
            self.worker.stop()
        self._stream_started_at = None
        self._stream_start_timeout_at = None
        self._stream_input_detected = False
        self.start_stop_btn.setText(self._t("start_stream"))
        if feedback_message:
            self._set_status(feedback_message, level)
        else:
            self._set_status(self._t("stream_stopping"), "INFO")
        self._refresh_action_states()

    def copy_sentence(self) -> None:
        sentence = " ".join(self.current_sentence_tokens).strip()
        if not sentence:
            self._set_status(self._t("no_sentence_to_copy"), "WARNING")
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(sentence)
        self._set_status(self._t("sentence_copied"), "INFO")

    def copy_refined(self) -> None:
        text = self.refined_box.toPlainText().strip()
        if not text:
            self._set_status(self._t("no_refined_text_to_copy"), "WARNING")
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self._set_status(self._t("refined_text_copied"), "INFO")

    def clear_sentence(self) -> None:
        self.current_sentence_tokens.clear()
        self.word_count_label.setText(self._format_word_count(0))
        self.sentence_box.setPlainText(self._t("placeholder_sentence"))
        self.refined_box.setPlainText(self._t("placeholder_refined"))
        if self.llm_enabled:
            self._set_llm_progress_state("idle")
        self._set_status(self._t("sentence_cleared"), "INFO")
        self._refresh_action_states()

    def export_sentence_text(self) -> None:
        sentence = " ".join(self.current_sentence_tokens)
        if not sentence.strip():
            self._set_status(self._t("no_sentence_to_export"), "WARNING")
            return

        target = export_sentence(sentence, Path(LOGS_OUTPUT_DIR))
        self._set_status(self._tf("sentence_exported", name=target.name), "INFO")
        self.logger.info("Sentence exported to %s", target)
