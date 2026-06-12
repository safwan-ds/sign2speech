"""Action and workflow helpers for the dashboard window."""

# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
)

from config.config import LOGS_OUTPUT_DIR, MODELS_DIR
from gui.services.model_service import ModelMetadata
from gui.services.serial_service import SerialService, SerialSettings
from gui.services.stream_service import StreamConfig, StreamWorker
from gui.utils.exporter import export_sentence
from gui.utils.formatting import now_stamp
from utils.serial_utils import select_serial_port


class AppWindowActionsMixin:
    """Handle model, streaming, recording, and file operations."""

    def _discover_model_dirs(self) -> list[Path]:
        root = Path(MODELS_DIR)
        if not root.exists():
            return []

        discovered: list[Path] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            has_single = (entry / "model.pth").exists()
            has_ensemble = (entry / "model_0.pth").exists()
            if (has_single or has_ensemble) and (entry / "encoder.npy").exists():
                discovered.append(entry)

        discovered.sort(key=lambda item: item.name.lower(), reverse=True)
        latest_dir = root / "latest"
        if latest_dir in discovered:
            discovered.remove(latest_dir)
            discovered.insert(0, latest_dir)
        return discovered

    def _model_display_name(self, model_dir: Path) -> str:
        if model_dir.name == "latest":
            return self._tf("latest_model", name=self._t("latest"))
        return model_dir.name

    def _set_selected_model_dir(self, model_dir: Path, custom: bool = False) -> None:
        if self.model_dir_combo.count() == 1 and self.model_dir_combo.itemData(0) == "":
            self.model_dir_combo.clear()

        self.model_dir_combo.setEnabled(True)
        target = str(model_dir)
        index = self.model_dir_combo.findData(target)
        if index < 0:
            label = (
                self._tf("custom_model", name=model_dir.name)
                if custom
                else self._model_display_name(model_dir)
            )
            self.model_dir_combo.addItem(label, target)
            index = self.model_dir_combo.count() - 1

        self.model_dir_combo.setCurrentIndex(index)
        self.model_path_edit.setText(target)

    def refresh_model_dirs(self) -> None:
        current_text = self.model_path_edit.text().strip()
        current_dir = Path(current_text) if current_text else None
        model_dirs = self._discover_model_dirs()

        self.model_dir_combo.blockSignals(True)
        self.model_dir_combo.clear()

        for model_dir in model_dirs:
            self.model_dir_combo.addItem(
                self._model_display_name(model_dir),
                str(model_dir),
            )

        if current_dir and current_dir.exists() and current_dir not in model_dirs:
            self.model_dir_combo.addItem(
                self._tf("custom_model", name=current_dir.name),
                str(current_dir),
            )

        if self.model_dir_combo.count() == 0:
            self.model_dir_combo.addItem(self._t("no_models_found"), "")
            self.model_dir_combo.setEnabled(False)
            self.model_path_edit.clear()
            self.model_dir_combo.blockSignals(False)
            return

        self.model_dir_combo.setEnabled(True)
        latest = Path(MODELS_DIR) / "latest"
        target = current_dir if current_dir else latest
        index = self.model_dir_combo.findData(str(target))
        if index < 0:
            index = 0
        self.model_dir_combo.setCurrentIndex(index)

        selected = self.model_dir_combo.currentData()
        if isinstance(selected, str):
            self.model_path_edit.setText(selected)

        self.model_dir_combo.blockSignals(False)

    def _on_model_selection_changed(self, _index: int) -> None:
        selected = self.model_dir_combo.currentData()
        if isinstance(selected, str) and selected:
            self.model_path_edit.setText(selected)
            
            # Auto-detect if this is an ensemble and update checkbox
            model_dir = Path(selected)
            is_ensemble = (model_dir / "model_0.pth").exists()
            has_single = (model_dir / "model.pth").exists()
            
            if is_ensemble and not has_single:
                if hasattr(self, "ensemble_checkbox"):
                    self.ensemble_checkbox.setChecked(True)
            elif has_single and not is_ensemble:
                if hasattr(self, "ensemble_checkbox"):
                    self.ensemble_checkbox.setChecked(False)

    def _filter_model_classes(self, text: str) -> None:
        query = text.strip().lower()
        if query:
            filtered = [
                class_name
                for class_name in self._all_model_classes
                if query in class_name.lower()
            ]
        else:
            filtered = list(self._all_model_classes)

        self.model_classes_list.clear()
        if filtered:
            self.model_classes_list.addItems(filtered)
        elif self._all_model_classes:
            self.model_classes_list.addItem(self._t("no_classes_match"))
        else:
            self.model_classes_list.addItem(self._t("model_not_loaded"))

        self._filtered_model_class_count = len(filtered)
        self.model_classes_header.setText(
            self._tf("model_classes_count", count=self._filtered_model_class_count)
        )

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
            None,
            self._t("select_model_directory"),
            str(self.project_root),
        )
        if chosen:
            self._set_selected_model_dir(Path(chosen), custom=True)

    def use_latest_model(self) -> None:
        latest_dir = Path(MODELS_DIR) / "latest"
        if latest_dir.exists():
            self._set_selected_model_dir(latest_dir)
            return
        self._set_status(self._t("latest_model_not_found"), "WARNING")

    def load_model_async(self) -> None:
        selected = self.model_dir_combo.currentData()
        model_path = (
            selected.strip()
            if isinstance(selected, str) and selected.strip()
            else self.model_path_edit.text().strip()
        )
        if not model_path:
            self._set_status(self._t("no_models_found"), "WARNING")
            return

        model_dir = Path(model_path)
        self.model_path_edit.setText(str(model_dir))
        self._set_status(self._t("model_loading_progress"), "INFO")
        self._set_model_badge(self._t("model_loading"), "loading")
        self._model_loaded = False
        self.load_btn.setEnabled(False)
        self.model_load_progress.setVisible(True)
        self._refresh_action_states()

        def _load() -> None:
            try:
                # ensemble_enabled is defined in Sign2SpeechDashboard (AppWindow)
                use_ensemble = getattr(self, "ensemble_enabled", False)
                metadata = self.model_service.load(model_dir, use_ensemble=use_ensemble)
                self.event_queue.put({"type": "model_loaded", "metadata": metadata})
                self.logger.info("Model loaded (ensemble=%s) from %s", use_ensemble, model_dir)
            except Exception as exc:
                self.logger.exception("Model load failed")
                self.event_queue.put(
                    {"type": "error", "message": f"Model load failed: {exc}"}
                )

        threading.Thread(target=_load, daemon=True).start()

    def _update_model_meta(self, metadata: ModelMetadata) -> None:
        self._all_model_classes = [str(label) for label in metadata.classes]
        self._set_selected_model_dir(metadata.model_dir)
        self.model_classes_value.setText(str(len(self._all_model_classes)))
        self.model_sequence_value.setText(str(metadata.sequence_length))
        self.model_input_value.setText(metadata.input_shape)
        self.model_loaded_value.setText(metadata.loaded_at)
        self.model_load_progress.setVisible(False)
        self.model_class_filter.clear()
        self._filter_model_classes("")
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
            serial_service=self.serial_service,
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

    def remove_last_word(self) -> None:
        if not self.current_sentence_tokens:
            self._set_status(self._t("no_words_to_remove"), "WARNING")
            return
        removed = self.current_sentence_tokens.pop()
        self.word_count_label.setText(self._format_word_count(len(self.current_sentence_tokens)))
        sentence_text = " ".join(self.current_sentence_tokens)
        if sentence_text:
            self.sentence_box.setPlainText(sentence_text)
        else:
            self.sentence_box.setPlainText(self._t("placeholder_sentence"))
        self._set_status(self._tf("word_removed", word=removed), "INFO")
        self._refresh_action_states()

    def request_llm_refinement(self) -> None:
        if not self.llm_enabled:
            self._set_status(self._t("llm_not_enabled"), "WARNING")
            return
        sentence = " ".join(self.current_sentence_tokens).strip()
        if not sentence:
            self._set_status(self._t("no_sentence_for_llm"), "WARNING")
            return
        self.event_queue.put({"type": "llm_request", "text": sentence})
        
        self.current_sentence_tokens.clear()
        self.word_count_label.setText(self._format_word_count(0))
        self.sentence_box.setPlainText(self._t("placeholder_sentence"))
        self.refined_box.setPlainText(self._t("placeholder_refined"))
        
        self._set_status(self._t("llm_request_sent"), "INFO")
        self._refresh_action_states()

    def export_sentence_text(self) -> None:
        sentence = " ".join(self.current_sentence_tokens)
        if not sentence.strip():
            self._set_status(self._t("no_sentence_to_export"), "WARNING")
            return

        target = export_sentence(sentence, Path(LOGS_OUTPUT_DIR))
        self._set_status(self._tf("sentence_exported", name=target.name), "INFO")
        self.logger.info("Sentence exported to %s", target)
