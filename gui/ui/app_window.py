"""Main PySide6 dashboard window."""

from __future__ import annotations

import os
import queue
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QMainWindow

from config.architecture import architecture
from config.config import LOGS_OUTPUT_DIR
from config.config import MODELS_DIR
from core.inference.gesture_translations import load_gesture_translations
from gui.services.llm_service import LLMService
from gui.services.logging_service import configure_gui_logger
from gui.services.model_service import ModelService
from gui.services.serial_service import SerialService
from gui.services.tts_service import TTSService
from gui.ui.app_window_actions import AppWindowActionsMixin
from gui.ui.app_window_events import AppWindowEventMixin
from gui.ui.app_window_layout import AppWindowLayoutMixin
from gui.ui.custom_widgets_adapter import apply_custom_widgets_theme
from gui.ui.localization import LOCALIZATION
from gui.ui.theme_manager import get_connection_badge_style
from gui.ui.theme_manager import get_model_badge_style
from gui.ui.theme_manager import get_status_banner_style
from gui.utils.formatting import percent
from gui.utils.icon_utils import apply_app_icon
from gui.utils.icon_utils import resolve_app_icon_path


class Sign2SpeechDashboard(
    AppWindowLayoutMixin,
    AppWindowActionsMixin,
    AppWindowEventMixin,
    QMainWindow,
):
    """Desktop dashboard for real-time sign prediction using PySide6."""

    STREAM_STARTUP_TIMEOUT_SECONDS = 8.0

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        apply_app_icon(self, self.project_root)
        self._i18n = LOCALIZATION
        self.ui_language = (
            architecture.gui.default_ui_language
            if architecture.gui.default_ui_language in architecture.gui.supported_ui_languages
            else "tr"
        )
        self.llm_language = "auto"
        self.setWindowTitle(self._t("title"))
        self.resize(1520, 900)
        self.setMinimumSize(1200, 700)

        self.event_queue: queue.Queue[dict] = queue.Queue()
        self.logger = configure_gui_logger(Path(LOGS_OUTPUT_DIR), self.event_queue)

        self.model_service = ModelService()
        self.serial_service = SerialService()
        self.llm_service = LLMService(event_queue=self.event_queue, logger=self.logger)
        self.worker = None
        self.tts_service = TTSService(logger=self.logger, event_queue=self.event_queue)
        self.tts_enabled = True
        self.tts_mode = "instant"
        self.ensemble_enabled = False
        self._tts_status_state = "waiting"
        self._tts_status_backend = "local"
        self._stream_started_at: float | None = None
        self._stream_stop_requested = False
        self._stream_had_error = False
        self._stream_connected = False
        self._stream_stop_message: str | None = None
        self._stream_stop_level = "INFO"
        self._active_stream_port: str | None = None
        self._stream_start_timeout_at: float | None = None
        self._stream_input_detected = False
        self._model_loaded = False
        self._last_confidence = 0.0

        self.current_sentence_tokens: list[str] = []
        self.model_dir = str(Path(MODELS_DIR) / "latest")
        self.llm_enabled = True
        self._llm_progress_state = "idle"
        self._llm_backend_state = "unknown"
        self._gesture_translations = load_gesture_translations()
        self._all_model_classes: list[str] = []
        self._filtered_model_class_count = 0
        self._theme_name = "dark"

        self._build_ui()
        self._build_shortcuts()
        self.refresh_ports()
        self._set_connection_badge(False)
        self._set_model_badge(self._t("model_not_ready"), "idle")
        self._refresh_action_states()

        self.use_latest_model()
        self.load_model_async()

        self.event_timer = QTimer(self)
        self.event_timer.setInterval(25)
        self.event_timer.timeout.connect(self._poll_events)
        self.event_timer.start()

        self.runtime_timer = QTimer(self)
        self.runtime_timer.setInterval(500)
        self.runtime_timer.timeout.connect(self._refresh_runtime_status)
        self.runtime_timer.start()

        self._apply_localization()

    def _t(self, key: str) -> str:
        table = self._i18n.get(self.ui_language, self._i18n["tr"])
        return table.get(key, key)

    def _tf(self, key: str, **kwargs: object) -> str:
        return self._t(key).format(**kwargs)

    def _format_runtime(self, value: str) -> str:
        return f"{self._t('runtime')}: {value}"

    def _format_confidence(self, value: float | str) -> str:
        if isinstance(value, str):
            formatted = value
        else:
            formatted = percent(value)
        return f"{self._t('confidence')}: {formatted}"

    def _format_word_count(self, count: int) -> str:
        return f"{self._t('word_count')}: {count}"

    def _format_llm_progress(self, state: str) -> str:
        state_map = {
            "idle": self._t("llm_state_idle"),
            "loading": self._t("llm_state_loading"),
            "generating": self._t("llm_state_generating"),
            "ready": self._t("llm_state_ready"),
            "error": self._t("llm_state_error"),
            "unavailable": self._t("llm_state_unavailable"),
            "disabled": self._t("llm_state_disabled"),
        }
        return f"{self._t('llm_progress')}: {state_map.get(state, state)}"

    def _format_llm_backend(self, backend: str) -> str:
        backend_map = {
            "gpu": self._t("llm_backend_gpu"),
            "cpu": self._t("llm_backend_cpu"),
            "unknown": self._t("llm_backend_unknown"),
        }
        return f"{self._t('llm_backend')}: {backend_map.get(backend, backend)}"

    def _set_llm_progress_state(self, state: str) -> None:
        self._llm_progress_state = state
        if hasattr(self, "llm_progress_label"):
            self.llm_progress_label.setText(self._format_llm_progress(state))

    def _set_llm_backend_state(self, backend: str) -> None:
        self._llm_backend_state = backend
        if hasattr(self, "llm_backend_label"):
            self.llm_backend_label.setText(self._format_llm_backend(backend))

    def _format_tts_status(self, state: str, backend: str) -> str:
        state_map = {
            "waiting": self._t("tts_state_waiting"),
            "working": self._t("tts_state_working"),
            "error": self._t("tts_state_error"),
        }
        backend_map = {
            "local": self._t("tts_backend_local"),
            "edge": self._t("tts_backend_edge"),
        }
        state_text = state_map.get(state, state)
        backend_text = backend_map.get(backend, self._t("tts_backend_unknown"))
        return self._tf("tts_status_format", backend=backend_text, state=state_text)

    def _set_tts_status_state(
        self,
        state: str,
        backend: str,
        message: str = "",
        update_banner: bool = True,
    ) -> None:
        self._tts_status_state = state
        self._tts_status_backend = backend

        if hasattr(self, "tts_status_value_label"):
            self.tts_status_value_label.setText(self._format_tts_status(state, backend))
            badge_state = "idle"
            if state == "working":
                badge_state = "loading"
            elif state == "error":
                badge_state = "error"
            self.tts_status_value_label.setStyleSheet(
                get_model_badge_style(badge_state, self._theme_name)
            )

        if update_banner:
            level = "ERROR" if state == "error" else "INFO"
            banner_text = (
                message.strip()
                if message.strip()
                else self._format_tts_status(state, backend)
            )
            self._set_status(banner_text, level)

    def _effective_llm_language(self) -> str:
        return self.ui_language if self.llm_language == "auto" else self.llm_language

    def _refresh_language_option_labels(self) -> None:
        self.ui_language_combo.setItemText(0, self._t("language_turkish"))
        self.ui_language_combo.setItemText(1, self._t("language_english"))
        self.llm_language_combo.setItemText(0, self._t("auto"))
        self.llm_language_combo.setItemText(1, self._t("language_turkish"))
        self.llm_language_combo.setItemText(2, self._t("language_english"))
        self.tts_mode_combo.setItemText(0, self._t("tts_mode_instant"))
        self.tts_mode_combo.setItemText(1, self._t("tts_mode_llm"))
        self.tts_mode_combo.setItemText(2, self._t("tts_mode_hybrid"))

    def _on_language_changed(self, _index: int) -> None:
        selected = self.ui_language_combo.currentData()
        language = str(selected) if isinstance(selected, str) else "tr"
        if language not in architecture.gui.supported_ui_languages or language == self.ui_language:
            return

        self.ui_language = language
        if self.worker and self.worker.is_alive():
            self.stop_stream(self._t("language_switch_stream_stopped"), "INFO")
        self.current_sentence_tokens.clear()
        self.sentence_box.setPlainText(self._t("placeholder_sentence"))
        self.refined_box.setPlainText(self._t("placeholder_refined"))
        self.prediction_card.setText(self._t("rest_label"))
        self._last_confidence = 0.0
        self._apply_localization()
        self._refresh_action_states()

    def _on_llm_language_changed(self, _index: int) -> None:
        selected = self.llm_language_combo.currentData()
        language = str(selected) if isinstance(selected, str) else "auto"
        if language not in {"auto", *architecture.gui.supported_ui_languages}:
            return
        self.llm_language = language

    def _set_connection_badge(self, connected: bool) -> None:
        if connected:
            self.connection_label.setText(self._t("device_connected"))
        else:
            self.connection_label.setText(self._t("device_disconnected"))
        self.connection_label.setStyleSheet(
            get_connection_badge_style(connected, self._theme_name)
        )

    def _set_model_badge(self, text: str, state: str) -> None:
        self.model_status_label.setText(text)
        self.model_status_label.setStyleSheet(
            get_model_badge_style(state, self._theme_name)
        )

    def _refresh_action_states(self) -> None:
        streaming = bool(self.worker and self.worker.is_alive())
        has_model = self._model_loaded
        has_sentence = bool(self.current_sentence_tokens)
        has_port = bool(self._selected_port())

        self.start_stop_btn.setEnabled(streaming or (has_model and has_port))
        self.copy_btn.setEnabled(has_sentence)
        self.export_btn.setEnabled(has_sentence)

    def _selected_port(self) -> str:
        data = self.port_combo.currentData()
        if isinstance(data, str):
            return data.strip()
        return ""

    def _set_status(self, message: str, level: str = "INFO") -> None:
        self.status_banner.setStyleSheet(
            get_status_banner_style(level, self._theme_name)
        )
        self.status_banner.setText(message)
        self.statusBar().showMessage(message, 3000)

    def _on_threshold_change(self, value: int) -> None:
        self.threshold_label.setText(f"{self._t('threshold')}: {value / 100:.2f}")

    def _on_smoothing_change(self, value: int) -> None:
        self.smoothing_label.setText(f"{self._t('smoothing')}: {value}")

    def _on_llm_changed(self, state: int) -> None:
        self.llm_enabled = state == 2
        if self.llm_enabled:
            self._set_llm_progress_state("idle")
        else:
            self._set_llm_progress_state("disabled")

    def _on_llm_backend_changed(self, _index: int) -> None:
        selected = self.llm_backend_combo.currentData()
        backend = str(selected) if isinstance(selected, str) else "local"
        if backend not in {"local", "remote"}:
            return
        os.environ["LLM_BACKEND"] = backend
        # Reload LLM backend on next request
        self.llm_service._llm = None
        self.llm_service._backend_meta = {}
        if self.llm_enabled:
            self.llm_service.preload_model()

    def _on_tts_changed(self, state: int) -> None:
        self.tts_enabled = state == 2
        self.tts_mode_label.setEnabled(self.tts_enabled)
        self.tts_mode_combo.setEnabled(self.tts_enabled)
        self.tts_status_label.setEnabled(self.tts_enabled)
        self.tts_status_value_label.setEnabled(self.tts_enabled)

    def _on_tts_mode_changed(self, _index: int) -> None:
        selected = self.tts_mode_combo.currentData()
        mode = str(selected) if isinstance(selected, str) else "instant"
        if mode not in {"instant", "llm", "hybrid"}:
            mode = "instant"
        self.tts_mode = mode

    def _on_ensemble_changed(self, state: int) -> None:
        self.ensemble_enabled = state == 2
        # If model is already loaded, we might want to reload it with/without ensemble
        if self._model_loaded:
            self.load_model_async()


def run_dashboard(project_root: Path) -> None:
    """Run the Sign Language dashboard application."""
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    if isinstance(app, QApplication):
        font = app.font()
        if font.pointSize() <= 0:
            safe_font = QFont(font)
            safe_font.setPointSize(10)
            app.setFont(safe_font)

    window = Sign2SpeechDashboard(project_root=project_root)
    icon_path = resolve_app_icon_path(project_root)
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    apply_app_icon(window, project_root)
    theme_loaded = apply_custom_widgets_theme(
        window,
        project_root,
    )
    if not theme_loaded:
        raise RuntimeError(
            "CustomWidgets theme could not be loaded for dashboard startup. "
            "Verify QT-PyQt-PySide-Custom-Widgets is installed and the style JSON is present."
        )
    window.show()

    if owns_app:
        app.exec()
