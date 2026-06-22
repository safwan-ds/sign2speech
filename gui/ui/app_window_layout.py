"""UI construction and styling helpers for the dashboard window.

This module composes sub-mixins for building panels, settings, and logs tabs
into the main AppWindowLayoutMixin used by the dashboard window.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gui.ui.app_window_panels import AppWindowPanelsMixin
from gui.ui.app_window_settings_tab import AppWindowSettingsTabMixin
from gui.ui.theme_manager import build_dashboard_stylesheet


class AppWindowLayoutMixin(AppWindowPanelsMixin, AppWindowSettingsTabMixin):
    """Build and localize the main dashboard window widgets."""

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("centralRoot")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        self.title_label = QLabel(self._t("title"))
        self.title_label.setObjectName("title")
        self.subtitle_label = QLabel("")
        self.subtitle_label.setObjectName("subtitle")

        header = QVBoxLayout()
        header.addWidget(self.title_label)
        header.addWidget(self.subtitle_label)
        outer.addLayout(header)

        top_controls = QHBoxLayout()
        top_controls.addStretch(1)

        self.top_language_label = QLabel("")
        top_controls.addWidget(self.top_language_label)
        self.ui_language_combo = QComboBox()
        self.ui_language_combo.addItem(self._t("language_turkish"), "tr")
        self.ui_language_combo.addItem(self._t("language_english"), "en")
        current_ui_idx = self.ui_language_combo.findData(self.ui_language)
        self.ui_language_combo.setCurrentIndex(
            current_ui_idx if current_ui_idx >= 0 else 0
        )
        self.ui_language_combo.currentIndexChanged.connect(self._on_language_changed)
        top_controls.addWidget(self.ui_language_combo)

        self.top_llm_language_label = QLabel("")
        top_controls.addWidget(self.top_llm_language_label)
        self.llm_language_combo = QComboBox()
        self.llm_language_combo.addItem(self._t("auto"), "auto")
        self.llm_language_combo.addItem(self._t("language_turkish"), "tr")
        self.llm_language_combo.addItem(self._t("language_english"), "en")
        current_llm_idx = self.llm_language_combo.findData(self.llm_language)
        self.llm_language_combo.setCurrentIndex(
            current_llm_idx if current_llm_idx >= 0 else 0
        )
        self.llm_language_combo.currentIndexChanged.connect(
            self._on_llm_language_changed
        )
        top_controls.addWidget(self.llm_language_combo)

        outer.addLayout(top_controls)

        self.status_banner = QLabel("")
        self.status_banner.setObjectName("statusInfo")
        self.status_banner.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        outer.addWidget(self.status_banner)

        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, stretch=1)

        body.addWidget(self._build_left_panel(), stretch=1)
        body.addWidget(self._build_center_panel(), stretch=3)
        body.addWidget(self._build_right_panel(), stretch=2)

        status = QStatusBar(self)
        self.setStatusBar(status)

        self._apply_window_styles(getattr(self, "_theme_name", "dark"))

    def _apply_window_styles(self, theme: str) -> None:
        self.setStyleSheet(build_dashboard_stylesheet(theme))

    def _build_shortcuts(self) -> None:
        parent = cast(QWidget, self)

        start_action = QAction("", parent)
        start_action.setShortcut(QKeySequence("Ctrl+S"))
        start_action.triggered.connect(self.toggle_stream)
        self.addAction(start_action)

        clear_action = QAction("", parent)
        clear_action.setShortcut(QKeySequence("Ctrl+L"))
        clear_action.triggered.connect(self.clear_sentence)
        self.addAction(clear_action)

        export_action = QAction("", parent)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_sentence_text)
        self.addAction(export_action)

        copy_action = QAction("", parent)
        copy_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        copy_action.triggered.connect(self.copy_sentence)
        self.addAction(copy_action)

        remove_last_action = QAction("", parent)
        remove_last_action.setShortcut(QKeySequence("Z"))
        remove_last_action.triggered.connect(self.remove_last_word)
        self.addAction(remove_last_action)

        refine_action = QAction("", parent)
        refine_action.setShortcut(QKeySequence("R"))
        refine_action.triggered.connect(self.request_llm_refinement)
        self.addAction(refine_action)

    def _apply_localization(self) -> None:
        self._refresh_language_option_labels()
        self.setWindowTitle(self._t("title"))
        self.title_label.setText(self._t("title"))
        self.subtitle_label.setText(self._t("subtitle"))
        self.status_banner.setText(self._t("ready"))

        self.control_center_label.setText(self._t("control_center"))
        self.runtime_label.setText(self._format_runtime("00:00"))
        self.start_stop_btn.setText(
            self._t("stop_stream")
            if self.worker and self.worker.is_alive()
            else self._t("start_stream")
        )
        self.clear_btn.setText(self._t("clear_sentence"))
        self.copy_btn.setText(self._t("copy_sentence"))
        self.export_btn.setText(self._t("export_text"))

        self.confidence_label.setText(self._format_confidence(self._last_confidence))
        self.word_count_label.setText(
            self._format_word_count(len(self.current_sentence_tokens))
        )
        self.sentence_header_label.setText(self._t("translated_sentence"))
        self.refined_header_label.setText(self._t("refined_sentence"))
        self.llm_progress_label.setText(
            self._format_llm_progress(self._llm_progress_state)
        )
        self.llm_backend_label.setText(
            self._format_llm_backend(self._llm_backend_state)
        )
        self.sentence_copy_btn.setText(self._t("copy"))
        self.sentence_remove_last_btn.setText(self._t("remove_last"))
        self.sentence_clear_btn.setText(self._t("clear"))

        self.refined_copy_btn.setText(self._t("copy"))
        self.refined_refine_btn.setText(self._t("refine"))

        self.history_label.setText(self._t("history"))
        self.history_table.setHorizontalHeaderLabels(
            [self._t("table_time"), self._t("table_class"), self._t("table_confidence")]
        )
        if not self.current_sentence_tokens:
            self.sentence_box.setPlainText(self._t("placeholder_sentence"))
        if not self.refined_box.toPlainText().strip():
            self.refined_box.setPlainText(self._t("placeholder_refined"))

        self.right_tabs.setTabText(0, self._t("tab_model_port"))
        self.right_tabs.setTabText(1, self._t("tab_logs"))

        self.model_group.setTitle(self._t("group_model"))
        self.settings_group.setTitle(self._t("group_settings"))

        self.top_language_label.setText(self._t("language"))
        self.top_llm_language_label.setText(self._t("llm_language"))
        self.serial_port_label.setText(self._t("port"))
        self.refresh_ports_btn.setText(self._t("refresh_ports"))
        self.baud_rate_label.setText(self._t("baud_rate"))
        self.baud_edit.setPlaceholderText(self._t("baud_placeholder"))
        self._on_threshold_change(self.threshold_slider.value())
        self._on_smoothing_change(self.smoothing_slider.value())
        self.llm_checkbox.setText(self._t("enable_llm"))
        self.llm_backend_label.setText(self._t("llm_backend"))
        self.llm_backend_combo.setItemText(0, self._t("llm_backend_local"))
        self.llm_backend_combo.setItemText(1, self._t("llm_backend_remote"))
        self.tts_checkbox.setText(self._t("enable_tts"))
        self.ensemble_checkbox.setText(self._t("enable_ensemble"))
        self.tts_mode_label.setText(self._t("tts_mode"))
        self.tts_status_label.setText(self._t("tts_status"))
        self._set_tts_status_state(
            self._tts_status_state,
            self._tts_status_backend,
            update_banner=False,
        )
        self.model_picker_label.setText(self._t("available_models"))
        self.model_path_label.setText(self._t("model_path"))
        self.refresh_models_btn.setText(self._t("refresh_models"))
        self.model_browse_btn.setText(self._t("browse"))
        self.latest_btn.setText(self._t("latest"))
        self.load_btn.setText(self._t("load"))
        self.model_classes_stat_title.setText(self._t("model_classes"))
        self.model_sequence_stat_title.setText(self._t("sequence_length"))
        self.model_input_stat_title.setText(self._t("input_shape"))
        self.model_loaded_stat_title.setText(self._t("loaded_at"))
        self.model_classes_header.setText(
            self._tf("model_classes_count", count=self._filtered_model_class_count)
        )
        self.model_class_filter.setPlaceholderText(
            self._t("filter_classes_placeholder")
        )
        if self._all_model_classes:
            self._filter_model_classes(self.model_class_filter.text())
        else:
            self._filtered_model_class_count = 0
            self.model_classes_header.setText(
                self._tf("model_classes_count", count=self._filtered_model_class_count)
            )
            self.model_classes_list.clear()
            self.model_classes_list.addItem(self._t("model_not_loaded"))
        self.refresh_model_dirs()

        self.clear_logs_btn.setText(self._t("clear_logs"))
        self.export_logs_btn.setText(self._t("export_logs"))

        self._set_connection_badge(self._stream_connected)
        if self._model_loaded:
            self._set_model_badge(self._t("model_ready"), "ready")
        else:
            self._set_model_badge(self._t("model_not_ready"), "idle")
