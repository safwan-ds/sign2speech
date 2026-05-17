"""UI construction and styling helpers for the dashboard window."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QListWidget,
    QScrollArea,
    QSlider,
    QStatusBar,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.config import (
    BAUD_RATE,
    CONFIDENCE_THRESHOLD,
    USE_QWEN_LLM,
    MOTION_DETECTION_SMOOTHING_WINDOW,
    USE_TTS,
)
from gui.ui.theme_manager import build_dashboard_stylesheet


class AppWindowLayoutMixin:
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

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        self.control_center_label = QLabel("")
        self.control_center_label.setObjectName("panelTitle")
        layout.addWidget(self.control_center_label)

        self.runtime_label = QLabel("")
        self.connection_label = QLabel("")
        self.model_status_label = QLabel("")
        layout.addWidget(self.runtime_label)
        layout.addWidget(self.connection_label)
        layout.addWidget(self.model_status_label)

        self.start_stop_btn = QPushButton("")
        self.start_stop_btn.clicked.connect(self.toggle_stream)
        self.start_stop_btn.setToolTip(self._t("tooltip_stream_toggle"))
        layout.addWidget(self.start_stop_btn)

        self.clear_btn = QPushButton("")
        self.clear_btn.clicked.connect(self.clear_sentence)
        self.clear_btn.setToolTip(self._t("tooltip_clear_sentence"))
        layout.addWidget(self.clear_btn)

        self.copy_btn = QPushButton("")
        self.copy_btn.clicked.connect(self.copy_sentence)
        layout.addWidget(self.copy_btn)

        self.export_btn = QPushButton("")
        self.export_btn.clicked.connect(self.export_sentence_text)
        self.export_btn.setToolTip(self._t("tooltip_export_sentence"))
        layout.addWidget(self.export_btn)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self.prediction_card = QLabel(self._t("rest_label"))
        self.prediction_card.setObjectName("predictionCard")
        self.prediction_card.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction_card)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        layout.addWidget(self.confidence_bar)

        stats = QHBoxLayout()
        self.confidence_label = QLabel("")
        self.word_count_label = QLabel("")
        stats.addWidget(self.confidence_label)
        stats.addStretch(1)
        stats.addWidget(self.word_count_label)
        layout.addLayout(stats)

        sentence_container = QWidget()
        sentence_layout = QVBoxLayout(sentence_container)
        sentence_layout.setContentsMargins(0, 0, 0, 0)
        sentence_layout.setSpacing(6)

        sentence_header = QHBoxLayout()
        self.sentence_header_label = QLabel("")
        sentence_header.addWidget(self.sentence_header_label)
        sentence_header.addStretch(1)
        self.sentence_copy_btn = QPushButton("")
        self.sentence_copy_btn.clicked.connect(self.copy_sentence)
        sentence_header.addWidget(self.sentence_copy_btn)
        sentence_layout.addLayout(sentence_header)

        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setPlainText("")
        sentence_layout.addWidget(self.sentence_box)

        refined_container = QWidget()
        refined_layout = QVBoxLayout(refined_container)
        refined_layout.setContentsMargins(0, 0, 0, 0)
        refined_layout.setSpacing(6)

        refined_header = QHBoxLayout()
        self.refined_header_label = QLabel("")
        refined_header.addWidget(self.refined_header_label)
        refined_header.addStretch(1)
        self.refined_copy_btn = QPushButton("")
        self.refined_copy_btn.clicked.connect(self.copy_refined)
        refined_header.addWidget(self.refined_copy_btn)
        refined_layout.addLayout(refined_header)

        self.llm_progress_label = QLabel("")
        refined_layout.addWidget(self.llm_progress_label)

        self.llm_backend_label = QLabel("")
        refined_layout.addWidget(self.llm_backend_label)

        self.refined_box = QTextEdit()
        self.refined_box.setReadOnly(True)
        self.refined_box.setPlainText("")
        refined_layout.addWidget(self.refined_box)

        layout.addWidget(sentence_container, stretch=1)
        layout.addWidget(refined_container, stretch=1)

        history_container = QWidget()
        history_layout = QVBoxLayout(history_container)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(6)
        self.history_label = QLabel("")
        history_layout.addWidget(self.history_label)

        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["", "", ""])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)

        layout.addWidget(history_container, stretch=1)

        return panel

    def _build_right_panel(self) -> QWidget:
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.settings_tab = self._build_settings_tab()
        self.logs_tab = self._build_logs_tab()
        self.right_tabs.addTab(self.settings_tab, "")
        self.right_tabs.addTab(self.logs_tab, "")
        return self.right_tabs

    def _build_settings_tab(self) -> QWidget:
        tab = QScrollArea()
        tab.setObjectName("settingsTab")
        tab.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        self.model_group = QGroupBox("")
        model_layout = QVBoxLayout(self.model_group)
        model_layout.setSpacing(8)

        self.model_picker_label = QLabel("")
        model_layout.addWidget(self.model_picker_label)

        picker_row = QHBoxLayout()
        self.model_dir_combo = QComboBox()
        self.model_dir_combo.currentIndexChanged.connect(
            self._on_model_selection_changed
        )
        picker_row.addWidget(self.model_dir_combo, stretch=1)

        self.refresh_models_btn = QPushButton("")
        self.refresh_models_btn.clicked.connect(self.refresh_model_dirs)
        picker_row.addWidget(self.refresh_models_btn)
        model_layout.addLayout(picker_row)

        self.model_path_label = QLabel("")
        model_layout.addWidget(self.model_path_label)

        self.model_path_edit = QLineEdit(self.model_dir)
        self.model_path_edit.setReadOnly(True)
        model_layout.addWidget(self.model_path_edit)

        model_btns = QHBoxLayout()
        self.model_browse_btn = QPushButton("")
        self.model_browse_btn.clicked.connect(self.select_model_dir)
        self.latest_btn = QPushButton("")
        self.latest_btn.clicked.connect(self.use_latest_model)
        self.load_btn = QPushButton("")
        self.load_btn.clicked.connect(self.load_model_async)
        model_btns.addWidget(self.model_browse_btn)
        model_btns.addWidget(self.latest_btn)
        model_btns.addWidget(self.load_btn)
        model_layout.addLayout(model_btns)

        self.model_load_progress = QProgressBar()
        self.model_load_progress.setObjectName("modelLoadProgress")
        self.model_load_progress.setRange(0, 0)
        self.model_load_progress.setTextVisible(False)
        self.model_load_progress.setVisible(False)
        self.model_load_progress.setFixedHeight(4)
        model_layout.addWidget(self.model_load_progress)

        model_stats = QGridLayout()
        model_stats.setHorizontalSpacing(6)
        model_stats.setVerticalSpacing(6)
        model_stats.setColumnStretch(0, 1)
        model_stats.setColumnStretch(1, 1)

        self.model_classes_card = QWidget()
        self.model_classes_card.setObjectName("modelMetricCard")
        classes_card_layout = QVBoxLayout(self.model_classes_card)
        classes_card_layout.setContentsMargins(10, 8, 10, 8)
        classes_card_layout.setSpacing(2)
        self.model_classes_stat_title = QLabel("")
        self.model_classes_stat_title.setObjectName("modelMetricLabel")
        self.model_classes_stat_title.setWordWrap(True)
        self.model_classes_value = QLabel("0")
        self.model_classes_value.setObjectName("modelMetricValue")
        self.model_classes_value.setMinimumHeight(20)
        classes_card_layout.addWidget(self.model_classes_stat_title)
        classes_card_layout.addWidget(self.model_classes_value)
        model_stats.addWidget(self.model_classes_card, 0, 0)

        self.model_sequence_card = QWidget()
        self.model_sequence_card.setObjectName("modelMetricCard")
        sequence_card_layout = QVBoxLayout(self.model_sequence_card)
        sequence_card_layout.setContentsMargins(10, 8, 10, 8)
        sequence_card_layout.setSpacing(2)
        self.model_sequence_stat_title = QLabel("")
        self.model_sequence_stat_title.setObjectName("modelMetricLabel")
        self.model_sequence_stat_title.setWordWrap(True)
        self.model_sequence_value = QLabel("--")
        self.model_sequence_value.setObjectName("modelMetricValue")
        self.model_sequence_value.setMinimumHeight(20)
        sequence_card_layout.addWidget(self.model_sequence_stat_title)
        sequence_card_layout.addWidget(self.model_sequence_value)
        model_stats.addWidget(self.model_sequence_card, 0, 1)

        self.model_input_card = QWidget()
        self.model_input_card.setObjectName("modelMetricCard")
        input_card_layout = QVBoxLayout(self.model_input_card)
        input_card_layout.setContentsMargins(10, 8, 10, 8)
        input_card_layout.setSpacing(2)
        self.model_input_stat_title = QLabel("")
        self.model_input_stat_title.setObjectName("modelMetricLabel")
        self.model_input_stat_title.setWordWrap(True)
        self.model_input_value = QLabel("--")
        self.model_input_value.setObjectName("modelMetricValue")
        self.model_input_value.setMinimumHeight(20)
        input_card_layout.addWidget(self.model_input_stat_title)
        input_card_layout.addWidget(self.model_input_value)
        model_stats.addWidget(self.model_input_card, 1, 0)

        self.model_loaded_card = QWidget()
        self.model_loaded_card.setObjectName("modelMetricCard")
        loaded_card_layout = QVBoxLayout(self.model_loaded_card)
        loaded_card_layout.setContentsMargins(10, 8, 10, 8)
        loaded_card_layout.setSpacing(2)
        self.model_loaded_stat_title = QLabel("")
        self.model_loaded_stat_title.setObjectName("modelMetricLabel")
        self.model_loaded_stat_title.setWordWrap(True)
        self.model_loaded_value = QLabel("--")
        self.model_loaded_value.setObjectName("modelMetricValue")
        self.model_loaded_value.setMinimumHeight(20)
        loaded_card_layout.addWidget(self.model_loaded_stat_title)
        loaded_card_layout.addWidget(self.model_loaded_value)
        model_stats.addWidget(self.model_loaded_card, 1, 1)

        model_layout.addLayout(model_stats)

        self.model_classes_header = QLabel("")
        model_layout.addWidget(self.model_classes_header)

        self.model_class_filter = QLineEdit()
        self.model_class_filter.textChanged.connect(self._filter_model_classes)
        model_layout.addWidget(self.model_class_filter)

        self.model_classes_list = QListWidget()
        self.model_classes_list.setObjectName("modelClassesList")
        self.model_classes_list.setAlternatingRowColors(True)
        self.model_classes_list.setMinimumHeight(120)
        model_layout.addWidget(self.model_classes_list, stretch=1)

        layout.addWidget(self.model_group)

        self.settings_group = QGroupBox("")
        settings_layout = QGridLayout(self.settings_group)

        self.serial_port_label = QLabel("")
        settings_layout.addWidget(self.serial_port_label, 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.currentTextChanged.connect(
            lambda _text: self._refresh_action_states()
        )
        settings_layout.addWidget(self.port_combo, 1, 0, 1, 2)

        self.refresh_ports_btn = QPushButton("")
        self.refresh_ports_btn.clicked.connect(self.refresh_ports)
        settings_layout.addWidget(self.refresh_ports_btn, 2, 0, 1, 2)

        self.baud_rate_label = QLabel("")
        settings_layout.addWidget(self.baud_rate_label, 3, 0)
        self.baud_edit = QLineEdit(str(BAUD_RATE))
        settings_layout.addWidget(self.baud_edit, 4, 0, 1, 2)

        self.threshold_label = QLabel("")
        settings_layout.addWidget(self.threshold_label, 5, 0, 1, 2)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 99)
        self.threshold_slider.setValue(int(CONFIDENCE_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        settings_layout.addWidget(self.threshold_slider, 6, 0, 1, 2)

        self.smoothing_label = QLabel("")
        settings_layout.addWidget(self.smoothing_label, 7, 0, 1, 2)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 12)
        self.smoothing_slider.setValue(MOTION_DETECTION_SMOOTHING_WINDOW)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_change)
        settings_layout.addWidget(self.smoothing_slider, 8, 0, 1, 2)

        self.llm_checkbox = QCheckBox("")
        self.llm_checkbox.setChecked(USE_QWEN_LLM)
        self.llm_checkbox.stateChanged.connect(self._on_llm_changed)
        settings_layout.addWidget(self.llm_checkbox, 9, 0, 1, 2)

        self.tts_checkbox = QCheckBox("")
        self.tts_checkbox.setChecked(USE_TTS)
        self.tts_checkbox.stateChanged.connect(self._on_tts_changed)
        settings_layout.addWidget(self.tts_checkbox, 10, 0, 1, 2)

        self.tts_mode_label = QLabel("")
        settings_layout.addWidget(self.tts_mode_label, 11, 0, 1, 2)
        self.tts_mode_combo = QComboBox()
        self.tts_mode_combo.addItem("", "instant")
        self.tts_mode_combo.addItem("", "llm")
        self.tts_mode_combo.addItem("", "hybrid")
        self.tts_mode_combo.currentIndexChanged.connect(self._on_tts_mode_changed)
        self.tts_mode_combo.setCurrentIndex(0)
        settings_layout.addWidget(self.tts_mode_combo, 12, 0, 1, 2)

        self.tts_status_label = QLabel("")
        settings_layout.addWidget(self.tts_status_label, 13, 0, 1, 2)
        self.tts_status_value_label = QLabel("")
        settings_layout.addWidget(self.tts_status_value_label, 14, 0, 1, 2)

        layout.addWidget(self.settings_group)
        self.refresh_model_dirs()
        layout.addStretch(1)
        tab.setWidget(content)
        return tab

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("logsTab")
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        self.clear_logs_btn = QPushButton("")
        self.clear_logs_btn.clicked.connect(self._clear_logs_view)
        self.export_logs_btn = QPushButton("")
        self.export_logs_btn.clicked.connect(self._export_logs_view)
        controls.addWidget(self.clear_logs_btn)
        controls.addWidget(self.export_logs_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return tab

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
        self.refined_copy_btn.setText(self._t("copy"))
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
        self.tts_checkbox.setText(self._t("enable_tts"))
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
