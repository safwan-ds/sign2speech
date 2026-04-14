"""Main PySide6 dashboard window."""

from __future__ import annotations

import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSlider,
    QStatusBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    BAUD_RATE,
    DEFAULT_UI_LANGUAGE,
    LOGS_DIR,
    LOGS_OUTPUT_DIR,
    MODELS_DIR,
    SUPPORTED_UI_LANGUAGES,
)
from core.inference.gesture_translations import (
    load_gesture_translations,
    translate_gesture,
)
from gui.services.llm_service import LLMService
from gui.services.logging_service import configure_gui_logger
from gui.services.model_service import ModelMetadata, ModelService
from gui.services.script_runner import ScriptRunner
from gui.services.serial_service import SerialService, SerialSettings
from gui.services.stream_service import StreamConfig, StreamWorker
from gui.services.tts_service import TTSService
from gui.ui.localization import LOCALIZATION
from gui.utils.exporter import export_sentence
from gui.utils.formatting import now_stamp, percent
from utils.serial_utils import select_serial_port


class SignLanguageDashboard(QMainWindow):
    """Desktop dashboard for real-time sign prediction using PySide6."""

    STREAM_STARTUP_TIMEOUT_SECONDS = 8.0

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.setWindowTitle(self._t("title"))
        self.resize(1520, 900)
        self.setMinimumSize(1200, 700)

        self.event_queue: queue.Queue[dict] = queue.Queue()
        self.logger = configure_gui_logger(Path(LOGS_OUTPUT_DIR), self.event_queue)

        self.model_service = ModelService()
        self.script_runner = ScriptRunner(
            project_root=self.project_root,
            logger=self.logger,
        )
        self.llm_service = LLMService(event_queue=self.event_queue, logger=self.logger)
        self.worker: StreamWorker | None = None
        self.tts_service = TTSService(logger=self.logger)
        self.tts_enabled = True
        self.tts_mode = "instant"
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
        self.ui_language = (
            DEFAULT_UI_LANGUAGE
            if DEFAULT_UI_LANGUAGE in SUPPORTED_UI_LANGUAGES
            else "tr"
        )
        self.llm_language = "auto"
        self._gesture_translations = load_gesture_translations()
        self._i18n = LOCALIZATION

        self._recording_active = False
        self._recording_stop_event = threading.Event()
        self._recording_thread: threading.Thread | None = None
        self._recording_started_at: float | None = None
        self._recording_serial = SerialService()
        self._recording_countdown_active = False
        self._recording_countdown_remaining = 0
        self._recording_pending_request: tuple[str, str, int] | None = None
        self._recording_countdown_timer = QTimer(self)
        self._recording_countdown_timer.setInterval(1000)
        self._recording_countdown_timer.timeout.connect(
            self._on_recording_countdown_tick
        )

        self._build_ui()
        self._build_shortcuts()
        self.refresh_ports()
        self._load_gesture_options()
        self._set_connection_badge(False)
        self._set_model_badge(self._t("model_not_ready"), "idle")
        self._refresh_action_states()

        # Load models/latest automatically so streaming is ready sooner.
        self.use_latest_model()
        self.load_model_async()

        self.event_timer = QTimer(self)
        self.event_timer.setInterval(80)
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

    def _format_confidence(self, value: str) -> str:
        return f"{self._t('confidence')}: {value}"

    def _format_word_count(self, count: int) -> str:
        return f"{self._t('word_count')}: {count}"

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

        self._apply_window_styles("dark")

    def _apply_window_styles(self, theme: str) -> None:
        palettes: dict[str, dict[str, str]] = {
            "light": {
                "bg": "#f6f7fb",
                "panel": "#ffffff",
                "text": "#1f2937",
                "subtext": "#5a6777",
                "input_bg": "#ffffff",
                "input_border": "#d1d5db",
                "button_bg": "#f3f4f6",
                "button_hover": "#e5e7eb",
                "group_border": "#d1d5db",
                "prediction_bg": "#eef2ff",
                "prediction_text": "#13264d",
            },
            "dark": {
                "bg": "#1f2329",
                "panel": "#2b3139",
                "text": "#e5e7eb",
                "subtext": "#b3bcc9",
                "input_bg": "#242a31",
                "input_border": "#3d4651",
                "button_bg": "#353d48",
                "button_hover": "#46505d",
                "group_border": "#4b5563",
                "prediction_bg": "#2f3540",
                "prediction_text": "#f3f4f6",
            },
        }

        palette = palettes.get(theme, palettes["light"])
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget#centralRoot {{ background: {palette['bg']}; }}
            QWidget {{ color: {palette['text']}; }}
            QWidget#centralRoot {{ background: {palette['bg']}; }}
            QTabWidget::pane, QGroupBox, QPlainTextEdit, QTextEdit, QTableWidget, QStatusBar {{ background: {palette['panel']}; }}
            QTabWidget#rightTabs {{ background: {palette['panel']}; }}
            QWidget#settingsTab, QWidget#actionsTab, QWidget#logsTab {{ background: {palette['panel']}; }}
            QTabWidget#rightTabs QGroupBox {{ background: {palette['panel']}; }}
            QTabBar::tab {{
                background: {palette['button_bg']};
                border: 1px solid {palette['input_border']};
                border-bottom: none;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {palette['panel']};
                color: {palette['text']};
            }}
            QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QTableWidget {{
                background: {palette['input_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QHeaderView::section {{
                background: {palette['button_bg']};
                color: {palette['text']};
                border: 1px solid {palette['input_border']};
                padding: 4px;
            }}
            QPushButton {{
                background: {palette['button_bg']};
                border: 1px solid {palette['input_border']};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{ background: {palette['button_hover']}; }}
            QLabel#title {{ font-size: 30px; font-weight: 700; }}
            QLabel#subtitle {{ color: {palette['subtext']}; }}
            QLabel#statusInfo {{ padding: 8px 10px; border-radius: 6px; background: #e9f5ec; color: #12381f; }}
            QLabel#predictionCard {{
                font-size: 64px;
                font-weight: 700;
                border-radius: 12px;
                padding: 20px;
                background: {palette['prediction_bg']};
                color: {palette['prediction_text']};
            }}
            QLabel#panelTitle {{ font-size: 20px; font-weight: 700; }}
            QGroupBox {{ font-weight: 600; margin-top: 8px; border: 1px solid {palette['group_border']}; border-radius: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; }}
            """
        )

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

        text_boxes_splitter = QSplitter(Qt.Orientation.Vertical)

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

        self.refined_box = QTextEdit()
        self.refined_box.setReadOnly(True)
        self.refined_box.setPlainText("")
        refined_layout.addWidget(self.refined_box)

        text_boxes_splitter.addWidget(sentence_container)
        text_boxes_splitter.addWidget(refined_container)
        text_boxes_splitter.setStretchFactor(0, 1)
        text_boxes_splitter.setStretchFactor(1, 1)
        text_boxes_splitter.setChildrenCollapsible(False)
        text_boxes_splitter.setSizes([220, 220])

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

        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.addWidget(text_boxes_splitter)
        content_splitter.addWidget(history_container)
        content_splitter.setStretchFactor(0, 2)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setSizes([460, 260])

        layout.addWidget(content_splitter, stretch=1)

        return panel

    def _build_right_panel(self) -> QWidget:
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.settings_tab = self._build_settings_tab()
        self.actions_tab = self._build_actions_tab()
        self.logs_tab = self._build_logs_tab()
        self.right_tabs.addTab(self.settings_tab, "")
        self.right_tabs.addTab(self.actions_tab, "")
        self.right_tabs.addTab(self.logs_tab, "")
        return self.right_tabs

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("settingsTab")
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.model_group = QGroupBox("")
        model_layout = QVBoxLayout(self.model_group)

        self.model_path_edit = QLineEdit(self.model_dir)
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

        self.model_meta_label = QLabel("")
        self.model_meta_label.setWordWrap(True)
        model_layout.addWidget(self.model_meta_label)

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
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        settings_layout.addWidget(self.threshold_slider, 6, 0, 1, 2)

        self.smoothing_label = QLabel("")
        settings_layout.addWidget(self.smoothing_label, 7, 0, 1, 2)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 12)
        self.smoothing_slider.setValue(5)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_change)
        settings_layout.addWidget(self.smoothing_slider, 8, 0, 1, 2)

        self.llm_checkbox = QCheckBox("")
        self.llm_checkbox.setChecked(True)
        self.llm_checkbox.stateChanged.connect(self._on_llm_changed)
        settings_layout.addWidget(self.llm_checkbox, 9, 0, 1, 2)

        self.tts_checkbox = QCheckBox("")
        self.tts_checkbox.setChecked(True)
        self.tts_checkbox.stateChanged.connect(self._on_tts_changed)
        settings_layout.addWidget(self.tts_checkbox, 10, 0, 1, 2)

        self.tts_mode_label = QLabel("")
        settings_layout.addWidget(self.tts_mode_label, 11, 0, 1, 2)
        self.tts_mode_combo = QComboBox()
        self.tts_mode_combo.addItem("", "instant")
        self.tts_mode_combo.addItem("", "llm")
        self.tts_mode_combo.currentIndexChanged.connect(self._on_tts_mode_changed)
        self.tts_mode_combo.setCurrentIndex(0)
        settings_layout.addWidget(self.tts_mode_combo, 12, 0, 1, 2)

        layout.addWidget(self.settings_group)
        layout.addStretch(1)
        return tab

    def _build_actions_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("actionsTab")
        layout = QVBoxLayout(tab)

        self.script_group = QGroupBox("")
        script_layout = QVBoxLayout(self.script_group)

        self.process_btn = QPushButton("")
        self.process_btn.clicked.connect(
            lambda: self.run_script("scripts/process_data.py")
        )
        self.train_btn = QPushButton("")
        self.train_btn.clicked.connect(
            lambda: self.run_script("scripts/train_model.py")
        )
        self.predict_btn = QPushButton("")
        self.predict_btn.clicked.connect(lambda: self.run_script("scripts/predict.py"))

        script_layout.addWidget(self.process_btn)
        script_layout.addWidget(self.train_btn)
        script_layout.addWidget(self.predict_btn)
        layout.addWidget(self.script_group)

        self.record_group = QGroupBox("")
        record_layout = QVBoxLayout(self.record_group)

        self.gesture_label = QLabel("")
        record_layout.addWidget(self.gesture_label)
        self.record_gesture_combo = QComboBox()
        record_layout.addWidget(self.record_gesture_combo)

        self.refresh_gesture_btn = QPushButton("")
        self.refresh_gesture_btn.clicked.connect(self._load_gesture_options)
        record_layout.addWidget(self.refresh_gesture_btn)

        row = QHBoxLayout()
        self.record_start_btn = QPushButton("")
        self.record_start_btn.clicked.connect(
            self._start_manual_recording_with_countdown
        )
        self.record_stop_btn = QPushButton("")
        self.record_stop_btn.clicked.connect(self.stop_manual_recording)
        self.record_stop_btn.setEnabled(False)
        row.addWidget(self.record_start_btn)
        row.addWidget(self.record_stop_btn)
        record_layout.addLayout(row)

        self.recording_state_label = QLabel("")
        record_layout.addWidget(self.recording_state_label)

        self.recording_helper_label = QLabel("")
        self.recording_helper_label.setWordWrap(True)
        record_layout.addWidget(self.recording_helper_label)

        layout.addWidget(self.record_group)
        layout.addStretch(1)
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
        start_action = QAction(self)
        start_action.setShortcut(QKeySequence("Ctrl+S"))
        start_action.triggered.connect(self.toggle_stream)
        self.addAction(start_action)

        clear_action = QAction(self)
        clear_action.setShortcut(QKeySequence("Ctrl+L"))
        clear_action.triggered.connect(self.clear_sentence)
        self.addAction(clear_action)

        export_action = QAction(self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_sentence_text)
        self.addAction(export_action)

        copy_action = QAction(self)
        copy_action.setShortcut(QKeySequence("Ctrl+C"))
        copy_action.triggered.connect(self.copy_sentence)
        self.addAction(copy_action)

        start_recording_action = QAction(self)
        start_recording_action.setShortcut(QKeySequence("Ctrl+R"))
        start_recording_action.triggered.connect(self.start_manual_recording)
        self.addAction(start_recording_action)

        stop_recording_action = QAction(self)
        stop_recording_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        stop_recording_action.triggered.connect(self.stop_manual_recording)
        self.addAction(stop_recording_action)

    def _set_connection_badge(self, connected: bool) -> None:
        if connected:
            self.connection_label.setText(self._t("device_connected"))
            self.connection_label.setStyleSheet(
                "padding: 6px 8px; border-radius: 6px; background: #e7f6ea; color: #1e6c35;"
            )
        else:
            self.connection_label.setText(self._t("device_disconnected"))
            self.connection_label.setStyleSheet(
                "padding: 6px 8px; border-radius: 6px; background: #f4ecec; color: #7d1f1f;"
            )

    def _set_model_badge(self, text: str, state: str) -> None:
        palette = {
            "idle": "padding: 6px 8px; border-radius: 6px; background: #fff4e5; color: #7d5200;",
            "loading": "padding: 6px 8px; border-radius: 6px; background: #e8f0ff; color: #1f4a86;",
            "ready": "padding: 6px 8px; border-radius: 6px; background: #e7f6ea; color: #1e6c35;",
            "error": "padding: 6px 8px; border-radius: 6px; background: #fdecea; color: #8a1c1c;",
        }
        self.model_status_label.setText(text)
        self.model_status_label.setStyleSheet(palette.get(state, palette["idle"]))

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

        self.confidence_label.setText(
            self._format_confidence(percent(self._last_confidence))
        )
        self.word_count_label.setText(
            self._format_word_count(len(self.current_sentence_tokens))
        )
        self.sentence_header_label.setText(self._t("translated_sentence"))
        self.refined_header_label.setText(self._t("refined_sentence"))
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
        self.right_tabs.setTabText(1, self._t("tab_actions"))
        self.right_tabs.setTabText(2, self._t("tab_logs"))

        self.model_group.setTitle(self._t("group_model"))
        self.settings_group.setTitle(self._t("group_settings"))
        self.script_group.setTitle(self._t("group_helpers"))
        self.record_group.setTitle(self._t("group_recording"))

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
        self.model_browse_btn.setText(self._t("browse"))
        self.latest_btn.setText(self._t("latest"))
        self.load_btn.setText(self._t("load"))
        if not self.model_meta_label.text().strip():
            self.model_meta_label.setText(self._t("model_not_loaded"))

        self.process_btn.setText(self._t("process_data"))
        self.train_btn.setText(self._t("train_model"))
        self.predict_btn.setText(self._t("run_predict"))
        self.gesture_label.setText(self._t("gesture"))
        self.refresh_gesture_btn.setText(self._t("refresh_gestures"))
        self.record_start_btn.setText(self._t("start_recording"))
        self.record_stop_btn.setText(self._t("stop_recording"))
        self.recording_helper_label.setText(self._t("recording_hint"))
        self.clear_logs_btn.setText(self._t("clear_logs"))
        self.export_logs_btn.setText(self._t("export_logs"))

        if self.recording_state_label.text().strip() in {"", "Hazır", "Ready"}:
            self.recording_state_label.setText(self._t("recording_ready"))

        self._set_connection_badge(self._stream_connected)
        if self._model_loaded:
            self._set_model_badge(self._t("model_ready"), "ready")
        else:
            self._set_model_badge(self._t("model_not_ready"), "idle")

    def _on_language_changed(self, _index: int) -> None:
        selected = self.ui_language_combo.currentData()
        language = str(selected) if isinstance(selected, str) else "tr"
        if language not in SUPPORTED_UI_LANGUAGES or language == self.ui_language:
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
        self._load_gesture_options()
        self._refresh_action_states()

    def _on_llm_language_changed(self, _index: int) -> None:
        selected = self.llm_language_combo.currentData()
        language = str(selected) if isinstance(selected, str) else "auto"
        if language not in {"auto", *SUPPORTED_UI_LANGUAGES}:
            return
        self.llm_language = language

    def _refresh_action_states(self) -> None:
        streaming = bool(self.worker and self.worker.is_alive())
        has_model = self._model_loaded
        has_sentence = bool(self.current_sentence_tokens)
        has_port = bool(self._selected_port())
        has_gesture = bool(self._selected_recording_gesture())

        self.start_stop_btn.setEnabled(
            streaming or (has_model and has_port and not self._recording_active)
        )
        self.copy_btn.setEnabled(has_sentence)
        self.export_btn.setEnabled(has_sentence)
        self.record_start_btn.setEnabled(
            (not self._recording_active)
            and (not self._recording_countdown_active)
            and (not streaming)
            and has_port
            and has_gesture
        )
        self.record_stop_btn.setEnabled(
            self._recording_active or self._recording_countdown_active
        )

    def _selected_port(self) -> str:
        data = self.port_combo.currentData()
        if isinstance(data, str):
            return data.strip()
        return ""

    def _selected_recording_gesture(self) -> str:
        data = self.record_gesture_combo.currentData()
        if isinstance(data, str):
            return data.strip()
        return ""

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

    def _set_status(self, message: str, level: str = "INFO") -> None:
        if level == "ERROR":
            self.status_banner.setStyleSheet(
                "padding: 8px 10px; border-radius: 6px; background: #fdecea; color: #8a1c1c;"
            )
        elif level == "WARNING":
            self.status_banner.setStyleSheet(
                "padding: 8px 10px; border-radius: 6px; background: #fff4e5; color: #7d5200;"
            )
        else:
            self.status_banner.setStyleSheet(
                "padding: 8px 10px; border-radius: 6px; background: #e9f5ec; color: #12381f;"
            )
        self.status_banner.setText(message)
        self.statusBar().showMessage(message, 3000)

    def _on_threshold_change(self, value: int) -> None:
        self.threshold_label.setText(f"{self._t('threshold')}: {value / 100:.2f}")

    def _on_smoothing_change(self, value: int) -> None:
        self.smoothing_label.setText(f"{self._t('smoothing')}: {value}")

    def _on_llm_changed(self, state: int) -> None:
        self.llm_enabled = state == Qt.CheckState.Checked.value

    def _on_tts_changed(self, state: int) -> None:
        self.tts_enabled = state == Qt.CheckState.Checked.value
        self.tts_mode_label.setEnabled(self.tts_enabled)
        self.tts_mode_combo.setEnabled(self.tts_enabled)

    def _on_tts_mode_changed(self, _index: int) -> None:
        selected = self.tts_mode_combo.currentData()
        mode = str(selected) if isinstance(selected, str) else "instant"
        if mode not in {"instant", "llm"}:
            mode = "instant"
        self.tts_mode = mode

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

    def _load_gesture_options(self) -> None:
        labels: list[tuple[str, str]] = []
        seen: set[str] = set()
        gestures_file = self.project_root / "config" / "gestures.txt"
        if gestures_file.exists():
            with gestures_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if " - " in line:
                        english_name, turkish_name = [
                            part.strip() for part in line.split(" - ", 1)
                        ]
                    else:
                        english_name, turkish_name = line, line
                    if english_name and english_name not in seen:
                        seen.add(english_name)
                        display_name = (
                            turkish_name if self.ui_language == "tr" else english_name
                        )
                        labels.append((display_name, english_name))

        raw_dir = Path(LOGS_DIR)
        if raw_dir.exists():
            extra_labels = sorted(
                [child.name for child in raw_dir.iterdir() if child.is_dir()],
                key=str.lower,
            )
            for name in extra_labels:
                if name not in seen:
                    seen.add(name)
                    labels.append((name, name))

        if not labels:
            labels = [
                (
                    self._t("no_gestures_found"),
                    "",
                )
            ]

        current = self._selected_recording_gesture()
        self.record_gesture_combo.clear()
        for display_name, gesture_key in labels:
            self.record_gesture_combo.addItem(display_name, gesture_key)

        idx = self.record_gesture_combo.findData(current)
        if idx >= 0:
            self.record_gesture_combo.setCurrentIndex(idx)
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

    def run_script(self, relative_script_path: str) -> None:
        ok = self.script_runner.run_script(relative_script_path)
        if not ok:
            self._set_status(self._t("script_already_running"), "WARNING")

    def _resolve_manual_recording_request(self) -> tuple[str, str, int] | None:
        if self.worker and self.worker.is_alive():
            self._set_status(self._t("stop_stream_before_recording"), "WARNING")
            return None
        if self._recording_active:
            self._set_status(self._t("recording_already_in_progress"), "WARNING")
            return None

        if self._recording_countdown_active:
            self._set_status(self._t("recording_countdown_in_progress"), "WARNING")
            return None

        gesture = self._selected_recording_gesture()
        if not gesture:
            self._set_status(self._t("select_gesture_before_recording"), "WARNING")
            return None

        port = self._selected_port()
        if not port:
            self._set_status(self._t("select_port_before_recording"), "WARNING")
            return None

        try:
            baud = int(self.baud_edit.text().strip())
        except ValueError:
            self._set_status(self._t("baud_must_be_numeric"), "WARNING")
            return None

        return (gesture, port, baud)

    def _start_manual_recording_with_countdown(self) -> None:
        request = self._resolve_manual_recording_request()
        if request is None:
            return

        self._recording_pending_request = request
        self._recording_countdown_active = True
        self._recording_countdown_remaining = 3
        self.recording_state_label.setText(
            self._tf("recording_starts_in_seconds", seconds=3)
        )
        self._set_status(self._tf("recording_starts_in_seconds", seconds=3), "INFO")
        self._refresh_action_states()
        self._recording_countdown_timer.start()

    def _on_recording_countdown_tick(self) -> None:
        if not self._recording_countdown_active:
            self._recording_countdown_timer.stop()
            return

        self._recording_countdown_remaining -= 1
        if self._recording_countdown_remaining <= 0:
            self._recording_countdown_timer.stop()
            self._recording_countdown_active = False
            request = self._recording_pending_request
            self._recording_pending_request = None
            self._refresh_action_states()
            if request is not None:
                self._start_manual_recording_core(*request)
            return

        self.recording_state_label.setText(
            self._tf(
                "recording_starts_in_seconds",
                seconds=self._recording_countdown_remaining,
            )
        )

    def start_manual_recording(self) -> None:
        request = self._resolve_manual_recording_request()
        if request is None:
            return
        self._start_manual_recording_core(*request)

    def _start_manual_recording_core(self, gesture: str, port: str, baud: int) -> None:
        try:
            self._recording_serial.connect(
                SerialSettings(port=port, baud_rate=baud, timeout=0.1)
            )
        except Exception as exc:
            self._set_status(
                self._tf("serial_connection_open_failed", error=exc), "ERROR"
            )
            return

        self._recording_stop_event.clear()
        self._recording_started_at = time.perf_counter()
        self._recording_active = True
        self.record_start_btn.setEnabled(False)
        self.record_stop_btn.setEnabled(True)
        self.recording_state_label.setText(
            self._tf("recording_progress_rows", gesture=gesture, rows=0)
        )
        self._set_status(self._t("recording_started_hint"), "INFO")
        self._recording_thread = threading.Thread(
            target=self._recording_loop,
            args=(gesture,),
            daemon=True,
        )
        self._recording_thread.start()
        self._refresh_action_states()

    def stop_manual_recording(self) -> None:
        if self._recording_countdown_active:
            self._recording_countdown_timer.stop()
            self._recording_countdown_active = False
            self._recording_pending_request = None
            self.recording_state_label.setText(self._t("recording_ready"))
            self._set_status(self._t("recording_countdown_cancelled"), "INFO")
            self._refresh_action_states()
            return

        if not self._recording_active:
            self._set_status(self._t("no_active_recording_to_stop"), "WARNING")
            return
        self._recording_stop_event.set()
        self.recording_state_label.setText(self._t("recording_stopping"))
        self._set_status(self._t("recording_stopping_preparing_plot"), "INFO")

    def _recording_loop(self, gesture: str) -> None:
        rows: list[dict[str, float | int]] = []
        try:
            start = self._recording_started_at or time.perf_counter()
            while not self._recording_stop_event.is_set():
                sensor_row = self._recording_serial.read_sensor_row()
                if sensor_row is None:
                    time.sleep(0.005)
                    continue

                elapsed_ms = int((time.perf_counter() - start) * 1000)
                row: dict[str, float | int] = {"t_ms": elapsed_ms}
                row.update(sensor_row)
                rows.append(row)

                if len(rows) % 20 == 0:
                    self.event_queue.put(
                        {
                            "type": "recording_progress",
                            "rows": len(rows),
                            "gesture": gesture,
                        }
                    )
        except Exception as exc:
            self.event_queue.put({"type": "recording_error", "message": str(exc)})
        finally:
            self._recording_serial.disconnect()
            self.event_queue.put(
                {"type": "recording_ready_review", "rows": rows, "gesture": gesture}
            )

    def _open_recording_review(
        self,
        gesture: str,
        rows: list[dict[str, float | int]],
    ) -> None:
        if not rows:
            self._set_status(self._t("recording_no_valid_sensor_data"), "WARNING")
            self.recording_state_label.setText(self._t("recording_ready"))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(self._tf("recording_preview_title", gesture=gesture))
        dialog.resize(1120, 760)

        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                self._tf("recording_preview_summary", gesture=gesture, rows=len(rows))
            )
        )

        fig = Figure(figsize=(11, 7), dpi=100)
        ax_flex = fig.add_subplot(3, 1, 1)
        ax_accel = fig.add_subplot(3, 1, 2)
        ax_gyro = fig.add_subplot(3, 1, 3)

        t_values = [float(r.get("t_ms", 0.0)) for r in rows]
        for i in range(5):
            key = f"flex{i}"
            series = [float(r.get(key, 0.0)) for r in rows]
            ax_flex.plot(t_values, series, label=key)
        ax_flex.set_title(self._t("plot_flex"))
        ax_flex.grid(True, alpha=0.3)
        ax_flex.legend(loc="upper right")

        for key in ["accelX", "accelY", "accelZ"]:
            series = [float(r.get(key, 0.0)) for r in rows]
            ax_accel.plot(t_values, series, label=key)
        ax_accel.set_title(self._t("plot_accelerometer"))
        ax_accel.grid(True, alpha=0.3)
        ax_accel.legend(loc="upper right")

        for key in ["gyroX", "gyroY", "gyroZ"]:
            series = [float(r.get(key, 0.0)) for r in rows]
            ax_gyro.plot(t_values, series, label=key)
        ax_gyro.set_title(self._t("plot_gyroscope"))
        ax_gyro.set_xlabel(self._t("plot_time_ms"))
        ax_gyro.grid(True, alpha=0.3)
        ax_gyro.legend(loc="upper right")

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas, stretch=1)

        btn_row = QHBoxLayout()
        save_btn = QPushButton(self._t("save"))
        discard_btn = QPushButton(self._t("delete"))
        btn_row.addWidget(save_btn)
        btn_row.addWidget(discard_btn)
        layout.addLayout(btn_row)

        def _save() -> None:
            path = self._save_recording_rows(gesture, rows)
            self._set_status(self._tf("recording_saved", name=path.name), "INFO")
            self.recording_state_label.setText(self._t("recording_ready"))
            self.logger.info("Recording saved to %s", path)
            dialog.accept()

        def _discard() -> None:
            self._set_status(self._t("recording_deleted"), "WARNING")
            self.recording_state_label.setText(self._t("recording_ready"))
            dialog.reject()

        save_btn.clicked.connect(_save)
        discard_btn.clicked.connect(_discard)

        dialog.exec()

    def _save_recording_rows(
        self,
        gesture: str,
        rows: list[dict[str, float | int]],
    ) -> Path:
        target_dir = Path(LOGS_DIR) / gesture
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = target_dir / f"{gesture}_{timestamp}.csv"
        fieldnames = [
            "t_ms",
            "flex0",
            "flex1",
            "flex2",
            "flex3",
            "flex4",
            "accelX",
            "accelY",
            "accelZ",
            "gyroX",
            "gyroY",
            "gyroZ",
        ]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target

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
        self.confidence_label.setText(self._format_confidence(percent(confidence)))

        self.history_table.insertRow(0)
        self.history_table.setItem(
            0, 0, QTableWidgetItem(str(event.get("timestamp", "--:--:--")))
        )
        self.history_table.setItem(0, 1, QTableWidgetItem(history_gesture))
        self.history_table.setItem(0, 2, QTableWidgetItem(percent(confidence)))

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

        if self.tts_enabled and self.tts_mode == "llm":
            self.tts_service.speak(text, self._effective_llm_language())

    def _on_llm_status(self, event: dict) -> None:
        message = str(event.get("message", "")).strip()
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
                elif event_type == "recording_progress":
                    gesture = str(event.get("gesture", ""))
                    rows = int(event.get("rows", 0))
                    self.recording_state_label.setText(
                        self._tf("recording_progress_rows", gesture=gesture, rows=rows)
                    )
                elif event_type == "recording_ready_review":
                    self._recording_active = False
                    self.record_start_btn.setEnabled(True)
                    self.record_stop_btn.setEnabled(False)
                    gesture = str(event.get("gesture", ""))
                    rows = list(event.get("rows", []))
                    self._open_recording_review(gesture, rows)
                    self._refresh_action_states()
                elif event_type == "recording_error":
                    self._recording_active = False
                    self.record_start_btn.setEnabled(True)
                    self.record_stop_btn.setEnabled(False)
                    self.recording_state_label.setText(self._t("recording_ready"))
                    self._set_status(
                        str(
                            event.get(
                                "message",
                                self._t("recording_error"),
                            )
                        ),
                        "ERROR",
                    )
                    self._refresh_action_states()
        except queue.Empty:
            return

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop_stream()
        self._recording_countdown_timer.stop()
        self._recording_countdown_active = False
        self._recording_pending_request = None
        self._recording_stop_event.set()
        self._recording_serial.disconnect()
        self.llm_service.shutdown()
        self.tts_service.stop()
        event.accept()


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

    window = SignLanguageDashboard(project_root=project_root)
    window.show()

    if owns_app:
        app.exec()
