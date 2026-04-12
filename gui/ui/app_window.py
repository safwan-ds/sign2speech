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

from config import BAUD_RATE, LOGS_DIR, LOGS_OUTPUT_DIR, MODELS_DIR
from gui.services.llm_service import LLMService
from gui.services.logging_service import configure_gui_logger
from gui.services.model_service import ModelMetadata, ModelService
from gui.services.script_runner import ScriptRunner
from gui.services.serial_service import SerialService, SerialSettings
from gui.services.stream_service import StreamConfig, StreamWorker
from gui.utils.exporter import export_sentence
from gui.utils.formatting import now_stamp, percent
from utils.serial_utils import select_serial_port


class SignLanguageDashboard(QMainWindow):
    """Desktop dashboard for real-time sign prediction using PySide6."""

    STREAM_STARTUP_TIMEOUT_SECONDS = 8.0

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.setWindowTitle("Sign2Speech")
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

        self.current_sentence_tokens: list[str] = []
        self.model_dir = str(Path(MODELS_DIR) / "latest")
        self.llm_enabled = True

        self._recording_active = False
        self._recording_stop_event = threading.Event()
        self._recording_thread: threading.Thread | None = None
        self._recording_started_at: float | None = None
        self._recording_serial = SerialService()

        self._build_ui()
        self._build_shortcuts()
        self.refresh_ports()
        self._load_gesture_options()
        self._set_connection_badge(False)
        self._set_model_badge("Model: Hazır değil", "idle")
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

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("centralRoot")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        title = QLabel("Sign2Speech")
        title.setObjectName("title")
        subtitle = QLabel(
            "Ctrl+S: Başlat/Durdur | Ctrl+L: Temizle | Ctrl+E: Dışa Aktar | Ctrl+C: Cümleyi Kopyala"
        )
        subtitle.setObjectName("subtitle")

        header = QVBoxLayout()
        header.addWidget(title)
        header.addWidget(subtitle)
        outer.addLayout(header)

        self.status_banner = QLabel("Hazır")
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

        label = QLabel("Kontrol Merkezi")
        label.setObjectName("panelTitle")
        layout.addWidget(label)

        self.runtime_label = QLabel("Yayın Süresi: 00:00")
        self.connection_label = QLabel("Cihaz: Bağlı Değil")
        self.model_status_label = QLabel("Model: Hazır değil")
        layout.addWidget(self.runtime_label)
        layout.addWidget(self.connection_label)
        layout.addWidget(self.model_status_label)

        self.start_stop_btn = QPushButton("Yayını Başlat")
        self.start_stop_btn.clicked.connect(self.toggle_stream)
        self.start_stop_btn.setToolTip(
            "Gerçek zamanlı tahmin akışını başlatır veya durdurur."
        )
        layout.addWidget(self.start_stop_btn)

        self.clear_btn = QPushButton("Cümleyi Temizle")
        self.clear_btn.clicked.connect(self.clear_sentence)
        self.clear_btn.setToolTip("Biriken cümleyi ve düzenlenmiş metni temizler.")
        layout.addWidget(self.clear_btn)

        self.copy_btn = QPushButton("Cümleyi Kopyala")
        self.copy_btn.clicked.connect(self.copy_sentence)
        layout.addWidget(self.copy_btn)

        self.export_btn = QPushButton("Metni Dışa Aktar")
        self.export_btn.clicked.connect(self.export_sentence_text)
        self.export_btn.setToolTip(
            "Cümleyi logs klasörüne metin dosyası olarak kaydeder."
        )
        layout.addWidget(self.export_btn)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self.prediction_card = QLabel("REST")
        self.prediction_card.setObjectName("predictionCard")
        self.prediction_card.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction_card)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        layout.addWidget(self.confidence_bar)

        stats = QHBoxLayout()
        self.confidence_label = QLabel("Güven: 0.0%")
        self.word_count_label = QLabel("Kelime Sayısı: 0")
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
        sentence_header.addWidget(QLabel("Çevrilen Cümle"))
        sentence_header.addStretch(1)
        sentence_copy = QPushButton("Kopyala")
        sentence_copy.clicked.connect(self.copy_sentence)
        sentence_header.addWidget(sentence_copy)
        sentence_layout.addLayout(sentence_header)

        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setPlainText(
            "Tahmin edilen işaretler burada birikerek cümleye dönüşecek."
        )
        sentence_layout.addWidget(self.sentence_box)

        refined_container = QWidget()
        refined_layout = QVBoxLayout(refined_container)
        refined_layout.setContentsMargins(0, 0, 0, 0)
        refined_layout.setSpacing(6)

        refined_header = QHBoxLayout()
        refined_header.addWidget(QLabel("QWEN Düzenlenmiş Cümle"))
        refined_header.addStretch(1)
        refined_copy = QPushButton("Kopyala")
        refined_copy.clicked.connect(self.copy_refined)
        refined_header.addWidget(refined_copy)
        refined_layout.addLayout(refined_header)

        self.refined_box = QTextEdit()
        self.refined_box.setReadOnly(True)
        self.refined_box.setPlainText(
            "QWEN etkinken burada daha akıcı cümle önerisi görünür."
        )
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
        history_layout.addWidget(QLabel("Son Tahminler"))

        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["Saat", "Sınıf", "Güven"])
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
        tabs = QTabWidget()
        tabs.setObjectName("rightTabs")
        tabs.addTab(self._build_settings_tab(), "Model ve Port")
        tabs.addTab(self._build_actions_tab(), "Yardımcı İşlemler")
        tabs.addTab(self._build_logs_tab(), "Kayıtlar")
        return tabs

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("settingsTab")
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)

        self.model_path_edit = QLineEdit(self.model_dir)
        model_layout.addWidget(self.model_path_edit)

        model_btns = QHBoxLayout()
        browse_btn = QPushButton("Gözat")
        browse_btn.clicked.connect(self.select_model_dir)
        latest_btn = QPushButton("En Son")
        latest_btn.clicked.connect(self.use_latest_model)
        self.load_btn = QPushButton("Yükle")
        self.load_btn.clicked.connect(self.load_model_async)
        model_btns.addWidget(browse_btn)
        model_btns.addWidget(latest_btn)
        model_btns.addWidget(self.load_btn)
        model_layout.addLayout(model_btns)

        self.model_meta_label = QLabel("Model yüklenmedi")
        self.model_meta_label.setWordWrap(True)
        model_layout.addWidget(self.model_meta_label)

        layout.addWidget(model_group)

        settings_group = QGroupBox("Ayarlar")
        settings_layout = QGridLayout(settings_group)

        settings_layout.addWidget(QLabel("Seri Port"), 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.currentTextChanged.connect(
            lambda _text: self._refresh_action_states()
        )
        settings_layout.addWidget(self.port_combo, 1, 0, 1, 2)

        refresh_btn = QPushButton("Portları Yenile")
        refresh_btn.clicked.connect(self.refresh_ports)
        settings_layout.addWidget(refresh_btn, 2, 0, 1, 2)

        settings_layout.addWidget(QLabel("Baud Hızı"), 3, 0)
        self.baud_edit = QLineEdit(str(BAUD_RATE))
        self.baud_edit.setPlaceholderText("Örn: 115200")
        settings_layout.addWidget(self.baud_edit, 4, 0, 1, 2)

        self.threshold_label = QLabel("Güven Eşiği: 0.70")
        settings_layout.addWidget(self.threshold_label, 5, 0, 1, 2)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 99)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        settings_layout.addWidget(self.threshold_slider, 6, 0, 1, 2)

        self.smoothing_label = QLabel("Yumuşatma Penceresi: 5")
        settings_layout.addWidget(self.smoothing_label, 7, 0, 1, 2)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 12)
        self.smoothing_slider.setValue(5)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_change)
        settings_layout.addWidget(self.smoothing_slider, 8, 0, 1, 2)

        self.llm_checkbox = QCheckBox("QWEN düzenlemesini etkinleştir")
        self.llm_checkbox.setChecked(True)
        self.llm_checkbox.stateChanged.connect(self._on_llm_changed)
        settings_layout.addWidget(self.llm_checkbox, 9, 0, 1, 2)

        layout.addWidget(settings_group)
        layout.addStretch(1)
        return tab

    def _build_actions_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("actionsTab")
        layout = QVBoxLayout(tab)

        script_group = QGroupBox("Yardımcı İşlemler")
        script_layout = QVBoxLayout(script_group)

        process_btn = QPushButton("Veriyi İşle")
        process_btn.clicked.connect(lambda: self.run_script("scripts/process_data.py"))
        train_btn = QPushButton("Modeli Eğit")
        train_btn.clicked.connect(lambda: self.run_script("scripts/train_model.py"))
        predict_btn = QPushButton("Tahmin Scriptini Çalıştır")
        predict_btn.clicked.connect(lambda: self.run_script("scripts/predict.py"))

        script_layout.addWidget(process_btn)
        script_layout.addWidget(train_btn)
        script_layout.addWidget(predict_btn)
        layout.addWidget(script_group)

        record_group = QGroupBox("Yeni Veri Kaydı")
        record_layout = QVBoxLayout(record_group)

        record_layout.addWidget(QLabel("Jest"))
        self.record_gesture_combo = QComboBox()
        record_layout.addWidget(self.record_gesture_combo)

        refresh_gesture_btn = QPushButton("Jest Listesini Yenile")
        refresh_gesture_btn.clicked.connect(self._load_gesture_options)
        record_layout.addWidget(refresh_gesture_btn)

        row = QHBoxLayout()
        self.record_start_btn = QPushButton("Kaydı Başlat")
        self.record_start_btn.clicked.connect(self.start_manual_recording)
        self.record_stop_btn = QPushButton("Kaydı Durdur")
        self.record_stop_btn.clicked.connect(self.stop_manual_recording)
        self.record_stop_btn.setEnabled(False)
        row.addWidget(self.record_start_btn)
        row.addWidget(self.record_stop_btn)
        record_layout.addLayout(row)

        self.recording_state_label = QLabel("Hazır")
        record_layout.addWidget(self.recording_state_label)

        helper = QLabel("Her kayıttan sonra grafiği inceleyip Kaydet/Sil seçimi yapın.")
        helper.setWordWrap(True)
        record_layout.addWidget(helper)

        layout.addWidget(record_group)
        layout.addStretch(1)
        return tab

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("logsTab")
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        self.clear_logs_btn = QPushButton("Kayıtları Temizle")
        self.clear_logs_btn.clicked.connect(self._clear_logs_view)
        self.export_logs_btn = QPushButton("Kayıtları Dışa Aktar")
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

    def _set_connection_badge(self, connected: bool) -> None:
        if connected:
            self.connection_label.setText("Cihaz: Bağlı")
            self.connection_label.setStyleSheet(
                "padding: 6px 8px; border-radius: 6px; background: #e7f6ea; color: #1e6c35;"
            )
        else:
            self.connection_label.setText("Cihaz: Bağlı Değil")
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

    def _refresh_action_states(self) -> None:
        streaming = bool(self.worker and self.worker.is_alive())
        has_model = self._model_loaded
        has_sentence = bool(self.current_sentence_tokens)
        has_port = bool(self._selected_port())
        has_gesture = self.record_gesture_combo.currentText().strip() not in {
            "",
            "Jest bulunamadı",
        }

        self.start_stop_btn.setEnabled(
            streaming or (has_model and has_port and not self._recording_active)
        )
        self.copy_btn.setEnabled(has_sentence)
        self.export_btn.setEnabled(has_sentence)
        self.record_start_btn.setEnabled(
            (not self._recording_active)
            and (not streaming)
            and has_port
            and has_gesture
        )
        self.record_stop_btn.setEnabled(self._recording_active)

    def _selected_port(self) -> str:
        data = self.port_combo.currentData()
        if isinstance(data, str):
            return data.strip()
        return ""

    def _clear_logs_view(self) -> None:
        self.log_box.clear()
        self._set_status("Kayıt görünümü temizlendi", "INFO")

    def _export_logs_view(self) -> None:
        text = self.log_box.toPlainText().strip()
        if not text:
            self._set_status("Dışa aktarılacak kayıt yok", "WARNING")
            return
        target = Path(LOGS_OUTPUT_DIR) / f"gui_view_{now_stamp()}.log"
        target.write_text(text + "\n", encoding="utf-8")
        self._set_status(f"Kayıtlar dışa aktarıldı: {target.name}", "INFO")

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
        self.threshold_label.setText(f"Güven Eşiği: {value / 100:.2f}")

    def _on_smoothing_change(self, value: int) -> None:
        self.smoothing_label.setText(f"Yumuşatma Penceresi: {value}")

    def _on_llm_changed(self, state: int) -> None:
        self.llm_enabled = state == Qt.CheckState.Checked.value

    def refresh_ports(self) -> None:
        current = self._selected_port()
        entries = SerialService.list_port_entries()
        self.port_combo.clear()
        if not entries:
            self.port_combo.addItem("Port Yok")
            self._set_status("Seri port bulunamadı", "WARNING")
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

        self._set_status("Seri portlar yenilendi", "INFO")
        self._refresh_action_states()

    def _load_gesture_options(self) -> None:
        labels: list[str] = []
        gestures_file = self.project_root / "config" / "gestures.txt"
        if gestures_file.exists():
            with gestures_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    name = line.split(" - ", 1)[0].strip()
                    if name:
                        labels.append(name)

        raw_dir = Path(LOGS_DIR)
        if raw_dir.exists():
            for child in raw_dir.iterdir():
                if child.is_dir() and child.name not in labels:
                    labels.append(child.name)

        labels = sorted(set(labels), key=str.lower)
        if not labels:
            labels = ["Jest bulunamadı"]

        current = self.record_gesture_combo.currentText()
        self.record_gesture_combo.clear()
        self.record_gesture_combo.addItems(labels)

        idx = self.record_gesture_combo.findText(current)
        if idx >= 0:
            self.record_gesture_combo.setCurrentIndex(idx)
        self._refresh_action_states()

    def select_model_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Model klasörü seç", str(self.project_root)
        )
        if chosen:
            self.model_path_edit.setText(chosen)

    def use_latest_model(self) -> None:
        self.model_path_edit.setText(str(Path(MODELS_DIR) / "latest"))

    def load_model_async(self) -> None:
        model_dir = Path(self.model_path_edit.text().strip())
        self._set_status("Model yükleniyor...", "INFO")
        self._set_model_badge("Model: Yükleniyor...", "loading")
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
                f"Sınıflar ({len(metadata.classes)}): {classes_preview}\n"
                f"Sekans Uzunluğu: {metadata.sequence_length}\n"
                f"Giriş Şekli: {metadata.input_shape}\n"
                f"Yüklenme: {metadata.loaded_at}"
            )
        )
        self._model_loaded = True
        self._set_model_badge("Model: Hazır", "ready")
        self.load_btn.setEnabled(True)
        self._refresh_action_states()

    def _get_stream_config(self) -> StreamConfig:
        port = self._selected_port()
        if not port:
            raise ValueError("Seri port seçilmedi")

        baud = int(self.baud_edit.text().strip())
        serial_settings = SerialSettings(port=port, baud_rate=baud)
        return StreamConfig(
            serial_settings=serial_settings,
            confidence_threshold=float(self.threshold_slider.value()) / 100.0,
            smoothing_window=int(self.smoothing_slider.value()),
        )

    def toggle_stream(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self) -> None:
        if self.model_service.predictor is None:
            self._set_status("Yayından önce bir model yükleyin", "WARNING")
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
        self.start_stop_btn.setText("Yayını Durdur")
        self._set_status("Yayın başlatılıyor... veri bekleniyor", "INFO")

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
        self.start_stop_btn.setText("Yayını Başlat")
        if feedback_message:
            self._set_status(feedback_message, level)
        else:
            self._set_status("Yayın durduruluyor...", "INFO")
        self._refresh_action_states()

    def copy_sentence(self) -> None:
        sentence = " ".join(self.current_sentence_tokens).strip()
        if not sentence:
            self._set_status("Kopyalanacak cümle yok", "WARNING")
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(sentence)
        self._set_status("Cümle panoya kopyalandı", "INFO")

    def copy_refined(self) -> None:
        text = self.refined_box.toPlainText().strip()
        if not text:
            self._set_status("Kopyalanacak düzenlenmiş metin yok", "WARNING")
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self._set_status("Düzenlenmiş metin panoya kopyalandı", "INFO")

    def clear_sentence(self) -> None:
        self.current_sentence_tokens.clear()
        self.word_count_label.setText("Kelime Sayısı: 0")
        self.sentence_box.setPlainText(
            "Tahmin edilen işaretler burada birikerek cümleye dönüşecek."
        )
        self.refined_box.setPlainText(
            "QWEN etkinken burada daha akıcı cümle önerisi görünür."
        )
        self._set_status("Cümle temizlendi", "INFO")
        self._refresh_action_states()

    def export_sentence_text(self) -> None:
        sentence = " ".join(self.current_sentence_tokens)
        if not sentence.strip():
            self._set_status("Dışa aktarılacak cümle yok", "WARNING")
            return

        target = export_sentence(sentence, Path(LOGS_OUTPUT_DIR))
        self._set_status(f"Cümle dışa aktarıldı: {target.name}", "INFO")
        self.logger.info("Sentence exported to %s", target)

    def run_script(self, relative_script_path: str) -> None:
        ok = self.script_runner.run_script(relative_script_path)
        if not ok:
            self._set_status("Zaten çalışan bir script var", "WARNING")

    def start_manual_recording(self) -> None:
        if self.worker and self.worker.is_alive():
            self._set_status("Örnek kaydı için önce yayını durdurun", "WARNING")
            return
        if self._recording_active:
            self._set_status("Kayıt zaten devam ediyor", "WARNING")
            return

        gesture = self.record_gesture_combo.currentText().strip()
        if not gesture or gesture == "Jest bulunamadı":
            self._set_status("Önce kayıt için bir jest seçin", "WARNING")
            return

        port = self._selected_port()
        if not port:
            self._set_status("Örnek kaydı için bir seri port seçin", "WARNING")
            return

        try:
            baud = int(self.baud_edit.text().strip())
        except ValueError:
            self._set_status("Baud hızı sayısal olmalıdır", "WARNING")
            return

        try:
            self._recording_serial.connect(
                SerialSettings(port=port, baud_rate=baud, timeout=0.1)
            )
        except Exception as exc:
            self._set_status(f"Seri bağlantı açılamadı: {exc}", "ERROR")
            return

        self._recording_stop_event.clear()
        self._recording_started_at = time.perf_counter()
        self._recording_active = True
        self.record_start_btn.setEnabled(False)
        self.record_stop_btn.setEnabled(True)
        self.recording_state_label.setText(f"Kaydediliyor: {gesture} (0 örnek satırı)")
        self._set_status("Kayıt başladı. Hareketi yapıp Kaydı Durdur'a basın", "INFO")
        self._recording_thread = threading.Thread(
            target=self._recording_loop,
            args=(gesture,),
            daemon=True,
        )
        self._recording_thread.start()
        self._refresh_action_states()

    def stop_manual_recording(self) -> None:
        if not self._recording_active:
            self._set_status("Durdurulacak aktif kayıt yok", "WARNING")
            return
        self._recording_stop_event.set()
        self.recording_state_label.setText("Kayıt durduruluyor...")
        self._set_status("Kayıt durduruluyor, grafik hazırlanıyor...", "INFO")

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
            self._set_status("Kayıtta geçerli sensör verisi yakalanmadı", "WARNING")
            self.recording_state_label.setText("Hazır")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Kayıt Önizleme - {gesture}")
        dialog.resize(1120, 760)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"{gesture} | {len(rows)} örnek satırı"))

        fig = Figure(figsize=(11, 7), dpi=100)
        ax_flex = fig.add_subplot(3, 1, 1)
        ax_accel = fig.add_subplot(3, 1, 2)
        ax_gyro = fig.add_subplot(3, 1, 3)

        t_values = [float(r.get("t_ms", 0.0)) for r in rows]
        for i in range(5):
            key = f"flex{i}"
            series = [float(r.get(key, 0.0)) for r in rows]
            ax_flex.plot(t_values, series, label=key)
        ax_flex.set_title("Flex")
        ax_flex.grid(True, alpha=0.3)
        ax_flex.legend(loc="upper right")

        for key in ["accelX", "accelY", "accelZ"]:
            series = [float(r.get(key, 0.0)) for r in rows]
            ax_accel.plot(t_values, series, label=key)
        ax_accel.set_title("Accelerometer")
        ax_accel.grid(True, alpha=0.3)
        ax_accel.legend(loc="upper right")

        for key in ["gyroX", "gyroY", "gyroZ"]:
            series = [float(r.get(key, 0.0)) for r in rows]
            ax_gyro.plot(t_values, series, label=key)
        ax_gyro.set_title("Gyroscope")
        ax_gyro.set_xlabel("Time (ms)")
        ax_gyro.grid(True, alpha=0.3)
        ax_gyro.legend(loc="upper right")

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas, stretch=1)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Kaydet")
        discard_btn = QPushButton("Sil")
        btn_row.addWidget(save_btn)
        btn_row.addWidget(discard_btn)
        layout.addLayout(btn_row)

        def _save() -> None:
            path = self._save_recording_rows(gesture, rows)
            self._set_status(f"Kayıt kaydedildi: {path.name}", "INFO")
            self.recording_state_label.setText("Hazır")
            self.logger.info("Recording saved to %s", path)
            dialog.accept()

        def _discard() -> None:
            self._set_status("Kayıt silindi", "WARNING")
            self.recording_state_label.setText("Hazır")
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
        gesture = str(event.get("gesture", "Unknown"))
        confidence = float(event.get("confidence", 0.0))

        self.prediction_card.setText(gesture.upper())
        self.confidence_bar.setValue(int(max(0.0, min(1.0, confidence)) * 100))
        self.confidence_label.setText(f"Güven: {percent(confidence)}")

        self.history_table.insertRow(0)
        self.history_table.setItem(
            0, 0, QTableWidgetItem(str(event.get("timestamp", "--:--:--")))
        )
        self.history_table.setItem(0, 1, QTableWidgetItem(gesture))
        self.history_table.setItem(0, 2, QTableWidgetItem(percent(confidence)))

        while self.history_table.rowCount() > 20:
            self.history_table.removeRow(self.history_table.rowCount() - 1)

    def _on_sentence(self, event: dict) -> None:
        token = str(event.get("token", "")).strip()
        if not token:
            return

        self.current_sentence_tokens.append(token)
        sentence_text = " ".join(self.current_sentence_tokens)
        self.word_count_label.setText(
            f"Kelime Sayısı: {len(self.current_sentence_tokens)}"
        )
        self.sentence_box.setPlainText(sentence_text)

    def _on_llm_request(self, event: dict) -> None:
        if not self.llm_enabled:
            return

        text = str(event.get("text", "")).strip()
        if not text:
            return
        self.llm_service.request_refinement(text)

    def _on_llm_text(self, event: dict) -> None:
        text = str(event.get("text", "")).strip()
        if not text:
            return
        self.refined_box.setPlainText(text)

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
                    f"Seçili seri port artık bulunamadı: {self._active_stream_port}. Yayın durduruldu.",
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
                "Seçili port doğru veri göndermiyor veya yanlış port seçildi. Yayın durduruldu.",
                "WARNING",
            )
            self._set_connection_badge(False)
            return

        if self._stream_started_at is None:
            self.runtime_label.setText("Yayın Süresi: 00:00")
            return

        elapsed = int(time.time() - self._stream_started_at)
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            text = f"Yayın Süresi: {hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            text = f"Yayın Süresi: {minutes:02d}:{seconds:02d}"
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
                        self.start_stop_btn.setText("Yayını Durdur")
                        if not self._stream_input_detected:
                            self._set_status(
                                "Port bağlandı, veri bekleniyor...", "INFO"
                            )
                elif event_type == "stream_input_detected":
                    self._stream_input_detected = True
                    self._stream_start_timeout_at = None
                    self._set_status("Port doğrulandı, yayın hazırlanıyor...", "INFO")
                elif event_type == "stream_started":
                    self._stream_started_at = time.time()
                    self._set_status("Yayın başlatıldı", "INFO")
                elif event_type == "model_loaded":
                    self._update_model_meta(event["metadata"])
                    self._set_status("Model başarıyla yüklendi", "INFO")
                elif event_type == "stopped":
                    was_connected = self._stream_connected
                    had_error = self._stream_had_error
                    stop_requested = self._stream_stop_requested
                    self._stream_started_at = None
                    self._stream_connected = False
                    self._active_stream_port = None
                    self.start_stop_btn.setText("Yayını Başlat")
                    if stop_requested:
                        if self._stream_stop_message:
                            self._set_status(
                                self._stream_stop_message, self._stream_stop_level
                            )
                        else:
                            self._set_status("Yayın durduruldu", "INFO")
                    elif not had_error:
                        if was_connected:
                            self._set_status(
                                "Yayın beklenmedik şekilde durdu", "WARNING"
                            )
                        else:
                            self._set_status("Yayın başlatılamadı", "ERROR")
                    self._refresh_action_states()
                elif event_type == "error":
                    message = str(event.get("message", "Unknown error"))
                    self._stream_had_error = True
                    self._set_status(message, "ERROR")
                    if "Model" in message:
                        self._model_loaded = False
                        self._set_model_badge("Model: Hata", "error")
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
                        f"Kaydediliyor: {gesture} ({rows} örnek satırı)"
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
                    self.recording_state_label.setText("Hazır")
                    self._set_status(str(event.get("message", "Kayıt hatası")), "ERROR")
                    self._refresh_action_states()
        except queue.Empty:
            return

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop_stream()
        self._recording_stop_event.set()
        self._recording_serial.disconnect()
        self.llm_service.shutdown()
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
