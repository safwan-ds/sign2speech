"""PySide6 management GUI for dataset and model lifecycle tasks."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QBrush, QKeySequence, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from config.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LOGS_DIR,
    LOGS_OUTPUT_DIR,
)
from gui.services.data_processing_service import DataProcessingService
from gui.services.logging_service import configure_gui_logger
from gui.services.recording_service import RecordingConfig, RecordingService
from gui.services.sample_review_service import SampleRecord, SampleReviewService
from gui.services.serial_service import SerialService
from gui.services.training_service import TrainingOverrides, TrainingService
from gui.ui.custom_widgets_adapter import apply_custom_widgets_theme
from gui.ui.theme_manager import (
    build_data_manager_stylesheet,
    get_confusion_cell_color,
    get_confusion_text_color,
    get_plot_palette,
    get_status_banner_style,
)
from gui.ui.trace_preview_widget import TracePreviewWidget
from utils.recording_utils import (
    count_csv_samples,
    load_gesture_names,
    sanitize_gesture_label,
)
from utils.serial_utils import select_serial_port


class DataManagerWindow(QMainWindow):
    """Standalone GUI for recording, processing, training, and data cleaning."""

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.raw_data_root = Path(LOGS_DIR)
        self.samples: list[SampleRecord] = []
        self._task_active = False
        self._task_name = ""
        self._process_total_gestures = 0
        self._process_seen_gestures: set[str] = set()
        self._process_current_gesture = ""
        self._process_table_rows: dict[str, int] = {}
        self._train_total_epochs = 0
        self._train_result_rows: dict[str, int] = {}
        self._train_metrics: dict[str, str] = {}
        self._train_model_dir = ""  # Track latest model dir for metrics loading
        self._theme_name = "dark"
        self._last_record_preview_update = 0.0
        self._record_preview_update_interval = 0.05

        self.event_queue: queue.Queue[dict] = queue.Queue()
        self.logger = configure_gui_logger(Path(LOGS_OUTPUT_DIR), self.event_queue)
        self.data_processing_service = DataProcessingService(
            logger=self.logger,
            event_queue=self.event_queue,
        )
        self.training_service = TrainingService(
            logger=self.logger,
            event_queue=self.event_queue,
        )
        self.serial_service = SerialService()
        self.recording_service = RecordingService(
            logger=self.logger,
            event_queue=self.event_queue,
            serial_service=self.serial_service,
        )
        self.sample_service = SampleReviewService(self.raw_data_root)

        self.setWindowTitle("Sign2Speech - Dataset Manager")
        self.resize(1560, 940)
        self.setMinimumSize(1220, 760)

        self._build_ui()
        self._build_shortcuts()
        self._apply_window_styles(self._theme_name)
        self.refresh_record_ports(announce=False)

        self._load_gesture_options()
        self.refresh_samples()
        self._refresh_record_count()

        self.event_timer = QTimer(self)
        self.event_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.event_timer.setInterval(50)
        self.event_timer.timeout.connect(self._poll_events)
        self.event_timer.start()

        self._set_status("Ready", "INFO")

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("centralRoot")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        self.title_label = QLabel("Dataset and Model Manager")
        self.title_label.setObjectName("title")
        self.subtitle_label = QLabel(
            "Record new samples, process datasets, train models, and clean invalid captures."
        )
        self.subtitle_label.setObjectName("subtitle")
        outer.addWidget(self.title_label)
        outer.addWidget(self.subtitle_label)

        self.status_banner = QLabel("")
        self.status_banner.setObjectName("statusInfo")
        self.status_banner.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        outer.addWidget(self.status_banner)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)
        outer.addWidget(body, stretch=1)

        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("mainTabs")
        self.main_tabs.addTab(self._build_record_tab(), "Record")
        self.main_tabs.addTab(self._build_process_tab(), "Process")
        self.main_tabs.addTab(self._build_train_tab(), "Train")
        self.main_tabs.addTab(self._build_review_tab(), "Review and Cleanup")

        log_panel = self._build_log_panel()
        body.addWidget(self.main_tabs)
        body.addWidget(log_panel)
        body.setStretchFactor(0, 4)
        body.setStretchFactor(1, 2)

        self.setStatusBar(QStatusBar(self))

    def _build_record_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        record_port = QHBoxLayout()

        self.record_group = QGroupBox("Record New Samples")
        form = QFormLayout(self.record_group)

        self.record_gesture_combo = QComboBox()
        self.record_gesture_combo.setEditable(False)
        self.record_gesture_combo.currentTextChanged.connect(
            lambda _text: self._refresh_record_count()
        )
        form.addRow("Gesture", self.record_gesture_combo)

        record_port.addWidget(self.record_group)

        self.record_port_group = QGroupBox("Serial Port Management")
        port_layout = QHBoxLayout(self.record_port_group)
        port_layout.setSpacing(6)

        self.record_port_combo = QComboBox()
        self.record_port_combo.setToolTip("Select serial port used for recording")
        port_layout.addWidget(self.record_port_combo)

        self.record_refresh_ports_btn = QPushButton("Refresh Ports")
        self.record_refresh_ports_btn.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Preferred
        )
        self.record_refresh_ports_btn.clicked.connect(self.refresh_record_ports)
        port_layout.addWidget(self.record_refresh_ports_btn)

        record_port.addWidget(self.record_port_group)
        layout.addLayout(record_port)

        action_row = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.record_btn.clicked.connect(self.start_recording)
        self.record_btn.setToolTip("Start recording (Ctrl+R)")
        action_row.addWidget(self.record_btn)

        self.record_stop_btn = QPushButton("Stop")
        self.record_stop_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.record_stop_btn.clicked.connect(self.stop_recording)
        self.record_stop_btn.setToolTip(
            "Stop recording and review before save (Ctrl+T)"
        )
        self.record_stop_btn.setEnabled(False)
        action_row.addWidget(self.record_stop_btn)

        record_port.addLayout(action_row)
        layout.addLayout(record_port)

        self.recording_status_label = QLabel("Recording status: idle")
        layout.addWidget(self.recording_status_label)

        self.record_row_count_label = QLabel("Rows captured: 0")
        layout.addWidget(self.record_row_count_label)

        self.record_stats_label = QLabel("Samples for selected gesture: 0")
        layout.addWidget(self.record_stats_label)

        self.record_hint_label = QLabel(
            "Use Start to begin capture and Stop to review before save. Shortcuts: Ctrl+R start, Ctrl+T stop."
        )
        self.record_hint_label.setWordWrap(True)
        layout.addWidget(self.record_hint_label)

        preview_group = QGroupBox("Live Capture Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)
        self.record_preview_plot = TracePreviewWidget(minimum_height=540)
        preview_layout.addWidget(self.record_preview_plot, stretch=1)
        layout.addWidget(preview_group, stretch=2)

        return tab

    def _build_shortcuts(self) -> None:
        self.start_recording_action = QAction(self)
        self.start_recording_action.setShortcut(QKeySequence("Ctrl+R"))
        self.start_recording_action.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self.start_recording_action.triggered.connect(self.start_recording)
        self.addAction(self.start_recording_action)

        self.stop_recording_action = QAction(self)
        self.stop_recording_action.setShortcut(QKeySequence("Ctrl+T"))
        self.stop_recording_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.stop_recording_action.triggered.connect(self.stop_recording)
        self.addAction(self.stop_recording_action)

    def _build_process_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        pipeline_group = QGroupBox("Processing Pipeline")
        grid = QGridLayout(pipeline_group)

        grid.addWidget(QLabel("1. Load all raw CSV recordings"), 0, 0)
        grid.addWidget(QLabel("2. Normalize and segment sequences"), 1, 0)
        grid.addWidget(
            QLabel("3. Regenerate data/processed and data/test outputs"), 2, 0
        )

        self.process_btn = QPushButton("Run Processing")
        self.process_btn.clicked.connect(self.run_processing)
        grid.addWidget(self.process_btn, 3, 0)

        layout.addWidget(pipeline_group)

        self.process_status_label = QLabel("Status: idle")
        layout.addWidget(self.process_status_label)

        self.process_progress_bar = QProgressBar()
        self.process_progress_bar.setRange(0, 100)
        self.process_progress_bar.setValue(0)
        self.process_progress_bar.setFormat("Progress: 0%")
        layout.addWidget(self.process_progress_bar)

        self.process_results_table = QTableWidget(0, 5)
        self.process_results_table.setHorizontalHeaderLabels(
            ["Gesture", "Files", "Samples", "Train Seq", "Test Seq"]
        )
        self.process_results_table.verticalHeader().setVisible(False)
        self.process_results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.process_results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.process_results_table)

        self.process_events_box = QPlainTextEdit()
        self.process_events_box.setReadOnly(True)
        self.process_events_box.setPlaceholderText(
            "Processing events will appear here..."
        )
        layout.addWidget(self.process_events_box, stretch=1)

        self.process_summary_label = QLabel(
            f"Raw directory: {self.raw_data_root}\nProcessed output: {self.project_root / 'data' / 'processed'}"
        )
        self.process_summary_label.setWordWrap(True)
        layout.addWidget(self.process_summary_label)
        return tab

    def _build_train_tab(self) -> QWidget:
        tab = QWidget()
        mainLayout = QVBoxLayout(tab)
        mainLayout.setSpacing(8)
        mainLayout.setContentsMargins(10, 10, 10, 10)

        # ========== TOP SECTION: Controls ==========
        base_group = QGroupBox("Train New Model")
        form = QFormLayout(base_group)

        train_config_label = QLabel("Training uses the default project configuration.")
        train_config_label.setWordWrap(True)
        form.addRow(train_config_label)

        mainLayout.addWidget(base_group)

        # ========== CONFIG OVERRIDES SECTION ==========
        override_group = QGroupBox("Training Configuration Overrides")
        override_form = QFormLayout(override_group)
        override_form.setSpacing(6)

        self.train_epochs_spin = QSpinBox()
        self.train_epochs_spin.setRange(1, 2000)
        self.train_epochs_spin.setValue(EPOCHS)
        self.train_epochs_spin.setToolTip(f"Max training epochs (config default: {EPOCHS})")
        override_form.addRow("Max Epochs:", self.train_epochs_spin)

        self.train_lr_spin = QDoubleSpinBox()
        self.train_lr_spin.setRange(0.00001, 0.1)
        self.train_lr_spin.setDecimals(6)
        self.train_lr_spin.setSingleStep(0.0001)
        self.train_lr_spin.setValue(LEARNING_RATE)
        self.train_lr_spin.setToolTip(f"Initial learning rate (config default: {LEARNING_RATE})")
        override_form.addRow("Learning Rate:", self.train_lr_spin)

        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(1, 512)
        self.train_batch_spin.setValue(BATCH_SIZE)
        self.train_batch_spin.setToolTip(f"Mini-batch size (config default: {BATCH_SIZE})")
        override_form.addRow("Batch Size:", self.train_batch_spin)

        self.train_patience_spin = QSpinBox()
        self.train_patience_spin.setRange(1, 200)
        self.train_patience_spin.setValue(EARLY_STOPPING_PATIENCE)
        self.train_patience_spin.setToolTip(
            f"Early stopping patience in epochs (config default: {EARLY_STOPPING_PATIENCE})"
        )
        override_form.addRow("Early Stop Patience:", self.train_patience_spin)

        mainLayout.addWidget(override_group)

        action_row = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        action_row.addWidget(self.train_btn)

        self.train_cancel_btn = QPushButton("Cancel Training")
        self.train_cancel_btn.clicked.connect(self.cancel_training)
        self.train_cancel_btn.setEnabled(False)
        action_row.addWidget(self.train_cancel_btn)

        action_row.addStretch(1)
        mainLayout.addLayout(action_row)

        # ========== PROGRESS SECTION ==========
        self.train_status_label = QLabel("Status: idle")
        self.train_status_label.setMaximumHeight(20)
        mainLayout.addWidget(self.train_status_label)

        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("Epoch progress: 0%")
        self.train_progress_bar.setMaximumHeight(25)
        mainLayout.addWidget(self.train_progress_bar)

        # ========== METRICS SECTION (compact grid) ==========
        metrics_group = QGroupBox("Live Training Metrics")
        metrics_grid = QGridLayout(metrics_group)
        metrics_grid.setSpacing(8)
        metrics_grid.setContentsMargins(8, 8, 8, 8)

        # Row 0: Loss metrics (side by side)
        self.train_loss_box = QPlainTextEdit()
        self.train_loss_box.setReadOnly(True)
        self.train_loss_box.setPlaceholderText("-")
        self.train_loss_box.setMaximumHeight(35)
        self.train_loss_box.setMinimumHeight(35)
        metrics_grid.addWidget(QLabel("Train Loss:"), 0, 0)
        metrics_grid.addWidget(self.train_loss_box, 0, 1)

        self.val_loss_box = QPlainTextEdit()
        self.val_loss_box.setReadOnly(True)
        self.val_loss_box.setPlaceholderText("-")
        self.val_loss_box.setMaximumHeight(35)
        self.val_loss_box.setMinimumHeight(35)
        metrics_grid.addWidget(QLabel("Val Loss:"), 0, 2)
        metrics_grid.addWidget(self.val_loss_box, 0, 3)

        # Row 1: Accuracy metrics (side by side)
        self.train_acc_box = QPlainTextEdit()
        self.train_acc_box.setReadOnly(True)
        self.train_acc_box.setPlaceholderText("-")
        self.train_acc_box.setMaximumHeight(35)
        self.train_acc_box.setMinimumHeight(35)
        metrics_grid.addWidget(QLabel("Train Acc:"), 1, 0)
        metrics_grid.addWidget(self.train_acc_box, 1, 1)

        self.val_acc_box = QPlainTextEdit()
        self.val_acc_box.setReadOnly(True)
        self.val_acc_box.setPlaceholderText("-")
        self.val_acc_box.setMaximumHeight(35)
        self.val_acc_box.setMinimumHeight(35)
        metrics_grid.addWidget(QLabel("Val Acc:"), 1, 2)
        metrics_grid.addWidget(self.val_acc_box, 1, 3)

        # Row 2: Learning rate and best accuracy (side by side)
        self.train_lr_box = QPlainTextEdit()
        self.train_lr_box.setReadOnly(True)
        self.train_lr_box.setPlaceholderText("-")
        self.train_lr_box.setMaximumHeight(35)
        self.train_lr_box.setMinimumHeight(35)
        metrics_grid.addWidget(QLabel("Learning Rate:"), 2, 0)
        metrics_grid.addWidget(self.train_lr_box, 2, 1)

        self.train_best_val_box = QPlainTextEdit()
        self.train_best_val_box.setReadOnly(True)
        self.train_best_val_box.setPlaceholderText("-")
        self.train_best_val_box.setMaximumHeight(35)
        self.train_best_val_box.setMinimumHeight(35)
        metrics_grid.addWidget(QLabel("Best Val Acc:"), 2, 2)
        metrics_grid.addWidget(self.train_best_val_box, 2, 3)

        metrics_group.setMaximumHeight(140)
        mainLayout.addWidget(metrics_group)

        # ========== BOTTOM SECTION: HORIZONTAL SPLITTER ==========
        # Results on LEFT, Confusion Matrix on RIGHT
        hsplitter = QSplitter(Qt.Orientation.Horizontal)
        hsplitter.setChildrenCollapsible(False)

        # LEFT: Results & Metrics Tables (vertical splitter)
        vsplitter = QSplitter(Qt.Orientation.Vertical)
        vsplitter.setChildrenCollapsible(False)

        # Results Table
        self.train_results_table = QTableWidget(0, 2)
        self.train_results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.train_results_table.verticalHeader().setVisible(False)
        self.train_results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.train_results_table.horizontalHeader().setStretchLastSection(True)
        self.train_results_table.setMinimumHeight(80)
        results_label = QLabel("Training Results:")
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(4)
        results_layout.addWidget(results_label)
        results_layout.addWidget(self.train_results_table)
        vsplitter.addWidget(results_container)

        # Evaluation Metrics Table
        self.train_eval_metrics_table = QTableWidget(0, 2)
        self.train_eval_metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.train_eval_metrics_table.verticalHeader().setVisible(False)
        self.train_eval_metrics_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.train_eval_metrics_table.horizontalHeader().setStretchLastSection(True)
        self.train_eval_metrics_table.setMinimumHeight(80)
        eval_label = QLabel("Evaluation Metrics:")
        eval_container = QWidget()
        eval_layout = QVBoxLayout(eval_container)
        eval_layout.setContentsMargins(0, 0, 0, 0)
        eval_layout.setSpacing(4)
        eval_layout.addWidget(eval_label)
        eval_layout.addWidget(self.train_eval_metrics_table)
        vsplitter.addWidget(eval_container)

        vsplitter.setStretchFactor(0, 1)
        vsplitter.setStretchFactor(1, 1)

        # RIGHT: Confusion Matrix
        cm_container = QWidget()
        cm_layout = QVBoxLayout(cm_container)
        cm_layout.setContentsMargins(0, 0, 0, 0)
        cm_layout.setSpacing(4)
        cm_label = QLabel("Confusion Matrix (Validation):")
        cm_layout.addWidget(cm_label)

        self.train_cm_table = QTableWidget(0, 0)
        self.train_cm_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.train_cm_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.train_cm_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.train_cm_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.train_cm_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.train_cm_table.setMinimumWidth(320)
        self.train_cm_table.setMinimumHeight(260)
        cm_layout.addWidget(self.train_cm_table, stretch=1)

        self.train_cm_legend_label = QLabel("Cell format: normalized value (count)")
        self.train_cm_legend_label.setWordWrap(True)
        cm_layout.addWidget(self.train_cm_legend_label)

        # Add both sections to horizontal splitter
        hsplitter.addWidget(vsplitter)
        hsplitter.addWidget(cm_container)
        hsplitter.setStretchFactor(0, 2)  # Tables get more space
        hsplitter.setStretchFactor(1, 1)  # Confusion matrix

        mainLayout.addWidget(hsplitter, stretch=1)

        return tab

    def _build_review_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Gesture Filter"))
        self.review_gesture_filter = QComboBox()
        self.review_gesture_filter.addItem("All", "__all__")
        self.review_gesture_filter.currentIndexChanged.connect(
            self._render_sample_table
        )
        controls.addWidget(self.review_gesture_filter)

        self.include_quarantine_checkbox = QCheckBox("Include quarantine")
        self.include_quarantine_checkbox.toggled.connect(self.refresh_samples)
        controls.addWidget(self.include_quarantine_checkbox)

        self.refresh_samples_btn = QPushButton("Refresh")
        self.refresh_samples_btn.clicked.connect(self.refresh_samples)
        controls.addWidget(self.refresh_samples_btn)

        controls.addStretch(1)

        self.quarantine_btn = QPushButton("Quarantine Selected")
        self.quarantine_btn.clicked.connect(self.quarantine_selected)
        controls.addWidget(self.quarantine_btn)

        self.restore_btn = QPushButton("Restore Selected")
        self.restore_btn.clicked.connect(self.restore_selected)
        controls.addWidget(self.restore_btn)

        layout.addLayout(controls)

        self.review_summary_label = QLabel("Samples loaded: 0")
        layout.addWidget(self.review_summary_label)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self.samples_table = QTableWidget(0, 7)
        self.samples_table.setHorizontalHeaderLabels(
            [
                "Gesture",
                "File",
                "Rows",
                "Orientation",
                "Recorded At",
                "Source",
                "Quality",
            ]
        )
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.samples_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.samples_table.horizontalHeader().setStretchLastSection(True)
        self.samples_table.itemSelectionChanged.connect(self._on_sample_selected)

        table_host = QWidget()
        table_layout = QVBoxLayout(table_host)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.samples_table)

        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_title_label = QLabel("Select a sample to inspect traces")
        plot_layout.addWidget(self.plot_title_label)
        self.sample_trace_plot = TracePreviewWidget(minimum_height=420)
        plot_layout.addWidget(self.sample_trace_plot)

        splitter.addWidget(table_host)
        splitter.addWidget(plot_host)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([360, 520])
        layout.addWidget(splitter, stretch=1)

        return tab

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        title = QLabel("Task Logs")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, stretch=1)

        row = QHBoxLayout()
        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.clicked.connect(lambda: self.log_box.setPlainText(""))
        row.addWidget(self.clear_logs_btn)
        row.addStretch(1)
        layout.addLayout(row)

        return panel

    def _apply_window_styles(self, theme: str) -> None:
        self._theme_name = theme
        self._plot_palette = get_plot_palette(theme)
        self.setStyleSheet(build_data_manager_stylesheet(theme))
        if hasattr(self, "record_preview_plot"):
            self.record_preview_plot.set_plot_palette(self._plot_palette)
        if hasattr(self, "sample_trace_plot"):
            self.sample_trace_plot.set_plot_palette(self._plot_palette)

    def _set_status(self, message: str, level: str = "INFO") -> None:
        self.status_banner.setStyleSheet(
            get_status_banner_style(level, self._theme_name)
        )
        self.status_banner.setText(message)
        self.statusBar().showMessage(message, 3500)

    def _set_task_state(self, active: bool, task_name: str = "") -> None:
        self._task_active = active
        self._task_name = task_name
        enabled = not active

        self.record_btn.setEnabled(enabled)
        self.record_port_combo.setEnabled(enabled)
        self.record_refresh_ports_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
        self.train_btn.setEnabled(enabled)
        self.record_stop_btn.setEnabled(active and task_name == "record_sample")
        self.train_cancel_btn.setEnabled(active and task_name == "train_model")

        if active:
            self._set_status(f"Running task: {task_name}", "INFO")
        else:
            self._set_status("Ready", "INFO")

    def _poll_events(self) -> None:
        live_preview_rows: list[dict[str, float | int]] | None = None

        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event.get("type")

            # ------ Recording events ------
            if event_type == "record_started":
                port = str(event.get("port", "unknown"))
                self.recording_status_label.setText(
                    f"Recording status: capturing on {port}"
                )
                continue

            if event_type == "record_progress":
                row_count = int(event.get("row_count", 0))
                elapsed_seconds = float(event.get("elapsed_seconds", 0.0))
                self.record_row_count_label.setText(f"Rows captured: {row_count}")
                self.recording_status_label.setText(
                    f"Recording status: capturing ({elapsed_seconds:.1f}s)"
                )
                rows = event.get("rows")
                if isinstance(rows, list) and rows:
                    live_preview_rows = rows
                continue

            if event_type == "record_error":
                message = str(event.get("message", "Recording failed"))
                live_preview_rows = None
                self._set_task_state(False)
                self._set_status(f"Recording failed: {message}", "ERROR")
                self.recording_status_label.setText("Recording status: error")
                continue

            if event_type == "record_warning":
                message = str(event.get("message", "")).strip()
                if message:
                    self._set_status(message, "WARNING")
                continue

            if event_type == "record_ready_for_review":
                gesture = str(event.get("gesture", ""))
                orientation = str(event.get("orientation", "unspecified"))
                row_count = int(event.get("row_count", 0))
                elapsed_seconds = float(event.get("elapsed_seconds", 0.0))
                rows = event.get("rows")

                self._set_task_state(False)
                self.record_row_count_label.setText(f"Rows captured: {row_count}")

                if not isinstance(rows, list) or not rows:
                    live_preview_rows = None
                    self._set_status(
                        "No valid sensor rows captured. Recording discarded.", "WARNING"
                    )
                    self.recording_status_label.setText("Recording status: discarded")
                    continue

                live_preview_rows = None
                self._plot_recording_preview(rows)
                decision = QMessageBox.question(
                    self,
                    "Save Recording",
                    f"Preview ready for gesture '{gesture}'.\n\nKeep and save this recording?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if decision == QMessageBox.StandardButton.Yes:
                    path = self.recording_service.save_recording(
                        gesture_label=gesture,
                        orientation=orientation,
                        rows=rows,
                        elapsed_seconds=elapsed_seconds,
                    )
                    self.refresh_samples()
                    self._refresh_record_count()
                    self._set_status("Recording saved", "INFO")
                    self.recording_status_label.setText("Recording status: completed")
                    self.logger.info("[record] Saved path: %s", path)
                else:
                    self._set_status("Recording discarded by user", "WARNING")
                    self.recording_status_label.setText("Recording status: discarded")
                continue

            # ------ Processing events ------
            if event_type == "process_started":
                self.process_status_label.setText("Status: running")
                continue

            if event_type == "process_total_gestures":
                total = int(event.get("total", 0))
                self._process_total_gestures = total
                self.process_progress_bar.setRange(0, 100)
                self.process_progress_bar.setValue(0)
                self.process_progress_bar.setFormat("Progress: 0%")
                continue

            if event_type == "process_gesture_summary":
                gesture = str(event.get("gesture", ""))
                files = int(event.get("files", 0))
                samples = int(event.get("samples", 0))
                self._set_process_table_value(gesture, 1, str(files))
                self._set_process_table_value(gesture, 2, str(samples))
                continue

            if event_type == "process_gesture_current":
                gesture = str(event.get("gesture", ""))
                self._process_current_gesture = gesture
                self._process_seen_gestures.add(gesture)
                if self._process_total_gestures > 0:
                    pct = int(
                        (len(self._process_seen_gestures) / self._process_total_gestures) * 100
                    )
                    self.process_progress_bar.setValue(max(0, min(100, pct)))
                    self.process_progress_bar.setFormat(f"Progress: {pct}%")
                self.process_status_label.setText(f"Status: processing {gesture}")
                self.process_events_box.appendPlainText(f"Processing gesture: '{gesture}'")
                continue

            if event_type == "process_train_sequences":
                gesture = str(event.get("gesture", ""))
                count = int(event.get("count", 0))
                self._set_process_table_value(gesture, 3, str(count))
                self.process_events_box.appendPlainText(
                    f"  Training: {count} sequences"
                )
                continue

            if event_type == "process_test_sequences":
                gesture = str(event.get("gesture", ""))
                count = int(event.get("count", 0))
                self._set_process_table_value(gesture, 4, str(count))
                self.process_events_box.appendPlainText(
                    f"  Test: {count} sequences"
                )
                continue

            if event_type == "process_progress":
                done = int(event.get("done", 0))
                total = max(int(event.get("total", 1)), 1)
                pct = int((done / total) * 100)
                self.process_progress_bar.setRange(0, 100)
                self.process_progress_bar.setValue(max(0, min(100, pct)))
                self.process_progress_bar.setFormat(f"Progress: {pct}%")
                self.process_status_label.setText(f"Status: processed {done}/{total}")
                continue

            if event_type == "process_completed":
                processed = int(event.get("processed", 0))
                total = int(event.get("total", 0))
                self.process_progress_bar.setRange(0, 100)
                self.process_progress_bar.setValue(100)
                self.process_progress_bar.setFormat("Progress: 100%")
                self.process_status_label.setText("Status: completed")
                self._set_status(f"Processing complete: {processed}/{total} gesture(s)", "INFO")
                self._set_task_state(False)
                self.refresh_samples()
                self._refresh_record_count()
                continue

            if event_type == "process_failed":
                message = str(event.get("message", "Processing failed"))
                self.process_progress_bar.setRange(0, 100)
                self.process_progress_bar.setValue(0)
                self.process_progress_bar.setFormat("Progress: failed")
                self.process_status_label.setText("Status: failed")
                self._set_status(f"Processing failed: {message}", "ERROR")
                self._set_task_state(False)
                continue

            if event_type == "process_cancelled":
                self.process_progress_bar.setRange(0, 100)
                self.process_progress_bar.setValue(0)
                self.process_progress_bar.setFormat("Progress: cancelled")
                self.process_status_label.setText("Status: cancelled")
                self._set_status("Processing cancelled", "WARNING")
                self._set_task_state(False)
                continue

            # ------ Training events ------
            if event_type == "train_started":
                self.train_status_label.setText("Status: running")
                continue

            if event_type == "train_epoch":
                epoch = int(event.get("epoch", 0))
                total = int(event.get("total", 1))
                train_loss = float(event.get("train_loss", 0.0))
                train_acc = float(event.get("train_acc", 0.0))
                val_loss = float(event.get("val_loss", 0.0))
                val_acc = float(event.get("val_acc", 0.0))
                lr = float(event.get("lr", 0.0))
                self._train_total_epochs = total
                pct = int((epoch / max(total, 1)) * 100)
                self.train_progress_bar.setRange(0, 100)
                self.train_progress_bar.setValue(max(0, min(100, pct)))
                self.train_progress_bar.setFormat(f"Epoch progress: {epoch}/{total}")
                self.train_status_label.setText(f"Status: epoch {epoch}/{total}")
                self._set_train_metric("train_loss", f"{train_loss:.4f}")
                self._set_train_metric("train_acc", f"{train_acc:.2f}%")
                self._set_train_metric("val_loss", f"{val_loss:.4f}")
                self._set_train_metric("val_acc", f"{val_acc:.2f}%")
                self._set_train_metric("lr", f"{lr:.6f}")
                continue

            if event_type == "train_model_dir":
                self._train_model_dir = str(event.get("model_dir", ""))
                continue

            if event_type == "train_completed":
                self._set_train_metric("Result", "Training complete")
                self.train_progress_bar.setRange(0, 100)
                self.train_progress_bar.setValue(100)
                self.train_progress_bar.setFormat("Epoch progress: 100%")
                self.train_status_label.setText("Status: completed")
                self._set_status("Training complete", "INFO")
                if self._train_model_dir:
                    self._load_train_evaluation_metrics()
                self._set_task_state(False)
                self.refresh_samples()
                self._refresh_record_count()
                continue

            if event_type == "train_cancelled":
                self.train_progress_bar.setRange(0, 100)
                self.train_progress_bar.setValue(0)
                self.train_progress_bar.setFormat("Epoch progress: cancelled")
                self.train_status_label.setText("Status: cancelled")
                self._set_status("Training cancelled", "WARNING")
                self._set_task_state(False)
                continue

            if event_type == "train_failed":
                message = str(event.get("message", "Training failed"))
                self.train_progress_bar.setRange(0, 100)
                self.train_progress_bar.setValue(0)
                self.train_progress_bar.setFormat("Epoch progress: failed")
                self.train_status_label.setText("Status: failed")
                self._set_status(f"Training failed: {message}", "ERROR")
                self._set_task_state(False)
                continue

            # ------ Log events (display only) ------
            if event_type != "log":
                continue

            level = str(event.get("level", "INFO"))
            timestamp = str(event.get("timestamp", ""))
            source = str(event.get("source", "app"))
            message = str(event.get("message", ""))
            self.log_box.appendPlainText(f"[{timestamp}] [{level}] {source}: {message}")

        if live_preview_rows is not None:
            self._plot_recording_preview(live_preview_rows, force=False)

    def _load_gesture_options(self) -> None:
        gestures = load_gesture_names(self.project_root)

        current = self.record_gesture_combo.currentText().strip()
        self.record_gesture_combo.blockSignals(True)
        self.record_gesture_combo.clear()
        for gesture in gestures:
            self.record_gesture_combo.addItem(gesture)
        if current:
            if self.record_gesture_combo.findText(current) < 0:
                self.record_gesture_combo.addItem(current)
            self.record_gesture_combo.setCurrentText(current)
        elif self.record_gesture_combo.count() > 0:
            self.record_gesture_combo.setCurrentIndex(0)
        self.record_gesture_combo.blockSignals(False)

        selected_filter = self.review_gesture_filter.currentData()
        self.review_gesture_filter.blockSignals(True)
        self.review_gesture_filter.clear()
        self.review_gesture_filter.addItem("All", "__all__")
        for gesture in sorted({sample.gesture for sample in self.samples}):
            self.review_gesture_filter.addItem(gesture, gesture)
        if isinstance(selected_filter, str):
            idx = self.review_gesture_filter.findData(selected_filter)
            if idx >= 0:
                self.review_gesture_filter.setCurrentIndex(idx)
        self.review_gesture_filter.blockSignals(False)

    def _selected_record_port(self) -> str:
        data = self.record_port_combo.currentData()
        if isinstance(data, str):
            return data.strip()
        return ""

    def refresh_record_ports(self, announce: bool = True) -> None:
        current = self._selected_record_port()
        entries = SerialService.list_port_entries()

        self.record_port_combo.blockSignals(True)
        self.record_port_combo.clear()

        if not entries:
            self.record_port_combo.addItem("No serial ports found")
            self.record_port_combo.blockSignals(False)
            if announce:
                self._set_status("No serial ports found", "WARNING")
            return

        ports = [device for _label, device in entries]
        for label, device in entries:
            self.record_port_combo.addItem(label, device)

        preferred_port = ""
        for label, device in entries:
            if "USB-SERIAL CH340" in label.upper():
                preferred_port = device
                break

        if not preferred_port:
            preferred_port = select_serial_port(current if current in ports else None)

        if preferred_port and preferred_port in ports:
            idx = self.record_port_combo.findData(preferred_port)
            if idx >= 0:
                self.record_port_combo.setCurrentIndex(idx)
        else:
            idx = self.record_port_combo.findData(current)
            if idx >= 0:
                self.record_port_combo.setCurrentIndex(idx)
            else:
                self.record_port_combo.setCurrentIndex(0)

        self.record_port_combo.blockSignals(False)
        if announce:
            self._set_status("Serial ports refreshed", "INFO")

    def start_recording(self) -> None:
        if self._task_active:
            self._set_status("Another task is already running", "WARNING")
            return

        gesture = self.record_gesture_combo.currentText().strip()
        if not gesture:
            self._set_status("Please provide a gesture name", "WARNING")
            return
        gesture = sanitize_gesture_label(gesture)
        self.record_gesture_combo.setCurrentText(gesture)
        selected_port = self._selected_record_port()

        config = RecordingConfig(
            gesture_label=gesture,
            orientation="unspecified",
            port=selected_port or None,
        )

        if not self.recording_service.start(config):
            self._set_status("Could not start recording. Recorder is busy.", "WARNING")
            return
        self._last_record_preview_update = 0.0
        self.record_preview_plot.clear_plot()
        self.record_row_count_label.setText("Rows captured: 0")
        self.recording_status_label.setText("Recording status: starting")
        self._set_task_state(True, "record_sample")
        self._set_status("Recording started", "INFO")

    def stop_recording(self) -> None:
        if not self.recording_service.is_running:
            return
        self.recording_service.stop()
        self.recording_status_label.setText("Recording status: stopping")
        self._set_status("Stopping recording...", "WARNING")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.recording_service.stop()
        self.recording_service.join(timeout=1.5)
        self.serial_service.disconnect()
        event.accept()

    def _plot_recording_preview(
        self,
        rows: list[dict[str, float | int]],
        *,
        force: bool = True,
    ) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self._last_record_preview_update
            < self._record_preview_update_interval
        ):
            return

        self.record_preview_plot.plot_rows(rows)
        self._last_record_preview_update = now

    def _reset_process_ui(self) -> None:
        self._process_total_gestures = 0
        self._process_seen_gestures.clear()
        self._process_current_gesture = ""
        self._process_table_rows.clear()
        self.process_status_label.setText("Status: running")
        self.process_progress_bar.setRange(0, 0)
        self.process_progress_bar.setFormat("Progress: preparing...")
        self.process_results_table.setRowCount(0)
        self.process_events_box.setPlainText("")

    def _reset_train_ui(self) -> None:
        self._train_total_epochs = 0
        self._train_result_rows.clear()
        self._train_metrics.clear()
        self._train_model_dir = ""
        self.train_status_label.setText("Status: running")
        self.train_progress_bar.setRange(0, 0)
        self.train_progress_bar.setFormat("Epoch progress: preparing...")
        self.train_loss_box.setPlainText("")
        self.train_acc_box.setPlainText("")
        self.val_loss_box.setPlainText("")
        self.val_acc_box.setPlainText("")
        self.train_lr_box.setPlainText("")
        self.train_best_val_box.setPlainText("")
        self.train_results_table.setRowCount(0)
        self.train_eval_metrics_table.setRowCount(0)
        self.train_cm_table.clearContents()
        self.train_cm_table.setRowCount(0)
        self.train_cm_table.setColumnCount(0)
        self.train_cm_legend_label.setText("Cell format: normalized value (count)")

    def _set_process_table_value(self, gesture: str, column: int, value: str) -> None:
        if gesture not in self._process_table_rows:
            row = self.process_results_table.rowCount()
            self.process_results_table.insertRow(row)
            self.process_results_table.setItem(row, 0, QTableWidgetItem(gesture))
            for idx in range(1, 5):
                self.process_results_table.setItem(row, idx, QTableWidgetItem("-"))
            self._process_table_rows[gesture] = row
        row = self._process_table_rows[gesture]
        self.process_results_table.setItem(row, column, QTableWidgetItem(value))

    def _set_train_metric(self, key: str, value: str) -> None:
        """Update training metric display based on metric key."""
        self._train_metrics[key] = value

        if key == "train_loss":
            self.train_loss_box.setPlainText(value)
        elif key == "train_acc":
            self.train_acc_box.setPlainText(value)
        elif key == "val_loss":
            self.val_loss_box.setPlainText(value)
        elif key == "val_acc":
            self.val_acc_box.setPlainText(value)
            # Also update best val acc if this is better
            try:
                current_val = float(value.rstrip("%"))
                best_val_text = self.train_best_val_box.toPlainText()
                if best_val_text:
                    best_val = float(best_val_text.rstrip("%"))
                    if current_val > best_val:
                        self.train_best_val_box.setPlainText(value)
                else:
                    self.train_best_val_box.setPlainText(value)
            except (ValueError, AttributeError):
                pass
        elif key == "lr":
            self.train_lr_box.setPlainText(value)
        elif key in ["Best Val Acc", "Result", "Mode", "Test Accuracy"]:
            # Add to final results table
            if key not in self._train_result_rows:
                row = self.train_results_table.rowCount()
                self.train_results_table.insertRow(row)
                self.train_results_table.setItem(row, 0, QTableWidgetItem(key))
                self._train_result_rows[key] = row
            row = self._train_result_rows[key]
            self.train_results_table.setItem(row, 1, QTableWidgetItem(value))

    def _load_train_evaluation_metrics(self) -> None:
        """Load and display evaluation metrics and confusion matrix from saved JSON."""
        try:
            eval_dir = Path(self._train_model_dir) / "evaluation"
            metrics_file = eval_dir / "metrics.json"

            if not metrics_file.exists():
                self.train_status_label.setText("Status: metrics file not found")
                return

            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            # Populate evaluation metrics table with validation metrics
            if "validation" in metrics_data:
                val_metrics = metrics_data["validation"]
                self.train_eval_metrics_table.setRowCount(0)

                metrics_to_display = [
                    ("Validation Accuracy", f"{val_metrics['accuracy']:.4f}"),
                    ("Precision", f"{val_metrics['precision']:.4f}"),
                    ("Recall", f"{val_metrics['recall']:.4f}"),
                    ("F1-Score", f"{val_metrics['f1_score']:.4f}"),
                ]

                if "test" in metrics_data:
                    test_metrics = metrics_data["test"]
                    metrics_to_display.extend(
                        [
                            ("Test Accuracy", f"{test_metrics['accuracy']:.4f}"),
                            ("Test Precision", f"{test_metrics['precision']:.4f}"),
                            ("Test Recall", f"{test_metrics['recall']:.4f}"),
                            ("Test F1-Score", f"{test_metrics['f1_score']:.4f}"),
                        ]
                    )

                for idx, (label, value) in enumerate(metrics_to_display):
                    self.train_eval_metrics_table.insertRow(idx)
                    self.train_eval_metrics_table.setItem(
                        idx, 0, QTableWidgetItem(label)
                    )
                    self.train_eval_metrics_table.setItem(
                        idx, 1, QTableWidgetItem(value)
                    )

                # Display confusion matrix
                if "confusion_matrix" in val_metrics and "class_names" in val_metrics:
                    cm = np.array(val_metrics["confusion_matrix"])
                    class_names = val_metrics["class_names"]
                    self._plot_confusion_matrix(cm, class_names)

            self.train_status_label.setText("Status: metrics loaded successfully")

        except Exception as e:
            self.train_status_label.setText(f"Status: error loading metrics - {str(e)}")
            self.logger.error(f"Failed to load training metrics: {e}")

    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list[str]) -> None:
        """Render confusion matrix using a custom Qt table widget."""
        matrix = np.asarray(cm, dtype=float)
        if matrix.size == 0 or not class_names:
            self.train_cm_table.clearContents()
            self.train_cm_table.setRowCount(0)
            self.train_cm_table.setColumnCount(0)
            self.train_cm_legend_label.setText("Cell format: normalized value (count)")
            return

        # Normalize each row safely to avoid divide-by-zero.
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_normalized = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix, dtype=float),
                where=row_sums != 0,
            )

        size = len(class_names)
        self.train_cm_table.setRowCount(size)
        self.train_cm_table.setColumnCount(size)
        self.train_cm_table.setHorizontalHeaderLabels(class_names)
        self.train_cm_table.setVerticalHeaderLabels(class_names)

        for row_idx in range(size):
            for col_idx in range(size):
                ratio = float(cm_normalized[row_idx, col_idx])
                count = int(matrix[row_idx, col_idx])
                item = QTableWidgetItem(f"{ratio:.2f}\n({count})")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                color = self._confusion_cell_color(ratio)
                item.setBackground(QBrush(color))
                text_color = get_confusion_text_color(ratio, self._theme_name)
                item.setForeground(QBrush(text_color))
                self.train_cm_table.setItem(row_idx, col_idx, item)

        total = float(matrix.sum())
        accuracy = float(np.trace(matrix) / total) if total > 0 else 0.0
        self.train_cm_legend_label.setText(
            f"Cell format: normalized value (count) | Overall accuracy: {accuracy:.2%}"
        )

    def _confusion_cell_color(self, ratio: float) -> QColor:
        """Resolve confusion-matrix background color from theme manager."""
        return get_confusion_cell_color(ratio, self._theme_name)

    def run_processing(self) -> None:
        """Start the data processing pipeline via the native service."""
        if self._task_active:
            self._set_status("Another task is already running", "WARNING")
            return
        self._reset_process_ui()
        if not self.data_processing_service.start():
            self.process_status_label.setText("Status: could not start")
            self._set_status("Processing service is already running", "WARNING")
            return
        self._set_task_state(True, "process_data")
        self._set_status("Processing started", "INFO")

    def start_training(self) -> None:
        """Start the training pipeline via the native service."""
        if self._task_active:
            self._set_status("Another task is already running", "WARNING")
            return
        self._reset_train_ui()
        overrides = TrainingOverrides(
            epochs=self.train_epochs_spin.value(),
            learning_rate=self.train_lr_spin.value(),
            batch_size=self.train_batch_spin.value(),
            early_stopping_patience=self.train_patience_spin.value(),
        )
        if not self.training_service.start(overrides):
            self.train_status_label.setText("Status: could not start")
            self._set_status("Training service is already running", "WARNING")
            return
        self._set_task_state(True, "train_model")
        self._set_status("Training started", "INFO")

    def cancel_training(self) -> None:
        """Request cancellation of an in-progress training run."""
        self.training_service.cancel()
        self.train_cancel_btn.setEnabled(False)
        self._set_status("Cancellation requested — stopping after current epoch…", "WARNING")

    def _refresh_record_count(self) -> None:
        gesture = self.record_gesture_combo.currentText().strip()
        if not gesture:
            self.record_stats_label.setText("Samples for selected gesture: 0")
            return
        sample_count = count_csv_samples(gesture, base_dir=str(self.raw_data_root))
        self.record_stats_label.setText(f"Samples for selected gesture: {sample_count}")

    def refresh_samples(self) -> None:
        include_quarantine = self.include_quarantine_checkbox.isChecked()
        self.samples = self.sample_service.list_samples(
            include_quarantine=include_quarantine
        )
        self._load_gesture_options()
        self._render_sample_table()

    def _render_sample_table(self) -> None:
        selected_filter = self.review_gesture_filter.currentData()
        samples = self.samples
        if isinstance(selected_filter, str) and selected_filter != "__all__":
            samples = [item for item in samples if item.gesture == selected_filter]

        self.samples_table.setRowCount(len(samples))
        for row, sample in enumerate(samples):
            quality = self._sample_quality(sample)
            values = [
                sample.gesture,
                sample.file_name,
                str(sample.row_count),
                sample.orientation,
                sample.recorded_at,
                sample.source,
                quality,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in {2}:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.samples_table.setItem(row, col, item)

            path_item = QTableWidgetItem(str(sample.csv_path))
            self.samples_table.setVerticalHeaderItem(row, path_item)

        self.review_summary_label.setText(f"Samples loaded: {len(samples)}")

        self.quarantine_btn.setEnabled(False)
        self.restore_btn.setEnabled(False)
        self.sample_trace_plot.clear_plot()

    @staticmethod
    def _sample_quality(sample: SampleRecord) -> str:
        if sample.row_count <= 0:
            return "NO_DATA"
        if sample.row_count < 20:
            return "LOW_ROWS"
        return "OK"

    def _selected_sample(self) -> SampleRecord | None:
        row = self.samples_table.currentRow()
        if row < 0:
            return None

        path_item = self.samples_table.verticalHeaderItem(row)
        if path_item is None:
            return None

        selected_path = Path(path_item.text())
        for sample in self.samples:
            if sample.csv_path == selected_path:
                return sample
        return None

    def _on_sample_selected(self) -> None:
        sample = self._selected_sample()
        if sample is None:
            self.quarantine_btn.setEnabled(False)
            self.restore_btn.setEnabled(False)
            return

        self.quarantine_btn.setEnabled(sample.source == "raw")
        self.restore_btn.setEnabled(sample.source == "quarantine")
        self._plot_sample(sample)

    def _plot_sample(self, sample: SampleRecord) -> None:
        self.plot_title_label.setText(f"Trace Preview: {sample.file_name}")

        try:
            data = pd.read_csv(sample.csv_path)
        except Exception as exc:
            self._set_status(f"Could not read sample: {exc}", "ERROR")
            self.sample_trace_plot.clear_plot()
            return

        self.sample_trace_plot.plot_dataframe(data)

    def quarantine_selected(self) -> None:
        sample = self._selected_sample()
        if sample is None:
            return
        if sample.source != "raw":
            self._set_status("Only raw samples can be quarantined", "WARNING")
            return

        response = QMessageBox.question(
            self,
            "Quarantine Sample",
            f"Move sample to quarantine?\n\n{sample.file_name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        moved = self.sample_service.quarantine_sample(sample.csv_path)
        self.logger.info("Quarantined sample: %s", moved)
        self._set_status("Sample moved to quarantine", "INFO")
        self.refresh_samples()

    def restore_selected(self) -> None:
        sample = self._selected_sample()
        if sample is None:
            return
        if sample.source != "quarantine":
            self._set_status("Select a quarantined sample to restore", "WARNING")
            return

        response = QMessageBox.question(
            self,
            "Restore Sample",
            f"Restore sample back to raw dataset?\n\n{sample.file_name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        moved = self.sample_service.restore_sample(sample.csv_path)
        self.logger.info("Restored sample: %s", moved)
        self._set_status("Sample restored to raw dataset", "INFO")
        self.refresh_samples()


def run_data_manager(project_root: Path) -> None:
    """Open dataset manager window."""
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    window = DataManagerWindow(project_root=project_root)
    theme_loaded = apply_custom_widgets_theme(
        window,
        project_root,
    )
    if not theme_loaded:
        raise RuntimeError(
            "CustomWidgets theme could not be loaded for data manager startup. "
            "Verify QT-PyQt-PySide-Custom-Widgets is installed and the style JSON is present."
        )
    window.show()

    if owns_app:
        app.exec()
