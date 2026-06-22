"""Tab builder mixin for DataManagerWindow.

Provides tab construction methods used by the main window:
_record, _process, _train, _review tabs and the log panel.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.ui.stage_widget import StageWidget
from gui.ui.trace_preview_widget import TracePreviewWidget
from config.architecture import architecture


class DataManagerTabsMixin:
    """Mixin providing tab-builder methods for DataManagerWindow.

    These methods create and populate the Record, Process, Train, and
    Review tabs plus the log panel.  They rely on ``self`` being a
    fully-initialised DataManagerWindow instance at runtime (standard
    Python mixin pattern).
    """

    # ---- Record tab ---------------------------------------------------------

    def _build_record_tab(self) -> QWidget:
        """Build the Record tab for capturing new samples."""
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

        # Provide a small manager button to edit the gestures list
        self.manage_gestures_btn = QPushButton("Manage")
        self.manage_gestures_btn.setToolTip("Edit gestures list")
        self.manage_gestures_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.manage_gestures_btn.clicked.connect(self._open_gestures_editor)

        gesture_host = QWidget()
        gesture_layout = QHBoxLayout(gesture_host)
        gesture_layout.setContentsMargins(0, 0, 0, 0)
        gesture_layout.setSpacing(6)
        gesture_layout.addWidget(self.record_gesture_combo)
        gesture_layout.addWidget(self.manage_gestures_btn)

        form.addRow("Gesture", gesture_host)

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

    # ---- Process tab --------------------------------------------------------

    def _build_process_tab(self) -> QWidget:
        """Build the process tab with dedicated stage widgets.

        Each stage has its own small widget (StageWidget) that presents
        status/progress/metrics and actionable Retry/Skip buttons.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        header = QLabel("Processing Pipeline — Live Telemetry")
        header.setObjectName("panelTitle")
        layout.addWidget(header)

        # Top action row
        action_row = QHBoxLayout()
        self.process_btn = QPushButton("Run Processing")
        self.process_btn.clicked.connect(self.run_processing)
        action_row.addWidget(self.process_btn)

        self.process_cancel_btn = QPushButton("Cancel")
        self.process_cancel_btn.clicked.connect(
            lambda: self.data_processing_service.stop()
        )
        action_row.addWidget(self.process_cancel_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        # Stage widgets grid
        stages_layout = QGridLayout()
        stages_layout.setSpacing(8)

        self.stage_file_ingest = StageWidget("File Ingestion")
        self.stage_smoothing = StageWidget("Signal Smoothing")
        self.stage_augmentation = StageWidget("Data Augmentation")
        self.stage_feature = StageWidget("Feature Extraction")
        self.stage_tensor = StageWidget("Tensor Formatting")
        self.stage_save_train = StageWidget("Save Training Sequences")
        self.stage_save_test = StageWidget("Save Test Sequences")

        stages = [
            self.stage_file_ingest,
            self.stage_smoothing,
            self.stage_augmentation,
            self.stage_feature,
            self.stage_tensor,
            self.stage_save_train,
            self.stage_save_test,
        ]

        # Connect stage actions to backend controls
        for w in stages:
            w.retry_btn.clicked.connect(
                self.data_processing_service.retry_current_stage
            )
            w.skip_btn.clicked.connect(self.data_processing_service.skip_current_stage)

        stages_layout.addWidget(self.stage_file_ingest, 0, 0)
        stages_layout.addWidget(self.stage_smoothing, 0, 1)
        stages_layout.addWidget(self.stage_augmentation, 1, 0)
        stages_layout.addWidget(self.stage_feature, 1, 1)
        stages_layout.addWidget(self.stage_tensor, 2, 0)
        stages_layout.addWidget(self.stage_save_train, 2, 1)
        stages_layout.addWidget(self.stage_save_test, 3, 0)

        layout.addLayout(stages_layout)

        # Results table (keeps previous functionality)
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

        # Compact summary
        self.process_summary_label = QLabel(
            f"Raw directory: {self.raw_data_root}\n"
            f"Processed output: {self.project_root / 'data' / 'processed'}"
        )
        self.process_summary_label.setWordWrap(True)
        layout.addWidget(self.process_summary_label)

        return tab

    # ---- Train tab ----------------------------------------------------------

    def _build_train_tab(self) -> QWidget:
        """Build the Train tab for model training configuration and monitoring."""
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
        self.train_epochs_spin.setValue(architecture.training.epochs)
        self.train_epochs_spin.setToolTip(
            f"Max training epochs (config default: {architecture.training.epochs})"
        )
        override_form.addRow("Max Epochs:", self.train_epochs_spin)

        self.train_lr_spin = QDoubleSpinBox()
        self.train_lr_spin.setRange(0.00001, 0.1)
        self.train_lr_spin.setDecimals(6)
        self.train_lr_spin.setSingleStep(0.0001)
        self.train_lr_spin.setValue(architecture.training.learning_rate)
        self.train_lr_spin.setToolTip(
            f"Initial learning rate (config default: {architecture.training.learning_rate})"
        )
        override_form.addRow("Learning Rate:", self.train_lr_spin)

        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(1, 512)
        self.train_batch_spin.setValue(architecture.training.batch_size)
        self.train_batch_spin.setToolTip(
            f"Mini-batch size (config default: {architecture.training.batch_size})"
        )
        override_form.addRow("Batch Size:", self.train_batch_spin)

        self.train_patience_spin = QSpinBox()
        self.train_patience_spin.setRange(1, 200)
        self.train_patience_spin.setValue(architecture.training.early_stopping_patience)
        self.train_patience_spin.setToolTip(
            f"Early stopping patience in epochs (config default: {architecture.training.early_stopping_patience})"
        )
        override_form.addRow("Early Stop Patience:", self.train_patience_spin)

        self.train_ensemble_check = QCheckBox("Train as Ensemble")
        self.train_ensemble_check.setChecked(architecture.training.use_ensemble)
        self.train_ensemble_check.setToolTip(
            f"Train an ensemble of multiple models (config default: {architecture.training.use_ensemble})"
        )
        override_form.addRow("Ensemble Mode:", self.train_ensemble_check)

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

    # ---- Review tab ---------------------------------------------------------

    def _build_review_tab(self) -> QWidget:
        """Build the Review and Cleanup tab for inspecting samples."""
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
        self.samples_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection
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

    # ---- Log panel ----------------------------------------------------------

    def _build_log_panel(self) -> QWidget:
        """Build the task log panel widget."""
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
