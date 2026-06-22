"""PySide6 management GUI for dataset and model lifecycle tasks."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QBrush, QKeySequence, QColor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config.config import (
    DATA_MANAGER_MIN_HEIGHT,
    DATA_MANAGER_MIN_WIDTH,
    DATA_MANAGER_WINDOW_HEIGHT,
    DATA_MANAGER_WINDOW_WIDTH,
    LOGS_OUTPUT_DIR,
    RAW_DATA_DIR,
)
from gui.services.data_processing_service import DataProcessingService
from gui.services.logging_service import configure_gui_logger
from gui.services.recording_service import RecordingConfig, RecordingService
from gui.services.sample_review_service import SampleRecord, SampleReviewService
from gui.services.serial_service import SerialService
from gui.services.training_service import TrainingOverrides, TrainingService
from gui.ui.custom_widgets_adapter import apply_custom_widgets_theme
from gui.ui.data_manager_event_handlers import DataManagerEventHandlersMixin
from gui.ui.data_manager_tabs import DataManagerTabsMixin
from gui.ui.gestures_editor_dialog import GesturesEditorDialog
from gui.ui.theme_manager import (
    build_data_manager_stylesheet,
    get_confusion_cell_color,
    get_confusion_text_color,
    get_plot_palette,
    get_status_banner_style,
)
from gui.utils.icon_utils import apply_app_icon, resolve_app_icon_path
from utils.recording_utils import (
    count_csv_samples,
    load_gesture_names,
    sanitize_gesture_label,
)
from utils.serial_utils import select_serial_port


class DataManagerWindow(DataManagerTabsMixin, DataManagerEventHandlersMixin, QMainWindow):
    """Standalone GUI for recording, processing, training, and data cleaning."""

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        apply_app_icon(self, self.project_root)
        self.raw_data_root = Path(RAW_DATA_DIR)
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
        self._live_preview_rows: list[dict[str, float | int]] | None = None

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
        self.resize(DATA_MANAGER_WINDOW_WIDTH, DATA_MANAGER_WINDOW_HEIGHT)
        self.setMinimumSize(DATA_MANAGER_MIN_WIDTH, DATA_MANAGER_MIN_HEIGHT)

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

    # ---- Gesture options ---------------------------------------------------

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

    def _open_gestures_editor(self) -> None:
        """Open a dialog allowing the user to edit the gestures list (name + translation)."""
        dialog = GesturesEditorDialog(parent=self, project_root=self.project_root)
        result = dialog.exec()
        if result == QDialog.Accepted:
            # Refresh options after a successful save
            self._load_gesture_options()

    # ---- Recording helpers -------------------------------------------------

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

    # ---- Process helpers ---------------------------------------------------

    def _reset_process_ui(self) -> None:
        self._process_total_gestures = 0
        self._process_seen_gestures.clear()
        self._process_current_gesture = ""
        self._process_table_rows.clear()
        # Reset stage widgets
        for w in (
            self.stage_file_ingest,
            self.stage_smoothing,
            self.stage_augmentation,
            self.stage_feature,
            self.stage_tensor,
            self.stage_save_train,
            self.stage_save_test,
        ):
            w.set_metrics(None)
            w.status_label.setText("Idle")
            w.progress.setRange(0, 100)
            w.progress.setValue(0)
        self.process_results_table.setRowCount(0)

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

                for idx, (label_text, metric_value) in enumerate(metrics_to_display):
                    self.train_eval_metrics_table.insertRow(idx)
                    self.train_eval_metrics_table.setItem(
                        idx, 0, QTableWidgetItem(label_text)
                    )
                    self.train_eval_metrics_table.setItem(
                        idx, 1, QTableWidgetItem(metric_value)
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

    # ---- Processing --------------------------------------------------------

    def run_processing(self) -> None:
        """Start the data processing pipeline via the native service."""
        if self._task_active:
            self._set_status("Another task is already running", "WARNING")
            return
        self._reset_process_ui()
        if not self.data_processing_service.start():
            self._set_status("Processing service is already running", "WARNING")
            return
        self._set_task_state(True, "process_data")
        self._set_status("Processing started", "INFO")

    # ---- Training ----------------------------------------------------------

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
            use_ensemble=self.train_ensemble_check.isChecked(),
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
        self._set_status(
            "Cancellation requested — stopping after current epoch…", "WARNING"
        )

    # ---- Record stats ------------------------------------------------------

    def _refresh_record_count(self) -> None:
        gesture = self.record_gesture_combo.currentText().strip()
        if not gesture:
            self.record_stats_label.setText("Samples for selected gesture: 0")
            return
        sample_count = count_csv_samples(gesture, base_dir=str(self.raw_data_root))
        self.record_stats_label.setText(f"Samples for selected gesture: {sample_count}")

    # ---- Sample review -----------------------------------------------------

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

    def _sample_for_row(self, row: int) -> SampleRecord | None:
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

    def _selected_samples(self) -> list[SampleRecord]:
        rows = sorted({index.row() for index in self.samples_table.selectedIndexes()})
        if not rows and self.samples_table.currentRow() >= 0:
            rows = [self.samples_table.currentRow()]

        selected: list[SampleRecord] = []
        seen_paths: set[Path] = set()
        for row in rows:
            sample = self._sample_for_row(row)
            if sample is None or sample.csv_path in seen_paths:
                continue
            selected.append(sample)
            seen_paths.add(sample.csv_path)
        return selected

    def _selected_sample(self) -> SampleRecord | None:
        current_sample = self._sample_for_row(self.samples_table.currentRow())
        if current_sample is not None:
            return current_sample

        samples = self._selected_samples()
        return samples[0] if samples else None

    def _on_sample_selected(self) -> None:
        selected_samples = self._selected_samples()
        if not selected_samples:
            self.quarantine_btn.setEnabled(False)
            self.restore_btn.setEnabled(False)
            return

        self.quarantine_btn.setEnabled(
            all(sample.source == "raw" for sample in selected_samples)
        )
        self.restore_btn.setEnabled(
            all(sample.source == "quarantine" for sample in selected_samples)
        )

        sample = self._selected_sample()
        if sample is None:
            return
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
        samples = self._selected_samples()
        if not samples:
            return
        if not all(sample.source == "raw" for sample in samples):
            self._set_status("Only raw samples can be quarantined", "WARNING")
            return

        sample_count = len(samples)
        if sample_count == 1:
            prompt = f"Move sample to quarantine?\n\n{samples[0].file_name}"
        else:
            prompt = f"Move {sample_count} samples to quarantine?"

        response = QMessageBox.question(
            self,
            "Quarantine Samples" if sample_count > 1 else "Quarantine Sample",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        moved_count = 0
        for sample in samples:
            moved = self.sample_service.quarantine_sample(sample.csv_path)
            moved_count += 1
            self.logger.info("Quarantined sample: %s", moved)

        if moved_count == 1:
            self._set_status("Sample moved to quarantine", "INFO")
        else:
            self._set_status(f"{moved_count} samples moved to quarantine", "INFO")
        self.refresh_samples()

    def restore_selected(self) -> None:
        samples = self._selected_samples()
        if not samples:
            return
        if not all(sample.source == "quarantine" for sample in samples):
            self._set_status("Select a quarantined sample to restore", "WARNING")
            return

        sample_count = len(samples)
        if sample_count == 1:
            prompt = f"Restore sample back to raw dataset?\n\n{samples[0].file_name}"
        else:
            prompt = f"Restore {sample_count} samples back to raw dataset?"

        response = QMessageBox.question(
            self,
            "Restore Samples" if sample_count > 1 else "Restore Sample",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        moved_count = 0
        for sample in samples:
            moved = self.sample_service.restore_sample(sample.csv_path)
            moved_count += 1
            self.logger.info("Restored sample: %s", moved)

        if moved_count == 1:
            self._set_status("Sample restored to raw dataset", "INFO")
        else:
            self._set_status(f"{moved_count} samples restored to raw dataset", "INFO")
        self.refresh_samples()


def run_data_manager(project_root: Path) -> None:
    """Open dataset manager window."""
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    window = DataManagerWindow(project_root=project_root)
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
            "CustomWidgets theme could not be loaded for data manager startup. "
            "Verify QT-PyQt-PySide-Custom-Widgets is installed and the style JSON is present."
        )
    window.show()

    if owns_app:
        app.exec()
