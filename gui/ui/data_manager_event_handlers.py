"""Event-handler mixin for DataManagerWindow.

Provides the event-polling dispatch loop and all ``_on_*`` handlers
that process service-generated events (recording, processing, training,
and logging).
"""

from __future__ import annotations

import queue

from PySide6.QtWidgets import QMessageBox

from gui.ui.stage_widget import StageWidget


class DataManagerEventHandlersMixin:
    """Mixin providing event dispatch and handler methods.

    Relies on ``self`` being a fully-initialised DataManagerWindow
    instance at runtime (standard Python mixin pattern).  Handlers
    access the window's UI widgets, services, and convenience methods
    (``_set_status``, ``_set_task_state``, ``refresh_samples``, …)
    via ``self``.
    """

    # ---- Event dispatch tables ---------------------------------------------

    _EVENT_HANDLER_NAMES: dict[str, str] = {
        "record_started": "_on_record_started",
        "record_progress": "_on_record_progress",
        "record_error": "_on_record_error",
        "record_warning": "_on_record_warning",
        "record_ready_for_review": "_on_record_ready_for_review",
        "process_started": "_on_process_started",
        "process_total_gestures": "_on_process_total_gestures",
        "process_gesture_summary": "_on_process_gesture_summary",
        "process_stage_started": "_on_process_stage_started",
        "process_stage_metrics": "_on_process_stage_metrics",
        "process_stage_completed": "_on_process_stage_completed",
        "process_stage_failed": "_on_process_stage_failed",
        "process_stage_skipped": "_on_process_stage_skipped",
        "process_train_sequences": "_on_process_train_sequences",
        "process_test_sequences": "_on_process_test_sequences",
        "process_progress": "_on_process_progress",
        "process_completed": "_on_process_completed",
        "process_failed": "_on_process_failed",
        "process_cancelled": "_on_process_cancelled",
        "train_started": "_on_train_started",
        "train_epoch": "_on_train_epoch",
        "train_model_dir": "_on_train_model_dir",
        "train_completed": "_on_train_completed",
        "train_cancelled": "_on_train_cancelled",
        "train_failed": "_on_train_failed",
        "log": "_on_log",
    }

    _STAGE_ATTR_MAP: dict[str, str] = {
        "file_ingest": "stage_file_ingest",
        "smoothing": "stage_smoothing",
        "augmentation": "stage_augmentation",
        "feature_extraction": "stage_feature",
        "tensor_formatting": "stage_tensor",
        "save_train": "stage_save_train",
        "save_test": "stage_save_test",
    }

    # ---- Stage widget lookup -----------------------------------------------

    def _get_stage_widget(self, stage: str) -> StageWidget | None:
        """Resolve a stage widget by name, falling back to a name mapping."""
        widget = getattr(self, f"stage_{stage}", None)
        if widget is not None:
            return widget
        attr_name = self._STAGE_ATTR_MAP.get(stage)
        if attr_name:
            return getattr(self, attr_name, None)
        return None

    # ---- Event poll loop ---------------------------------------------------

    def _poll_events(self) -> None:
        """Poll the event queue and dispatch handlers."""
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event.get("type")
            handler_name = self._EVENT_HANDLER_NAMES.get(event_type)
            if handler_name is not None:
                getattr(self, handler_name)(event)
                continue

            # Unknown event types are silently ignored

        if self._live_preview_rows is not None:
            self._plot_recording_preview(self._live_preview_rows, force=False)

    # ---- Event handlers ----------------------------------------------------

    def _on_record_started(self, event: dict) -> None:
        port = str(event.get("port", "unknown"))
        self.recording_status_label.setText(
            f"Recording status: capturing on {port}"
        )

    def _on_record_progress(self, event: dict) -> None:
        row_count = int(event.get("row_count", 0))
        elapsed_seconds = float(event.get("elapsed_seconds", 0.0))
        self.record_row_count_label.setText(f"Rows captured: {row_count}")
        self.recording_status_label.setText(
            f"Recording status: capturing ({elapsed_seconds:.1f}s)"
        )
        rows = event.get("rows")
        if isinstance(rows, list) and rows:
            self._live_preview_rows = rows

    def _on_record_error(self, event: dict) -> None:
        message = str(event.get("message", "Recording failed"))
        self._live_preview_rows = None
        self._set_task_state(False)
        self._set_status(f"Recording failed: {message}", "ERROR")
        self.recording_status_label.setText("Recording status: error")

    def _on_record_warning(self, event: dict) -> None:
        message = str(event.get("message", "")).strip()
        if message:
            self._set_status(message, "WARNING")

    def _on_record_ready_for_review(self, event: dict) -> None:
        gesture = str(event.get("gesture", ""))
        orientation = str(event.get("orientation", "unspecified"))
        row_count = int(event.get("row_count", 0))
        elapsed_seconds = float(event.get("elapsed_seconds", 0.0))
        rows = event.get("rows")

        self._set_task_state(False)
        self.record_row_count_label.setText(f"Rows captured: {row_count}")

        if not isinstance(rows, list) or not rows:
            self._live_preview_rows = None
            self._set_status(
                "No valid sensor rows captured. Recording discarded.", "WARNING"
            )
            self.recording_status_label.setText("Recording status: discarded")
            return

        self._live_preview_rows = None
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

    def _on_process_started(self, event: dict) -> None:
        for stage in (
            self.stage_file_ingest,
            self.stage_smoothing,
            self.stage_augmentation,
            self.stage_feature,
            self.stage_tensor,
            self.stage_save_train,
            self.stage_save_test,
        ):
            stage.set_metrics(None)
            stage.set_skipped()
        self.stage_file_ingest.set_started("Preparing")
        self._set_task_state(True, "process_data")

    def _on_process_total_gestures(self, event: dict) -> None:
        total = int(event.get("total", 0))
        self._process_total_gestures = total
        self.stage_file_ingest.set_metrics({"gestures": total})

    def _on_process_gesture_summary(self, event: dict) -> None:
        gesture = str(event.get("gesture", ""))
        files = int(event.get("files", 0))
        samples = int(event.get("samples", 0))
        self._set_process_table_value(gesture, 1, str(files))
        self._set_process_table_value(gesture, 2, str(samples))

    def _on_process_stage_started(self, event: dict) -> None:
        stage = str(event.get("stage", ""))
        gesture = event.get("gesture")
        widget = self._get_stage_widget(stage)
        if widget is not None:
            label = f"Running{': ' + str(gesture) if gesture else ''}"
            widget.set_started(label)

    def _on_process_stage_metrics(self, event: dict) -> None:
        stage = str(event.get("stage", ""))
        widget = self._get_stage_widget(stage)
        if widget is not None:
            metrics = {
                k: v
                for k, v in event.items()
                if k not in {"type", "stage", "gesture"}
            }
            widget.set_metrics(metrics)

    def _on_process_stage_completed(self, event: dict) -> None:
        stage = str(event.get("stage", ""))
        widget = self._get_stage_widget(stage)
        if widget is not None:
            widget.set_completed()

    def _on_process_stage_failed(self, event: dict) -> None:
        stage = str(event.get("stage", ""))
        message = str(event.get("message", ""))
        widget = self._get_stage_widget(stage)
        if widget is not None:
            widget.set_failed(message)
        self._set_status(f"Stage {stage} failed: {message}", "ERROR")

    def _on_process_stage_skipped(self, event: dict) -> None:
        stage = str(event.get("stage", ""))
        widget = self._get_stage_widget(stage)
        if widget is not None:
            widget.set_skipped()

    def _on_process_train_sequences(self, event: dict) -> None:
        gesture = str(event.get("gesture", ""))
        count = int(event.get("count", 0))
        self._set_process_table_value(gesture, 3, str(count))
        self.stage_save_train.set_metrics({"train_seq": count})

    def _on_process_test_sequences(self, event: dict) -> None:
        gesture = str(event.get("gesture", ""))
        count = int(event.get("count", 0))
        self._set_process_table_value(gesture, 4, str(count))
        self.stage_save_test.set_metrics({"test_seq": count})

    def _on_process_progress(self, event: dict) -> None:
        done = int(event.get("done", 0))
        total = max(int(event.get("total", 1)), 1)
        self._set_status(f"Status: processed {done}/{total}", "INFO")

    def _on_process_completed(self, event: dict) -> None:
        processed = int(event.get("processed", 0))
        total = int(event.get("total", 0))
        for w in (
            self.stage_file_ingest,
            self.stage_smoothing,
            self.stage_augmentation,
            self.stage_feature,
            self.stage_tensor,
            self.stage_save_train,
            self.stage_save_test,
        ):
            if w.status_label.text() != "Completed":
                w.set_completed()
        self._set_status(
            f"Processing complete: {processed}/{total} gesture(s)", "INFO"
        )
        self._set_task_state(False)
        self.refresh_samples()
        self._refresh_record_count()

    def _on_process_failed(self, event: dict) -> None:
        message = str(event.get("message", "Processing failed"))
        self._set_status(f"Processing failed: {message}", "ERROR")
        self._set_task_state(False)

    def _on_process_cancelled(self, event: dict) -> None:
        self._set_status("Processing cancelled", "WARNING")
        self._set_task_state(False)

    def _on_train_started(self, event: dict) -> None:
        self.train_status_label.setText("Status: running")

    def _on_train_epoch(self, event: dict) -> None:
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

    def _on_train_model_dir(self, event: dict) -> None:
        self._train_model_dir = str(event.get("model_dir", ""))

    def _on_train_completed(self, event: dict) -> None:
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

    def _on_train_cancelled(self, event: dict) -> None:
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("Epoch progress: cancelled")
        self.train_status_label.setText("Status: cancelled")
        self._set_status("Training cancelled", "WARNING")
        self._set_task_state(False)

    def _on_train_failed(self, event: dict) -> None:
        message = str(event.get("message", "Training failed"))
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("Epoch progress: failed")
        self.train_status_label.setText("Status: failed")
        self._set_status(f"Training failed: {message}", "ERROR")
        self._set_task_state(False)

    def _on_log(self, event: dict) -> None:
        level = str(event.get("level", "INFO"))
        timestamp = str(event.get("timestamp", ""))
        source = str(event.get("source", "app"))
        message = str(event.get("message", ""))
        self.log_box.appendPlainText(f"[{timestamp}] [{level}] {source}: {message}")
