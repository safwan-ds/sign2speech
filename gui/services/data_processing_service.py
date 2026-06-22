"""Background data-processing service for the dataset manager GUI.

Runs the full CSV -> NPZ processing pipeline in a daemon thread and emits
typed events to the shared ``event_queue`` so the GUI can update without
regex-parsing log lines.

Event types emitted
-------------------
``process_started``
    Pipeline has begun. No extra payload.
``process_total_gestures``
    ``total: int`` – number of gesture folders found.
``process_gesture_summary``
    ``gesture: str, files: int, samples: int`` – high-level summary per gesture.
``process_gesture_current``
    ``gesture: str`` – name of the gesture currently being processed.
``process_train_sequences``
    ``gesture: str, count: int`` – training sequences generated for this gesture.
``process_test_sequences``
    ``gesture: str, count: int`` – test sequences generated for this gesture.
``process_progress``
    ``done: int, total: int`` – gestures fully processed so far.
``process_completed``
    ``processed: int, total: int`` – final success counts.
``process_failed``
    ``message: str`` – human-readable error message on failure.
``process_cancelled``
    Processing was cancelled. No extra payload.
"""

from __future__ import annotations

import logging
import threading
from queue import Queue

import numpy as np

from config.architecture import architecture
from config.config import (
    PROCESSED_DIR,
    TEST_DATA_DIR,
)
from core.pipeline.data_processor import (
    clear_previous_sequence_files,
    load_all_logs,
    prepare_lstm_dataset,
    save_processed_data_lstm,
)


class DataProcessingService:
    """Run the CSV -> NPZ processing pipeline in a background thread."""

    def __init__(
        self,
        logger: logging.Logger,
        event_queue: Queue[dict],
    ) -> None:
        self._logger = logger
        self._event_queue = event_queue
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        # Stage control flags (set by UI via skip/retry)
        self._skip_current_stage = threading.Event()
        self._retry_current_stage = threading.Event()
        self._current_stage: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> bool:
        """Start the processing pipeline. Returns *False* when already running."""
        with self._lock:
            if self.is_running:
                return False
            self._cancel_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="DataProcessingService",
            )
            self._thread.start()
            return True

    def stop(self) -> bool:
        """Request cancellation. Returns *False* when not running."""
        if not self.is_running:
            return False
        self._cancel_event.set()
        self._logger.info("[process] Cancellation requested")
        return True

    def skip_current_stage(self) -> None:
        """Request that the currently running stage be skipped by the worker."""
        self._skip_current_stage.set()
        self._logger.info("[process] Skip requested for current stage")

    def retry_current_stage(self) -> None:
        """Request a one-time retry of the current stage after a failure."""
        self._retry_current_stage.set()
        self._logger.info("[process] Retry requested for current stage")

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **payload: object) -> None:
        self._event_queue.put({"type": event_type, **payload})

    @staticmethod
    def _coerce_row_count(data):
        if hasattr(data, "shape"):
            return int(data.shape[0])
        if hasattr(data, "__len__"):
            try:
                return int(len(data))
            except TypeError:
                return 0
        return 0

    def _run(self) -> None:
        """Run the processing pipeline emitting stage-level events."""
        try:
            self._emit("process_started")
            self._logger.info("[process] Starting data processing pipeline")

            # Stage: file ingestion
            self._current_stage = "file_ingest"
            self._emit("process_stage_started", stage="file_ingest")
            clear_previous_sequence_files()

            gestures_data = load_all_logs()
            if not gestures_data:
                self._emit("process_failed", message="No log data found in raw directory")
                self._emit("process_stage_failed", stage="file_ingest", message="No log data found")
                return

            total = len(gestures_data)
            self._emit("process_total_gestures", total=total)
            self._emit("process_stage_completed", stage="file_ingest", total=total)

            for gesture, dfs in gestures_data.items():
                sample_count = sum(len(df) for df in dfs)
                self._emit(
                    "process_gesture_summary",
                    gesture=gesture,
                    files=len(dfs),
                    samples=sample_count,
                )

            done = 0
            for gesture, dataframes in gestures_data.items():
                if self._cancel_event.is_set():
                    self._emit("process_cancelled")
                    self._logger.info("[process] Processing cancelled")
                    return

                self._emit("process_gesture_current", gesture=gesture)
                self._logger.info("[process] Processing gesture: '%s'", gesture)

                n_files = len(dataframes)
                rng = np.random.RandomState(architecture.training.random_state)
                shuffled = rng.permutation(n_files)

                if not architecture.training.use_test_split:
                    train_dfs = dataframes
                    test_dfs: list = []
                elif n_files >= 2:
                    split_idx = max(
                        1, int(n_files * (1 - architecture.training.test_data_split_percentage))
                    )
                    if split_idx >= n_files:
                        split_idx = n_files - 1
                    train_dfs = [dataframes[i] for i in shuffled[:split_idx]]
                    test_dfs = [dataframes[i] for i in shuffled[split_idx:]]
                else:
                    train_dfs = dataframes
                    test_dfs = []

                try:
                    self._current_stage = "smoothing"
                    self._emit("process_stage_started", stage="smoothing", gesture=gesture)
                    if self._skip_current_stage.is_set():
                        self._emit("process_stage_skipped", stage="smoothing", gesture=gesture)
                        self._skip_current_stage.clear()

                    self._current_stage = "augmentation"
                    aug_params = {
                        "use_enhanced_features": bool(architecture.model.use_enhanced_features),
                        "include_velocity": bool(architecture.model.include_velocity),
                        "include_acceleration": bool(architecture.model.include_acceleration),
                        "include_rolling_stats": bool(architecture.model.include_rolling_stats),
                        "rolling_window_size": int(architecture.model.rolling_window_size),
                    }
                    self._emit("process_stage_started", stage="augmentation", gesture=gesture, params=aug_params)
                    if self._skip_current_stage.is_set():
                        self._emit("process_stage_skipped", stage="augmentation", gesture=gesture)
                        self._skip_current_stage.clear()

                    self._current_stage = "feature_extraction"
                    self._emit("process_stage_started", stage="feature_extraction", gesture=gesture)

                    X_train, y_train = prepare_lstm_dataset(train_dfs, gesture)

                    if hasattr(X_train, "shape"):
                        tensor_shape = list(X_train.shape)
                    else:
                        try:
                            tensor_shape = [int(len(X_train)), X_train.shape[-1]]
                        except Exception as exc:
                            self._logger.debug("[process] Non-standard sequence container: %s", exc)
                            tensor_shape = []
                    self._emit(
                        "process_stage_metrics",
                        stage="feature_extraction",
                        gesture=gesture,
                        tensor_shape=tensor_shape,
                    )

                    self._current_stage = "save_train"
                    if X_train is not None and y_train is not None:
                        self._emit("process_stage_started", stage="save_train", gesture=gesture)
                        save_processed_data_lstm(X_train, y_train, gesture, PROCESSED_DIR)
                        train_count = self._coerce_row_count(X_train)
                        self._emit("process_train_sequences", gesture=gesture, count=train_count)
                        self._emit("process_stage_completed", stage="save_train", gesture=gesture, count=train_count)
                        self._logger.info("[process] Training: %d sequences for '%s'", train_count, gesture)
                    else:
                        self._logger.warning("[process] Could not create training sequences for '%s'", gesture)
                        self._emit("process_stage_failed", stage="feature_extraction", gesture=gesture, message="No training sequences")

                    if test_dfs:
                        self._current_stage = "save_test"
                        self._emit("process_stage_started", stage="save_test", gesture=gesture)
                        X_test, y_test = prepare_lstm_dataset(test_dfs, gesture)
                        if X_test is not None and y_test is not None:
                            save_processed_data_lstm(X_test, y_test, gesture, TEST_DATA_DIR)
                            test_count = self._coerce_row_count(X_test)
                            self._emit("process_test_sequences", gesture=gesture, count=test_count)
                            self._emit("process_stage_completed", stage="save_test", gesture=gesture, count=test_count)
                            self._logger.info("[process] Test: %d sequences for '%s'", test_count, gesture)
                        else:
                            self._logger.warning("[process] Could not create test sequences for '%s'", gesture)
                            self._emit("process_stage_failed", stage="save_test", gesture=gesture, message="No test sequences")

                except Exception as exc:
                    self._logger.exception("[process] Error processing gesture '%s': %s", gesture, exc)
                    self._emit("process_stage_failed", stage=self._current_stage or "unknown", gesture=gesture, message=str(exc))
                    if self._retry_current_stage.is_set():
                        self._logger.info("[process] Retry requested for stage %s", self._current_stage)
                        self._retry_current_stage.clear()
                        try:
                            X_train, y_train = prepare_lstm_dataset(train_dfs, gesture)
                            if X_train is not None and y_train is not None:
                                save_processed_data_lstm(X_train, y_train, gesture, PROCESSED_DIR)
                                self._emit("process_train_sequences", gesture=gesture, count=self._coerce_row_count(X_train))
                        except Exception:
                            self._logger.exception("[process] Retry failed for gesture %s", gesture)

                done += 1
                self._emit("process_progress", done=done, total=total)

            self._emit("process_completed", processed=done, total=total)
            self._logger.info("[process] Processing complete: %d/%d gesture(s)", done, total)

        except Exception as exc:
            self._logger.exception("[process] Pipeline failed: %s", exc)
            self._emit("process_failed", message=str(exc))
        finally:
            with self._lock:
                self._thread = None
                self._current_stage = None
