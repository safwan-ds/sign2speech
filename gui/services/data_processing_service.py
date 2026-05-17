"""Background data-processing service for the dataset manager GUI.

Runs the full CSV → NPZ processing pipeline in a daemon thread and emits
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
"""

from __future__ import annotations

import logging
import threading
from queue import Queue

import numpy as np

from config.config import (
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_DATA_DIR,
    TEST_DATA_SPLIT_PERCENTAGE,
    USE_TEST_SPLIT,
)
from core.pipeline.data_processor import (
    clear_previous_sequence_files,
    load_all_logs,
    prepare_lstm_dataset,
    save_processed_data_lstm,
)


class DataProcessingService:
    """Run the CSV → NPZ processing pipeline in a background thread."""

    def __init__(
        self,
        logger: logging.Logger,
        event_queue: Queue[dict],
    ) -> None:
        self._logger = logger
        self._event_queue = event_queue
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

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
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="DataProcessingService",
            )
            self._thread.start()
            return True

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **payload: object) -> None:
        self._event_queue.put({"type": event_type, **payload})

    def _run(self) -> None:
        try:
            self._emit("process_started")
            self._logger.info("[process] Starting data processing pipeline")

            clear_previous_sequence_files()

            gestures_data = load_all_logs()
            if not gestures_data:
                self._emit("process_failed", message="No log data found in raw directory")
                return

            total = len(gestures_data)
            self._emit("process_total_gestures", total=total)

            # Emit summary row for every gesture before processing starts
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
                self._emit("process_gesture_current", gesture=gesture)
                self._logger.info("[process] Processing gesture: '%s'", gesture)

                n_files = len(dataframes)
                rng = np.random.RandomState(RANDOM_STATE)
                shuffled = rng.permutation(n_files)

                if not USE_TEST_SPLIT:
                    train_dfs = dataframes
                    test_dfs: list = []
                elif n_files >= 2:
                    split_idx = max(
                        1, int(n_files * (1 - TEST_DATA_SPLIT_PERCENTAGE))
                    )
                    if split_idx >= n_files:
                        split_idx = n_files - 1
                    train_dfs = [dataframes[i] for i in shuffled[:split_idx]]
                    test_dfs = [dataframes[i] for i in shuffled[split_idx:]]
                else:
                    train_dfs = dataframes
                    test_dfs = []

                # Training sequences
                X_train, y_train = prepare_lstm_dataset(train_dfs, gesture)
                if X_train is not None and y_train is not None:
                    save_processed_data_lstm(X_train, y_train, gesture, PROCESSED_DIR)
                    self._emit(
                        "process_train_sequences",
                        gesture=gesture,
                        count=int(len(X_train)),
                    )
                    self._logger.info(
                        "[process] Training: %d sequences for '%s'",
                        len(X_train),
                        gesture,
                    )
                else:
                    self._logger.warning(
                        "[process] Could not create training sequences for '%s'",
                        gesture,
                    )

                # Test sequences (optional)
                if test_dfs:
                    X_test, y_test = prepare_lstm_dataset(test_dfs, gesture)
                    if X_test is not None and y_test is not None:
                        save_processed_data_lstm(
                            X_test, y_test, gesture, TEST_DATA_DIR
                        )
                        self._emit(
                            "process_test_sequences",
                            gesture=gesture,
                            count=int(len(X_test)),
                        )
                        self._logger.info(
                            "[process] Test: %d sequences for '%s'",
                            len(X_test),
                            gesture,
                        )

                done += 1
                self._emit("process_progress", done=done, total=total)

            self._emit("process_completed", processed=done, total=total)
            self._logger.info(
                "[process] Processing complete: %d/%d gesture(s)", done, total
            )

        except Exception as exc:
            self._logger.exception("[process] Pipeline failed: %s", exc)
            self._emit("process_failed", message=str(exc))
        finally:
            with self._lock:
                self._thread = None
