"""Background recording service used by the dataset manager GUI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import time
from queue import Queue

from config import BAUD_RATE, COM_PORT, LOGS_DIR, SERIAL_CONNECTION_DELAY, TIMEOUT
from utils.recording_utils import (
    build_recording_file_path,
    build_recording_metadata_path,
    save_recording_metadata,
    save_rows_to_csv,
)
from utils.serial_utils import connect_serial, parse_sensor_data, select_serial_port


@dataclass(slots=True)
class RecordingConfig:
    """Recording settings controlled by the GUI."""

    gesture_label: str
    orientation: str = "unspecified"


class RecordingService:
    """Capture one sample between start and stop, and stream progress to UI."""

    def __init__(self, logger: logging.Logger, event_queue: Queue[dict]) -> None:
        self._logger = logger
        self._event_queue = event_queue
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, config: RecordingConfig) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(config,),
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self, config: RecordingConfig) -> None:
        serial_conn = None
        selected_port: str | None = None
        rows: list[dict[str, float | int]] = []
        started_at = time.perf_counter()
        try:
            selected_port = select_serial_port(COM_PORT)
            if not selected_port:
                raise RuntimeError("No serial ports detected")

            self._logger.info(
                "[record] Connecting to %s at %s baud",
                selected_port,
                BAUD_RATE,
            )
            serial_conn = connect_serial(selected_port, BAUD_RATE, timeout=TIMEOUT)
            time.sleep(SERIAL_CONNECTION_DELAY)
            serial_conn.reset_input_buffer()

            self._event_queue.put(
                {
                    "type": "record_started",
                    "port": selected_port,
                    "gesture": config.gesture_label,
                }
            )

            last_emit = 0.0
            while not self._stop_event.is_set():
                try:
                    raw_line = (
                        serial_conn.readline().decode("utf-8", errors="ignore").strip()
                    )
                except Exception:
                    continue

                parsed = parse_sensor_data(raw_line)
                if parsed is None:
                    continue

                elapsed = time.perf_counter() - started_at
                elapsed_ms = int(elapsed * 1000)
                row: dict[str, float | int] = {"t_ms": elapsed_ms}
                row.update(parsed)
                rows.append(row)

                if elapsed - last_emit >= 0.1:
                    self._event_queue.put(
                        {
                            "type": "record_progress",
                            "row_count": len(rows),
                            "elapsed_seconds": elapsed,
                        }
                    )
                    last_emit = elapsed

            elapsed_seconds = max(0.0, time.perf_counter() - started_at)
            self._event_queue.put(
                {
                    "type": "record_ready_for_review",
                    "gesture": config.gesture_label,
                    "orientation": config.orientation,
                    "row_count": len(rows),
                    "elapsed_seconds": elapsed_seconds,
                    "rows": rows,
                }
            )
        except Exception as exc:
            self._logger.exception("[record] Recording failed: %s", exc)
            self._event_queue.put({"type": "record_error", "message": str(exc)})
        finally:
            if serial_conn is not None:
                try:
                    serial_conn.close()
                except Exception:
                    pass

    def save_recording(
        self,
        *,
        gesture_label: str,
        orientation: str,
        rows: list[dict[str, float | int]],
        elapsed_seconds: float,
    ) -> str:
        """Persist a reviewed recording and return saved CSV path."""
        target_path = build_recording_file_path(gesture_label, base_dir=LOGS_DIR)
        saved_path = save_rows_to_csv(target_path, rows)
        metadata_path = build_recording_metadata_path(saved_path)
        metadata = {
            "sample_id": saved_path.stem,
            "gesture": gesture_label,
            "orientation": orientation,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": elapsed_seconds,
            "row_count": len(rows),
            "csv_path": str(saved_path),
        }
        save_recording_metadata(metadata_path, metadata)
        self._logger.info("[record] Saved sample -> %s", saved_path)
        return str(saved_path)
