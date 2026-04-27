"""Serial I/O adapter used by the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import re
import threading

import serial
import serial.tools.list_ports

from utils.serial_utils import detect_glove_ports, parse_sensor_data


@dataclass(slots=True)
class SerialSettings:
    """Connection parameters for the glove device."""

    port: str
    baud_rate: int = 115200
    timeout: float = 0.2


class SerialService:
    """Thin service wrapping pyserial lifecycle and parsing.

    A dedicated background thread drains the serial port into a bounded queue
    so that consumers (the prediction worker) never block on physical I/O.
    """

    # Bounded so that a stalled consumer cannot grow memory unboundedly; the
    # buffer holds ~2-3 seconds of typical sensor frames at our rates.
    _QUEUE_MAXLEN = 256

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection: serial.Serial | None = None
        self._settings: SerialSettings | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._row_queue: queue.Queue[dict[str, float]] = queue.Queue(
            maxsize=self._QUEUE_MAXLEN
        )
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def list_ports() -> list[str]:
        """Return all available serial device names."""
        return [port.device for port in serial.tools.list_ports.comports()]

    @staticmethod
    def list_port_entries() -> list[tuple[str, str]]:
        """Return (label, device) entries for user-friendly port lists."""
        entries: list[tuple[str, str]] = []
        for port in serial.tools.list_ports.comports():
            description = (port.description or "").strip()
            # Hide trailing COM ids like "(COM9)" and keep a clean device name.
            description = re.sub(r"\s*\(COM\d+\)\s*$", "", description).strip()
            if description:
                label = f"{description} ({port.device})"
            else:
                label = port.device
            entries.append((label, port.device))
        return entries

    @staticmethod
    def list_matching_ports() -> list[str]:
        """Return serial ports that are emitting valid glove sensor packets."""
        return detect_glove_ports()

    def connect(self, settings: SerialSettings) -> bool:
        """Open serial connection if needed.

        Returns True when a new connection is opened or reopened, and False when
        the existing connection already matches the requested settings.
        """
        with self._lock:
            if self._connection and self._connection.is_open:
                if self._settings == settings:
                    return False
                self._close_locked()

            self._connection = serial.Serial(
                settings.port,
                settings.baud_rate,
                timeout=settings.timeout,
            )
            self._settings = settings
            self._start_reader_locked()
            return True

    def disconnect(self) -> None:
        """Close serial connection if open."""
        with self._lock:
            self._close_locked()

    def reset_input_buffer(self) -> None:
        """Discard buffered input when the connection is open."""
        with self._lock:
            connection = self._connection
        if connection and connection.is_open:
            connection.reset_input_buffer()
        # Drop any rows already queued from before the reset.
        self._drain_queue()

    def _close_locked(self) -> None:
        self._stop_reader_locked()
        if self._connection and self._connection.is_open:
            self._connection.close()
        self._connection = None
        self._settings = None

    def _start_reader_locked(self) -> None:
        # Caller must hold self._lock.
        self._reader_stop = threading.Event()
        self._drain_queue()
        thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread = thread
        thread.start()

    def _stop_reader_locked(self) -> None:
        # Caller must hold self._lock.
        self._reader_stop.set()
        thread = self._reader_thread
        self._reader_thread = None
        if thread is not None and thread.is_alive():
            # Release lock briefly so the reader can finish its current readline
            # without deadlocking against close().
            self._lock.release()
            try:
                thread.join(timeout=1.0)
            finally:
                self._lock.acquire()

    def _drain_queue(self) -> None:
        try:
            while True:
                self._row_queue.get_nowait()
        except queue.Empty:
            pass

    def _reader_loop(self) -> None:
        """Continuously read serial lines and push parsed rows onto the queue."""
        while not self._reader_stop.is_set():
            with self._lock:
                connection = self._connection

            if not connection or not connection.is_open:
                # Connection was closed mid-read; exit gracefully.
                return

            try:
                line = connection.readline().decode("utf-8", errors="ignore")
            except serial.SerialException:
                # Port was yanked / closed; exit and let connect() restart us.
                return
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.warning("Serial reader error: %s", exc)
                return

            if not line:
                # readline timed out; loop and recheck stop event.
                continue

            row = parse_sensor_data(line)
            if row is None:
                continue

            try:
                self._row_queue.put_nowait(row)
            except queue.Full:
                # Drop the oldest frame to keep latency bounded.
                try:
                    self._row_queue.get_nowait()
                    self._row_queue.put_nowait(row)
                except queue.Empty:
                    pass

    def read_sensor_row(
        self, timeout: float = 0.2
    ) -> dict[str, float] | None:
        """Block up to ``timeout`` seconds for the next parsed sensor row."""
        with self._lock:
            connection = self._connection
        if not connection or not connection.is_open:
            return None
        try:
            return self._row_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_connected(self) -> bool:
        """Return True when serial connection is open."""
        with self._lock:
            connection = self._connection
        return connection is not None and connection.is_open
