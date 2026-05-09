"""Serial communication utilities for Sign Language Glove project."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping

import serial
import serial.tools.list_ports

from config import BAUD_RATE, EXPECTED_SENSOR_COUNT

FLEX_SENSOR_NAMES = ("flex0", "flex1", "flex2", "flex3", "flex4")
SENSOR_NAMES = (
    *FLEX_SENSOR_NAMES,
    "accelX",
    "accelY",
    "accelZ",
    "gyroX",
    "gyroY",
    "gyroZ",
)

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows fallback
    winsound = None


class ContinuousWarningBeeper:
    """Play a continuous warning beep while active."""

    def __init__(self, *, frequency_hz: int = 1200, beep_ms: int = 250) -> None:
        self._frequency_hz = frequency_hz
        self._beep_ms = beep_ms
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        thread: threading.Thread | None = None
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=(self._beep_ms / 1000.0) + 0.3)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if winsound is not None:
                winsound.Beep(self._frequency_hz, self._beep_ms)
            else:
                self._stop_event.wait(self._beep_ms / 1000.0)


def flex_zero_sensors(sensor_row: Mapping[str, float]) -> tuple[str, ...]:
    """Return flex sensor names whose current reading is exactly zero."""
    return tuple(
        name for name in FLEX_SENSOR_NAMES if float(sensor_row.get(name, -1.0)) == 0.0
    )


def build_flex_zero_warning(zero_sensors: tuple[str, ...]) -> str:
    """Build a concise operator warning for zero-valued flex sensors."""
    if len(zero_sensors) == 1:
        return (
            f"Warning: {zero_sensors[0]} is reading 0. "
            "Check the flex sensor connection or calibration."
        )
    sensors = ", ".join(zero_sensors)
    return (
        f"Warning: flex sensors {sensors} are reading 0. "
        "Check the flex sensor connections or calibration."
    )


class FlexZeroWarningMonitor:
    """Warn when flex sensors stay at zero without flooding high-rate streams."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        min_consecutive_samples: int = 2,
        min_interval_seconds: float = 5.0,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        if min_consecutive_samples < 1:
            raise ValueError("min_consecutive_samples must be at least 1")
        self._logger = logger
        self._min_consecutive_samples = min_consecutive_samples
        self._min_interval_seconds = min_interval_seconds
        self._emit = emit
        self._last_warning_at: float | None = None
        self._zero_streak_counts = {name: 0 for name in FLEX_SENSOR_NAMES}

    def check(self, sensor_row: Mapping[str, float]) -> tuple[str, ...]:
        """Inspect one row and emit a throttled warning after repeated zero readings."""
        zero_sensors = flex_zero_sensors(sensor_row)
        if not zero_sensors:
            for name in FLEX_SENSOR_NAMES:
                self._zero_streak_counts[name] = 0
            return ()

        zero_sensor_set = set(zero_sensors)
        for name in FLEX_SENSOR_NAMES:
            if name in zero_sensor_set:
                self._zero_streak_counts[name] += 1
            else:
                self._zero_streak_counts[name] = 0

        persistent_zero_sensors = tuple(
            name
            for name in zero_sensors
            if self._zero_streak_counts[name] >= self._min_consecutive_samples
        )
        now = time.monotonic()
        should_warn = (
            bool(persistent_zero_sensors)
            and (
                self._last_warning_at is None
                or now - self._last_warning_at >= self._min_interval_seconds
            )
        )
        if should_warn:
            message = build_flex_zero_warning(zero_sensors)
            if self._logger is not None:
                self._logger.warning(message)
            if self._emit is not None:
                self._emit(message)
            self._last_warning_at = now

        return zero_sensors


def parse_sensor_data(line: str) -> dict[str, float] | None:
    """
    Parses sensor data in CSV format: >flex0,flex1,flex2,flex3,flex4,accelX,accelY,accelZ,gyroX,gyroY,gyroZ

    Args:
        line: Serial line string to parse

    Returns:
        dict: Dictionary with sensor names as keys, or None if parsing fails
    """
    line = line.strip()
    if line.startswith(">"):
        try:
            values = line[1:].split(",")
            if len(values) == EXPECTED_SENSOR_COUNT:
                return {
                    name: float(val.strip()) for name, val in zip(SENSOR_NAMES, values)
                }
        except (ValueError, IndexError):
            pass
    return None


def select_serial_port(preferred_port: str | None = None):
    """
    Auto-detect serial port, preferring known Arduino/Bluetooth devices

    Args:
        preferred_port: Preferred COM port to use if available

    Returns:
        str: Selected COM port or None if no ports found
    """
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        return None

    # Check if preferred port exists
    if preferred_port:
        available_ports = {p.device: p for p in ports}
        if preferred_port in available_ports:
            return preferred_port

    # Prefer known device types
    preferred_keywords = ["arduino", "usb serial", "ch340", "cp210", "usb-serial"]

    # Iterate through all ports and find the first one that matches our keywords
    for port in ports:
        haystack = f"{port.description} {port.manufacturer}".lower()
        if any(keyword in haystack for keyword in preferred_keywords):
            return port.device

    return None


def detect_glove_ports(
    baud_rate: int = BAUD_RATE,
    timeout: float = 0.12,
    read_attempts: int = 3,
    settle_delay: float = 0.15,
) -> list[str]:
    """Return ports that emit valid glove sensor packets.

    The probe keeps the scan conservative: only ports that produce at least
    one parseable sensor row are returned, so the GUI can hide unrelated COM
    ports.
    """

    matching_ports: list[str] = []
    for port_info in serial.tools.list_ports.comports():
        connection: serial.Serial | None = None
        try:
            connection = serial.Serial(port_info.device, baud_rate, timeout=timeout)
            time.sleep(settle_delay)
            for _ in range(read_attempts):
                line = connection.readline().decode("utf-8", errors="ignore")
                if parse_sensor_data(line) is not None:
                    matching_ports.append(port_info.device)
                    break
        except (serial.SerialException, OSError, ValueError):
            continue
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    return matching_ports


def connect_serial(port: str, baud_rate: int, timeout: float = 1.0):
    """
    Connect to serial port

    Args:
        port: COM port to connect to
        baud_rate: Baud rate for serial communication
        timeout: Timeout in seconds

    Returns:
        serial.Serial: Serial connection object

    Raises:
        Exception: If connection fails
    """
    ser = serial.Serial(port, baud_rate, timeout=timeout)
    return ser
