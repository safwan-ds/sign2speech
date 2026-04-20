"""Serial I/O adapter used by the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
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
    """Thin service wrapping pyserial lifecycle and parsing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection: serial.Serial | None = None
        self._settings: SerialSettings | None = None

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

    def _close_locked(self) -> None:
        if self._connection and self._connection.is_open:
            self._connection.close()
        self._connection = None
        self._settings = None

    def read_sensor_row(self) -> dict[str, float] | None:
        """Read and parse one line of sensor data."""
        with self._lock:
            connection = self._connection

        if not connection or not connection.is_open:
            return None

        line = connection.readline().decode("utf-8", errors="ignore")
        return parse_sensor_data(line)

    @property
    def is_connected(self) -> bool:
        """Return True when serial connection is open."""
        with self._lock:
            connection = self._connection
        return connection is not None and connection.is_open
