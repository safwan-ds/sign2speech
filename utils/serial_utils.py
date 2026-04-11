"""
Serial communication utilities for Sign Language Glove project
"""

import os
import sys
import serial
import serial.tools.list_ports

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXPECTED_SENSOR_COUNT


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
                sensor_names = [
                    "flex0",
                    "flex1",
                    "flex2",
                    "flex3",
                    "flex4",
                    "accelX",
                    "accelY",
                    "accelZ",
                    "gyroX",
                    "gyroY",
                    "gyroZ",
                ]
                return {
                    name: float(val.strip()) for name, val in zip(sensor_names, values)
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

    # Try single port
    if len(ports) == 1:
        return ports[0].device

    # Prefer known device types
    preferred_keywords = ["bluetooth", "arduino", "usb serial", "ch340", "cp210"]
    for port in ports:
        haystack = f"{port.description} {port.manufacturer}".lower()
        if any(keyword in haystack for keyword in preferred_keywords):
            return port.device

    return ports[0].device


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
