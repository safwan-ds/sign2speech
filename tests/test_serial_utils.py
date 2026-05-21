"""Unit tests for serial_utils module."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import utils.serial_utils as serial_utils
from utils.serial_utils import (
    FlexZeroWarningMonitor,
    build_flex_zero_warning,
    connect_serial,
    detect_glove_ports,
    flex_zero_sensors,
    parse_sensor_data,
    select_serial_port,
)


class TestParseSensorData:
    def _valid_line(
        self,
        flex0=100,
        flex1=110,
        flex2=120,
        flex3=130,
        flex4=140,
        accelX=1.0,
        accelY=2.0,
        accelZ=9.8,
        gyroX=0.1,
        gyroY=0.2,
        gyroZ=0.3,
    ) -> str:
        values = [flex0, flex1, flex2, flex3, flex4, accelX, accelY, accelZ, gyroX, gyroY, gyroZ]
        return ">" + ",".join(str(v) for v in values)

    def test_valid_line_returns_dict(self):
        result = parse_sensor_data(self._valid_line())
        assert isinstance(result, dict)

    def test_returns_all_sensor_keys(self):
        result = parse_sensor_data(self._valid_line())
        expected_keys = {
            "flex0", "flex1", "flex2", "flex3", "flex4",
            "accelX", "accelY", "accelZ",
            "gyroX", "gyroY", "gyroZ",
        }
        assert expected_keys == set(result.keys())

    def test_values_are_floats(self):
        result = parse_sensor_data(self._valid_line())
        for v in result.values():
            assert isinstance(v, float)

    def test_correct_values_parsed(self):
        line = self._valid_line(flex0=55, accelX=-1.5, gyroZ=3.14)
        result = parse_sensor_data(line)
        assert result["flex0"] == pytest.approx(55.0)
        assert result["accelX"] == pytest.approx(-1.5)
        assert result["gyroZ"] == pytest.approx(3.14)

    def test_line_without_prefix_returns_none(self):
        line = "100,110,120,130,140,1.0,2.0,9.8,0.1,0.2,0.3"
        assert parse_sensor_data(line) is None

    def test_empty_string_returns_none(self):
        assert parse_sensor_data("") is None

    def test_too_few_values_returns_none(self):
        assert parse_sensor_data(">100,110,120") is None

    def test_too_many_values_returns_none(self):
        line = ">100,110,120,130,140,1.0,2.0,9.8,0.1,0.2,0.3,99.9"
        assert parse_sensor_data(line) is None

    def test_non_numeric_value_returns_none(self):
        line = ">abc,110,120,130,140,1.0,2.0,9.8,0.1,0.2,0.3"
        assert parse_sensor_data(line) is None

    def test_line_with_trailing_newline_is_parsed(self):
        line = self._valid_line() + "\n"
        result = parse_sensor_data(line)
        assert result is not None

    def test_line_with_spaces_around_values_is_parsed(self):
        values = [100, 110, 120, 130, 140, 1.0, 2.0, 9.8, 0.1, 0.2, 0.3]
        line = ">" + ",".join(f" {v} " for v in values)
        result = parse_sensor_data(line)
        assert result is not None
        assert result["flex0"] == pytest.approx(100.0)


def test_flex_zero_sensors_returns_only_zero_flex_values() -> None:
    row = {
        "flex0": 0.0,
        "flex1": 110.0,
        "flex2": 0,
        "flex3": 130.0,
        "flex4": 140.0,
        "accelX": 0.0,
    }

    assert flex_zero_sensors(row) == ("flex0", "flex2")


def test_build_flex_zero_warning_mentions_sensor_and_connection() -> None:
    warning = build_flex_zero_warning(("flex1",))

    assert "flex1 is reading 0" in warning
    assert "connection or calibration" in warning


def test_build_flex_zero_warning_plural_message_lists_all_sensors() -> None:
    warning = build_flex_zero_warning(("flex0", "flex3"))
    assert "flex sensors flex0, flex3 are reading 0" in warning


def test_flex_zero_warning_monitor_rejects_invalid_min_samples() -> None:
    with pytest.raises(ValueError):
        FlexZeroWarningMonitor(min_consecutive_samples=0)


def test_flex_zero_warning_monitor_ignores_the_first_zero_sample() -> None:
    messages: list[str] = []
    monitor = FlexZeroWarningMonitor(
        min_consecutive_samples=2,
        min_interval_seconds=60.0,
        emit=messages.append,
    )
    row = {"flex0": 0.0}

    assert monitor.check(row) == ("flex0",)

    assert messages == []


def test_flex_zero_warning_monitor_warns_after_two_consecutive_zero_samples() -> None:
    messages: list[str] = []
    monitor = FlexZeroWarningMonitor(
        min_consecutive_samples=2,
        min_interval_seconds=60.0,
        emit=messages.append,
    )

    monitor.check({"flex0": 0.0})
    monitor.check({"flex0": 0.0, "flex2": 0.0})

    assert len(messages) == 1
    assert "flex0, flex2" in messages[0]


def test_flex_zero_warning_monitor_does_not_mix_sensor_streaks() -> None:
    messages: list[str] = []
    monitor = FlexZeroWarningMonitor(
        min_consecutive_samples=2,
        min_interval_seconds=60.0,
        emit=messages.append,
    )

    monitor.check({"flex0": 0.0})
    monitor.check({"flex2": 0.0})

    assert messages == []


def test_flex_zero_warning_monitor_resets_after_recovery() -> None:
    messages: list[str] = []
    monitor = FlexZeroWarningMonitor(
        min_consecutive_samples=2,
        min_interval_seconds=60.0,
        emit=messages.append,
    )

    monitor.check({"flex0": 0.0})
    monitor.check({"flex0": 100.0})
    monitor.check({"flex0": 0.0})
    monitor.check({"flex0": 0.0})

    assert len(messages) == 1


def test_select_serial_port_returns_none_when_no_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serial_utils.serial.tools.list_ports, "comports", lambda: [])
    assert select_serial_port() is None


def test_select_serial_port_prefers_explicit_available_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [
        SimpleNamespace(device="/dev/ttyS1", description="Other", manufacturer="Unknown"),
        SimpleNamespace(device="/dev/ttyUSB0", description="USB-serial", manufacturer="Vendor"),
    ]
    monkeypatch.setattr(serial_utils.serial.tools.list_ports, "comports", lambda: ports)
    assert select_serial_port("/dev/ttyS1") == "/dev/ttyS1"


def test_select_serial_port_prefers_known_device_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [
        SimpleNamespace(device="/dev/ttyS1", description="Debug adapter", manufacturer="Unknown"),
        SimpleNamespace(device="/dev/ttyUSB0", description="USB Serial CH340", manufacturer="QinHeng"),
    ]
    monkeypatch.setattr(serial_utils.serial.tools.list_ports, "comports", lambda: ports)
    assert select_serial_port() == "/dev/ttyUSB0"


def test_select_serial_port_returns_none_when_no_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [
        SimpleNamespace(device="/dev/ttyS1", description="Debug adapter", manufacturer="Unknown"),
        SimpleNamespace(device="/dev/ttyS2", description="Bluetooth", manufacturer="Unknown"),
    ]
    monkeypatch.setattr(serial_utils.serial.tools.list_ports, "comports", lambda: ports)
    assert select_serial_port() is None


def test_detect_glove_ports_returns_only_ports_with_valid_packets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [SimpleNamespace(device="COM1"), SimpleNamespace(device="COM2")]
    monkeypatch.setattr(serial_utils.serial.tools.list_ports, "comports", lambda: ports)
    monkeypatch.setattr(serial_utils.time, "sleep", lambda _: None)

    class _FakeSerial:
        def __init__(self, device: str, baud_rate: int, timeout: float) -> None:
            self.device = device
            self.closed = False
            self._lines = (
                [b"noise\n", b">1,2,3,4,5,6,7,8,9,10,11\n"]
                if device == "COM1"
                else [b"bad\n", b"still-bad\n"]
            )

        def readline(self) -> bytes:
            return self._lines.pop(0) if self._lines else b""

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(serial_utils.serial, "Serial", _FakeSerial)

    detected = detect_glove_ports(read_attempts=2)
    assert detected == ["COM1"]


def test_detect_glove_ports_skips_ports_with_serial_open_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [SimpleNamespace(device="COM1"), SimpleNamespace(device="COM2")]
    monkeypatch.setattr(serial_utils.serial.tools.list_ports, "comports", lambda: ports)
    monkeypatch.setattr(serial_utils.time, "sleep", lambda _: None)

    class _FakeSerial:
        def __init__(self, device: str, baud_rate: int, timeout: float) -> None:
            if device == "COM1":
                raise serial_utils.serial.SerialException("port busy")
            self._lines = [b">1,2,3,4,5,6,7,8,9,10,11\n"]

        def readline(self) -> bytes:
            return self._lines.pop(0) if self._lines else b""

        def close(self) -> None:
            return None

    monkeypatch.setattr(serial_utils.serial, "Serial", _FakeSerial)

    detected = detect_glove_ports(read_attempts=1)
    assert detected == ["COM2"]


def test_detect_glove_ports_ignores_close_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports = [SimpleNamespace(device="COM1")]
    monkeypatch.setattr(serial_utils.serial.tools.list_ports, "comports", lambda: ports)
    monkeypatch.setattr(serial_utils.time, "sleep", lambda _: None)

    class _FakeSerial:
        def __init__(self, device: str, baud_rate: int, timeout: float) -> None:
            self._lines = [b"not-valid\n"]

        def readline(self) -> bytes:
            return self._lines.pop(0) if self._lines else b""

        def close(self) -> None:
            raise RuntimeError("close failed")

    monkeypatch.setattr(serial_utils.serial, "Serial", _FakeSerial)

    detected = detect_glove_ports(read_attempts=1)
    assert detected == []


def test_connect_serial_returns_serial_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeSerial:
        def __init__(self, port: str, baud_rate: int, timeout: float) -> None:
            captured["port"] = port
            captured["baud_rate"] = baud_rate
            captured["timeout"] = timeout

    monkeypatch.setattr(serial_utils.serial, "Serial", _FakeSerial)

    connection = connect_serial("COM9", 115200, timeout=0.75)

    assert isinstance(connection, _FakeSerial)
    assert captured == {"port": "COM9", "baud_rate": 115200, "timeout": 0.75}
