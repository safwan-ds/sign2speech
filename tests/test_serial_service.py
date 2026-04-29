from __future__ import annotations

from dataclasses import dataclass

from config import BAUD_RATE
from gui.services.serial_service import SerialService, SerialSettings


@dataclass
class _FakeSerial:
    port: str
    baud_rate: int
    timeout: float

    def __post_init__(self) -> None:
        self.is_open = True
        self.close_calls = 0
        self.reset_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.is_open = False

    def reset_input_buffer(self) -> None:
        self.reset_calls += 1

    def readline(self) -> bytes:
        return b""


def test_connect_reuses_existing_open_connection(monkeypatch) -> None:
    created: list[_FakeSerial] = []

    def fake_serial(port: str, baud_rate: int, timeout: float) -> _FakeSerial:
        serial_conn = _FakeSerial(port, baud_rate, timeout)
        created.append(serial_conn)
        return serial_conn

    monkeypatch.setattr("gui.services.serial_service.serial.Serial", fake_serial)

    service = SerialService()
    settings = SerialSettings(port="COM3", baud_rate=BAUD_RATE, timeout=0.2)

    assert service.connect(settings) is True
    assert service.connect(settings) is False
    assert len(created) == 1
    assert service.is_connected is True

    service.disconnect()

    assert created[0].close_calls == 1
    assert service.is_connected is False


def test_is_connected_false_before_connect() -> None:
    service = SerialService()
    assert service.is_connected is False


def test_reconnect_with_different_settings_opens_new_connection(monkeypatch) -> None:
    created: list[_FakeSerial] = []

    def fake_serial(port: str, baud_rate: int, timeout: float) -> _FakeSerial:
        s = _FakeSerial(port, baud_rate, timeout)
        created.append(s)
        return s

    monkeypatch.setattr("gui.services.serial_service.serial.Serial", fake_serial)

    service = SerialService()
    settings_a = SerialSettings(port="COM3", baud_rate=BAUD_RATE, timeout=0.2)
    settings_b = SerialSettings(port="COM4", baud_rate=BAUD_RATE, timeout=0.2)

    service.connect(settings_a)
    service.connect(settings_b)

    assert len(created) == 2
    assert created[0].close_calls == 1  # first connection was closed
    assert service.is_connected is True


def test_disconnect_when_not_connected_is_safe() -> None:
    service = SerialService()
    service.disconnect()  # must not raise
    assert service.is_connected is False


def test_reset_input_buffer_when_not_connected_is_safe(monkeypatch) -> None:
    service = SerialService()
    # Must not raise even with no active connection
    service.reset_input_buffer()


def test_read_sensor_row_returns_none_when_not_connected() -> None:
    service = SerialService()
    result = service.read_sensor_row(timeout=0.05)
    assert result is None
