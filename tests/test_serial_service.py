from __future__ import annotations

from dataclasses import dataclass

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
    settings = SerialSettings(port="COM3", baud_rate=115200, timeout=0.2)

    assert service.connect(settings) is True
    assert service.connect(settings) is False
    assert len(created) == 1
    assert service.is_connected is True

    service.disconnect()

    assert created[0].close_calls == 1
    assert service.is_connected is False
