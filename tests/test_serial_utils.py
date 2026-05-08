"""Unit tests for serial_utils module."""

from __future__ import annotations

import pytest

from utils.serial_utils import (
    FlexZeroWarningMonitor,
    build_flex_zero_warning,
    flex_zero_sensors,
    parse_sensor_data,
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
