"""Unit tests for GUI text formatting helpers."""

from gui.utils.formatting import display_upper


def test_display_upper_uses_turkish_casing_for_dotted_i() -> None:
    assert display_upper("bir", "tr") == "BİR"


def test_display_upper_uses_turkish_casing_for_dotless_i() -> None:
    assert display_upper("ışık", "tr-TR") == "IŞIK"


def test_display_upper_uses_default_upper_for_non_turkish() -> None:
    assert display_upper("bir", "en") == "BIR"


def test_now_hms_returns_time_format() -> None:
    import re

    from gui.utils.formatting import now_hms

    result = now_hms()
    assert re.match(r"^\d{2}:\d{2}:\d{2}$", result)


def test_now_stamp_returns_filename_safe_timestamp() -> None:
    import re

    from gui.utils.formatting import now_stamp

    result = now_stamp()
    assert re.match(r"^\d{8}_\d{6}$", result)


class TestPercent:
    def test_none_returns_zero_percent(self):
        from gui.utils.formatting import percent

        assert percent(None) == "0.0%"

    def test_zero_returns_zero_percent(self):
        from gui.utils.formatting import percent

        assert percent(0.0) == "0.0%"

    def test_one_returns_hundred_percent(self):
        from gui.utils.formatting import percent

        assert percent(1.0) == "100.0%"

    def test_half_returns_fifty_percent(self):
        from gui.utils.formatting import percent

        assert percent(0.5) == "50.0%"

    def test_formats_to_one_decimal_place(self):
        from gui.utils.formatting import percent

        result = percent(0.123)
        assert result == "12.3%"
