"""Unit tests for GUI text formatting helpers."""

from gui.utils.formatting import display_upper


def test_display_upper_uses_turkish_casing_for_dotted_i() -> None:
    assert display_upper("bir", "tr") == "BİR"


def test_display_upper_uses_turkish_casing_for_dotless_i() -> None:
    assert display_upper("ışık", "tr-TR") == "IŞIK"


def test_display_upper_uses_default_upper_for_non_turkish() -> None:
    assert display_upper("bir", "en") == "BIR"
