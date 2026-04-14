"""Tests for bilingual gesture translation output."""

from pathlib import Path

from core.inference.gesture_translations import (
    load_gesture_translations,
    translate_gesture,
    translate_gestures,
)


def test_load_gesture_translations_reads_pairs(tmp_path: Path) -> None:
    gestures = tmp_path / "gestures.txt"
    gestures.write_text("hello - merhaba\nme - ben\nREST\n", encoding="utf-8")

    mapping = load_gesture_translations(str(gestures))

    assert mapping["hello"] == "merhaba"
    assert mapping["me"] == "ben"
    assert "rest" not in mapping


def test_translate_gesture_uses_target_language() -> None:
    mapping = {"hello": "merhaba", "thank_you": "teşekkür ederim"}

    assert translate_gesture("hello", mapping, target_language="tr") == "merhaba"
    assert translate_gesture("thank_you", mapping, target_language="en") == "thank you"


def test_translate_gesture_falls_back_to_normalized_token() -> None:
    mapping = {"hello": "merhaba"}

    assert (
        translate_gesture("unknown_label", mapping, target_language="tr")
        == "unknown label"
    )
    assert (
        translate_gesture("unknown_label", mapping, target_language="en")
        == "unknown label"
    )


def test_translate_gestures_translates_list_in_selected_language() -> None:
    mapping = {"hello": "merhaba", "me": "ben"}

    assert translate_gestures(["hello", "me"], mapping, target_language="tr") == [
        "merhaba",
        "ben",
    ]
    assert translate_gestures(["hello", "me"], mapping, target_language="en") == [
        "hello",
        "me",
    ]
