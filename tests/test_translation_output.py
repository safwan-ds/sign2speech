"""Tests for bilingual gesture translation output."""

import json
from pathlib import Path

from core.inference.gesture_translations import GestureTransitionStateMachine
from core.inference.gesture_translations import load_gesture_translations
from core.inference.gesture_translations import translate_gesture
from core.inference.gesture_translations import translate_gestures


def test_load_gesture_translations_reads_pairs(tmp_path: Path) -> None:
    # Use the new JSON approach (preferred). Create a JSON file containing
    # a simple mapping of gesture name -> translation.
    gestures = tmp_path / "gestures.json"
    data = {"hello": "merhaba", "me": "ben"}
    gestures.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    mapping = load_gesture_translations(str(gestures))

    assert mapping["hello"] == "merhaba"
    assert mapping["me"] == "ben"
    # REST token should not be present in the JSON mapping
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


def test_gesture_transition_state_machine_requires_rest_between_active_tokens() -> None:
    state = GestureTransitionStateMachine()

    assert state.observe("hello", valid=True) == ("HELLO", True, False)
    assert state.observe("me", valid=True) == ("HELLO", False, False)
    assert state.observe("REST", valid=True) == ("REST", True, True)
    assert state.observe("me", valid=True) == ("ME", True, False)
