"""Tests for the stream gesture lock state machine."""

from gui.services.stream_service import GestureLockState


def test_lock_state_suppresses_duplicate_static_holds() -> None:
    state = GestureLockState()

    should_emit, is_rest, token = state.observe("hello", 0.92, 0.85)
    assert should_emit is True
    assert is_rest is False
    assert token == "HELLO"
    assert state.locked_token == "HELLO"

    should_emit, is_rest, token = state.observe("hello", 0.93, 0.85)
    assert should_emit is False
    assert is_rest is False
    assert token == "HELLO"
    assert state.locked_token == "HELLO"


def test_lock_state_unlocks_only_on_rest_or_new_class() -> None:
    state = GestureLockState()

    state.observe("hello", 0.92, 0.85)

    should_emit, is_rest, token = state.observe("hello", 0.60, 0.85)
    assert should_emit is False
    assert is_rest is False
    assert token == "HELLO"
    assert state.locked_token == "HELLO"

    should_emit, is_rest, token = state.observe("REST", 1.0, 0.85)
    assert should_emit is False
    assert is_rest is True
    assert token == "REST"
    assert state.locked_token is None

    should_emit, is_rest, token = state.observe("me", 0.91, 0.85)
    assert should_emit is True
    assert is_rest is False
    assert token == "ME"
    assert state.locked_token == "ME"
