"""Tests for stream transition filtering helpers."""

from __future__ import annotations

import json
from collections import deque

import pytest

from gui.services.stream_service import (
    SequenceDecoder,
    TransitionHysteresis,
    _load_per_class_thresholds,
    calculate_motion_magnitude,
    validate_motion_consistency,
)


def test_transition_hysteresis_requires_consensus_for_switch() -> None:
    state = TransitionHysteresis(
        initial_consensus_frames=1,
        switch_consensus_frames=2,
        keep_last_stable_frames=1,
        uncertain_token="UNKNOWN",
    )
    assert state.resolve("HELLO", valid=True, is_rest=False) == "HELLO"
    assert state.resolve("ME", valid=True, is_rest=False) == "HELLO"
    assert state.resolve("ME", valid=True, is_rest=False) == "ME"


def test_transition_hysteresis_emits_unknown_after_invalid_streak() -> None:
    state = TransitionHysteresis(
        initial_consensus_frames=1,
        switch_consensus_frames=2,
        keep_last_stable_frames=1,
        uncertain_token="UNKNOWN",
    )
    assert state.resolve("HELLO", valid=True, is_rest=False) == "HELLO"
    assert state.resolve("HELLO", valid=False, is_rest=False) == "HELLO"
    assert state.resolve("HELLO", valid=False, is_rest=False) == "UNKNOWN"


def test_sequence_decoder_prefers_stay_with_transition_penalty() -> None:
    decoder = SequenceDecoder(
        ["REST", "HELLO", "ME"],
        enabled=True,
        switch_penalty=2.0,
        rest_switch_penalty=1.0,
    )
    first = decoder.decode({"REST": 0.1, "HELLO": 0.8, "ME": 0.1}, fallback="REST")
    second = decoder.decode({"REST": 0.1, "HELLO": 0.45, "ME": 0.46}, fallback="REST")
    assert first == "HELLO"
    assert second == "HELLO"


def test_calculate_motion_magnitude_uses_accel_and_gyro() -> None:
    magnitude = calculate_motion_magnitude(
        {"accelX": 3, "accelY": 4, "accelZ": 0, "gyroX": 0, "gyroY": 0, "gyroZ": 5}
    )
    assert magnitude == pytest.approx(10.0)


def test_load_per_class_thresholds_reads_uppercase_keys(tmp_path) -> None:
    model_dir = tmp_path / "model"
    eval_dir = model_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    payload = {
        "hello": {"confidence": 0.82, "gap": 0.2},
        "REST": {"confidence": 0.9, "gap": 0.3},
    }
    (eval_dir / "per_class_thresholds.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    loaded = _load_per_class_thresholds(model_dir)
    assert loaded["HELLO"]["confidence"] == 0.82
    assert loaded["REST"]["gap"] == 0.3


def test_validate_motion_consistency_rejects_low_motion() -> None:
    low_motion = deque([10.0] * 30, maxlen=30)
    assert validate_motion_consistency(low_motion) is False
