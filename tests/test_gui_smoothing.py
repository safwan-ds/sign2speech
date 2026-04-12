"""Unit tests for GUI smoothing utilities."""

from collections import deque

from gui.utils.smoothing import PredictionSmoother, SentenceAssembler, majority_vote


def test_majority_vote_returns_most_common_token() -> None:
    window = deque(["hello", "me", "hello", "hello"])
    assert majority_vote(window) == "hello"


def test_prediction_smoother_applies_majority_over_window() -> None:
    smoother = PredictionSmoother(window_size=3)
    assert smoother.update("hello") == "hello"
    assert smoother.update("me") == "hello"
    assert smoother.update("hello") == "hello"


def test_sentence_assembler_debounces_repeated_token() -> None:
    assembler = SentenceAssembler(debounce_window=2)
    assert assembler.try_append("hello") is True
    assert assembler.try_append("hello") is False
    assert assembler.try_append("me") is True
    assert assembler.text() == "hello me"
