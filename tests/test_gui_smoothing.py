"""Unit tests for GUI smoothing utilities."""

from collections import deque

from gui.utils.smoothing import PredictionSmoother, SentenceAssembler, majority_vote


def test_majority_vote_returns_most_common_token() -> None:
    window = deque(["hello", "me", "hello", "hello"])
    assert majority_vote(window) == "hello"


def test_prediction_smoother_applies_weighted_recency_over_window() -> None:
    smoother = PredictionSmoother(window_size=3)
    assert smoother.update("hello") == "hello"
    assert smoother.update("me") == "me"
    assert smoother.update("hello") == "hello"


def test_sentence_assembler_debounces_repeated_token() -> None:
    assembler = SentenceAssembler(debounce_window=2)
    assert assembler.try_append("hello") is True
    assert assembler.try_append("hello") is False
    assert assembler.try_append("me") is True
    assert assembler.text() == "hello me"


def test_majority_vote_returns_none_for_empty_window() -> None:
    assert majority_vote(deque()) is None


def test_prediction_smoother_reset_clears_window() -> None:
    smoother = PredictionSmoother(window_size=3)
    smoother.update("hello")
    smoother.update("hello")
    smoother.reset()
    # After reset the window is empty; next token should return itself
    result = smoother.update("me")
    assert result == "me"


def test_prediction_smoother_uses_confidence_weighting() -> None:
    smoother = PredictionSmoother(window_size=3)
    assert smoother.update("hello", confidence=0.95) == "hello"
    assert smoother.update("me", confidence=0.05) == "hello"


def test_prediction_smoother_downweights_rest_votes() -> None:
    smoother = PredictionSmoother(window_size=3, rest_weight=0.1)
    smoother.update("REST", confidence=1.0, is_rest=True)
    result = smoother.update("hello", confidence=0.9, is_rest=False)
    assert result == "hello"


def test_sentence_assembler_clear_resets_state() -> None:
    assembler = SentenceAssembler(debounce_window=2)
    assembler.try_append("hello")
    assembler.try_append("me")
    assembler.clear()
    assert assembler.text() == ""
    # After clear, same token should be accepted again
    assert assembler.try_append("hello") is True


def test_sentence_assembler_empty_token_not_appended() -> None:
    assembler = SentenceAssembler()
    assert assembler.try_append("") is False
    assert assembler.text() == ""


def test_sentence_assembler_text_joins_with_space() -> None:
    assembler = SentenceAssembler(debounce_window=1)
    assembler.try_append("a")
    assembler.try_append("b")
    assembler.try_append("c")
    assert assembler.text() == "a b c"
