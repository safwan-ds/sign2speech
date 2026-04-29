"""Unit tests for llm_utils module."""

from __future__ import annotations

from utils.llm_utils import generate_reply, generate_turkish_reply


def test_generate_reply_returns_none_when_llm_is_none() -> None:
    result = generate_reply(None, "hello me")
    assert result is None


def test_generate_reply_returns_none_when_llm_is_none_english() -> None:
    result = generate_reply(None, "hello me", language="en")
    assert result is None


def test_generate_turkish_reply_returns_none_when_llm_is_none() -> None:
    result = generate_turkish_reply(None, "merhaba ben")
    assert result is None


def test_generate_reply_uses_mock_llm_english() -> None:
    """Verify that generate_reply forwards the token text to the llm callable."""

    class _FakeLLM:
        def __init__(self) -> None:
            self.last_prompt: str | None = None

        def __call__(self, prompt: str, **kwargs):
            self.last_prompt = prompt
            return {"choices": [{"text": "Hello, I am here."}]}

    fake = _FakeLLM()
    result = generate_reply(fake, "hello me", language="en")
    assert result == "Hello, I am here."
    assert fake.last_prompt is not None
    assert "hello me" in fake.last_prompt


def test_generate_reply_uses_mock_llm_turkish() -> None:
    class _FakeLLM:
        def __call__(self, prompt: str, **kwargs):
            return {"choices": [{"text": "Merhaba ben buradayım."}]}

    result = generate_reply(_FakeLLM(), "merhaba ben", language="tr")
    assert result == "Merhaba ben buradayım."


def test_generate_reply_returns_none_for_empty_text() -> None:
    class _FakeLLM:
        def __call__(self, prompt: str, **kwargs):
            return {"choices": [{"text": "   "}]}

    result = generate_reply(_FakeLLM(), "words", language="en")
    assert result is None


def test_generate_reply_handles_unexpected_llm_response() -> None:
    class _BadLLM:
        def __call__(self, prompt: str, **kwargs):
            return {}  # no 'choices' key

    result = generate_reply(_BadLLM(), "words", language="en")
    assert result is None


def test_generate_reply_handles_llm_exception() -> None:
    class _ExplodingLLM:
        def __call__(self, prompt: str, **kwargs):
            raise RuntimeError("Inference failure")

    result = generate_reply(_ExplodingLLM(), "words", language="en")
    assert result is None
