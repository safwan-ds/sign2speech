"""Unit tests for llm_utils module."""

from __future__ import annotations

import sys
import types

import pytest

import utils.llm_utils as llm_utils
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


def test_generate_reply_strips_output_prefix_when_present() -> None:
    class _FakeLLM:
        def __call__(self, prompt: str, **kwargs):
            return {"choices": [{"text": "Output:  Merhaba dünya"}]}

    result = generate_reply(_FakeLLM(), "merhaba", language="tr")
    assert result == "Merhaba dünya"


def test_load_qwen_model_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_utils, "USE_QWEN_LLM", False)
    assert llm_utils.load_qwen_model() is None


def test_load_qwen_model_returns_none_when_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils, "USE_QWEN_LLM", True)
    monkeypatch.setattr(llm_utils, "QWEN_MODEL_PATH", "/tmp/does-not-exist.gguf")
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: False)
    assert llm_utils.load_qwen_model() is None


def test_load_qwen_model_raises_when_gpu_forced_with_zero_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils, "USE_QWEN_LLM", True)
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: True)
    monkeypatch.setattr(llm_utils, "QWEN_FORCE_GPU", True)
    monkeypatch.setattr(llm_utils, "QWEN_N_GPU_LAYERS", 0)

    with pytest.raises(ValueError):
        llm_utils.load_qwen_model()


def test_load_qwen_model_returns_none_when_llama_cpp_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils, "USE_QWEN_LLM", True)
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: True)
    monkeypatch.setattr(llm_utils, "QWEN_FORCE_GPU", False)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))

    assert llm_utils.load_qwen_model() is None


def test_load_qwen_model_builds_llama_with_expected_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils, "USE_QWEN_LLM", True)
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: True)
    monkeypatch.setattr(llm_utils, "QWEN_FORCE_GPU", False)
    monkeypatch.setattr(llm_utils, "QWEN_MODEL_PATH", "/tmp/model.gguf")
    monkeypatch.setattr(llm_utils, "QWEN_N_CTX", 4096)
    monkeypatch.setattr(llm_utils, "QWEN_N_GPU_LAYERS", -1)
    monkeypatch.setattr(llm_utils, "QWEN_N_BATCH", 256)

    class _FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    model = llm_utils.load_qwen_model()

    assert isinstance(model, _FakeLlama)
    assert model.kwargs["model_path"] == "/tmp/model.gguf"
    assert model.kwargs["n_ctx"] == 4096
    assert model.kwargs["n_gpu_layers"] == -1
    assert model.kwargs["n_batch"] == 256
    assert model.kwargs["flash_attn"] is True
    assert model.kwargs["verbose"] is False
