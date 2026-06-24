"""Unit tests for llm_utils module."""

from __future__ import annotations

import sys
import types

import pytest

import utils.llm_utils as llm_utils
from utils.llm_utils import generate_reply
from utils.llm_utils import generate_turkish_reply


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


# -- Remote callable integration tests (mocked OpenAI SDK) --------------------------

from unittest.mock import MagicMock

import openai


def _make_remote_callable(
    monkeypatch: pytest.MonkeyPatch,
    *,
    url: str = "https://api.example.com/v1",
    model: str = "test-model",
    api_key: str = "sk-test",
) -> tuple[object, MagicMock]:
    monkeypatch.setattr(llm_utils.architecture.llm, "llm_remote_url", url)
    monkeypatch.setattr(llm_utils.architecture.llm, "llm_remote_api_key", api_key)
    monkeypatch.setattr(llm_utils.architecture.llm, "llm_remote_model", model)

    # Mock the OpenAI client
    mock_client_instance = MagicMock()
    mock_openai_class = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr("openai.OpenAI", mock_openai_class)

    callable_fn, meta = llm_utils._make_remote_llm()
    return callable_fn, mock_client_instance


class TestRemoteCallable:
    def test_successful_chat_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        remote, mock_client = _make_remote_callable(monkeypatch)

        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Merhaba!"
        mock_client.chat.completions.create.return_value = mock_response

        result = remote("test prompt")
        assert result == {"choices": [{"text": "Merhaba!"}]}
        mock_client.chat.completions.create.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"][1]["content"] == "test prompt"

    def test_null_content_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        remote, mock_client = _make_remote_callable(monkeypatch)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""  # OpenAI SDK handles nulls typically as empty string or we simulate empty
        mock_client.chat.completions.create.return_value = mock_response

        assert remote("test") == {"choices": [{"text": ""}]}

    def test_api_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        remote, mock_client = _make_remote_callable(monkeypatch)

        # Simulate API error
        mock_client.chat.completions.create.side_effect = openai.APIError("invalid request", request=MagicMock(), body={})

        assert remote("test") == {}

    def test_network_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        remote, mock_client = _make_remote_callable(monkeypatch)

        # Simulate network error
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())

        assert remote("test") == {}

    def test_remote_through_generate_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        remote, mock_client = _make_remote_callable(monkeypatch)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Ben üniversiteye gidiyorum."
        mock_client.chat.completions.create.return_value = mock_response

        # Enable remote backend flag
        monkeypatch.setattr(llm_utils, "_remote_backend_active", True)
        result = generate_reply(remote, "ben üniversite gitmek", language="tr")
        assert result == "Ben üniversiteye gidiyorum."

    def test_remote_model_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_utils.architecture.llm, "llm_remote_model", "test-model")
        monkeypatch.setattr(llm_utils.architecture.llm, "llm_remote_url", "https://x.com/v1")
        monkeypatch.setattr(llm_utils.architecture.llm, "llm_remote_api_key", "k")

        mock_openai_class = MagicMock()
        monkeypatch.setattr("openai.OpenAI", mock_openai_class)

        _, meta = llm_utils._make_remote_llm()
        assert meta["model"] == "test-model"


def test_load_qwen_model_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_utils.architecture.llm, "use_qwen_llm", False)
    assert llm_utils.load_qwen_model() is None


def test_load_qwen_model_returns_none_when_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils.architecture.llm, "use_qwen_llm", True)
    monkeypatch.setattr(llm_utils, "QWEN_MODEL_PATH", "/tmp/does-not-exist.gguf")
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: False)
    assert llm_utils.load_qwen_model() is None


def test_load_qwen_model_raises_when_gpu_forced_with_zero_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils.architecture.llm, "use_qwen_llm", True)
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: True)
    monkeypatch.setattr(llm_utils.architecture.llm, "qwen_force_gpu", True)
    monkeypatch.setattr(llm_utils.architecture.llm, "qwen_n_gpu_layers", 0)

    with pytest.raises(ValueError):
        llm_utils.load_qwen_model()


def test_load_qwen_model_returns_none_when_llama_cpp_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils.architecture.llm, "use_qwen_llm", True)
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: True)
    monkeypatch.setattr(llm_utils.architecture.llm, "qwen_force_gpu", False)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))

    assert llm_utils.load_qwen_model() is None


def test_load_qwen_model_builds_llama_with_expected_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_utils.architecture.llm, "use_qwen_llm", True)
    monkeypatch.setattr(llm_utils.os.path, "exists", lambda _: True)
    monkeypatch.setattr(llm_utils.architecture.llm, "qwen_force_gpu", False)
    monkeypatch.setattr(llm_utils, "QWEN_MODEL_PATH", "/tmp/model.gguf")
    monkeypatch.setattr(llm_utils.architecture.llm, "qwen_n_ctx", 4096)
    monkeypatch.setattr(llm_utils.architecture.llm, "qwen_n_gpu_layers", -1)
    monkeypatch.setattr(llm_utils.architecture.llm, "qwen_n_batch", 256)

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
