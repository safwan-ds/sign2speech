"""Tests for instant/llm/hybrid TTS routing in dashboard events."""

from gui.ui.app_window_events import AppWindowEventMixin


class _FakeLabel:
    def __init__(self) -> None:
        self.value = ""

    def setText(self, text: str) -> None:
        self.value = text


class _FakeTextBox:
    def __init__(self) -> None:
        self.value = ""

    def setPlainText(self, text: str) -> None:
        self.value = text

    def toPlainText(self) -> str:
        return self.value


class _FakeTTSService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def speak(self, text: str, language: str, backend: str = "edge") -> None:
        self.calls.append((text, language, backend))


class _DummyWindow(AppWindowEventMixin):
    def __init__(self, tts_mode: str) -> None:
        self.current_sentence_tokens: list[str] = []
        self.tts_enabled = True
        self.tts_mode = tts_mode
        self.ui_language = "tr"
        self.tts_service = _FakeTTSService()
        self.word_count_label = _FakeLabel()
        self.sentence_box = _FakeTextBox()
        self.refined_box = _FakeTextBox()
        self.progress_state = "idle"

    def _format_word_count(self, count: int) -> str:
        return f"Word Count: {count}"

    def _effective_llm_language(self) -> str:
        return "en"

    def _set_llm_progress_state(self, state: str) -> None:
        self.progress_state = state


def test_hybrid_mode_routes_sentence_local_and_llm_edge() -> None:
    window = _DummyWindow(tts_mode="hybrid")

    window._on_sentence({"token": "merhaba", "sentence": "merhaba"})
    window._on_llm_text({"text": "hello world"})

    assert window.tts_service.calls == [
        ("merhaba", "tr", "local"),
        ("hello world", "en", "edge"),
    ]


def test_instant_mode_only_speaks_sentence_with_local_backend() -> None:
    window = _DummyWindow(tts_mode="instant")

    window._on_sentence({"token": "merhaba", "sentence": "merhaba"})
    window._on_llm_text({"text": "hello world"})

    assert window.tts_service.calls == [("merhaba", "tr", "local")]


def test_llm_mode_only_speaks_refined_text_with_edge_backend() -> None:
    window = _DummyWindow(tts_mode="llm")

    window._on_sentence({"token": "merhaba", "sentence": "merhaba"})
    window._on_llm_text({"text": "hello world"})

    assert window.tts_service.calls == [("hello world", "en", "edge")]
