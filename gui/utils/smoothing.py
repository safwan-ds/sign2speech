"""Prediction smoothing and sentence debouncing utilities."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque


def majority_vote(window: Deque[str]) -> str | None:
    """Return the most frequent token in a window, or None when empty."""
    if not window:
        return None

    counts = Counter(window)
    token, _ = counts.most_common(1)[0]
    return token


@dataclass(slots=True)
class PredictionSmoother:
    """Maintain a rolling window and return majority-vote predictions."""

    window_size: int
    _window: Deque[str] = field(init=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=max(1, int(self.window_size)))

    def update(self, token: str) -> str:
        """Push a token and return the smoothed token."""
        self._window.append(token)
        voted = majority_vote(self._window)
        return voted if voted is not None else token

    def reset(self) -> None:
        """Clear smoothing state."""
        self._window.clear()


@dataclass(slots=True)
class SentenceAssembler:
    """Build sentence text while debouncing repeated tokens."""

    debounce_window: int = 3
    tokens: list[str] = field(default_factory=list)
    _last_token: str | None = None
    _repeat_count: int = 0

    def try_append(self, token: str) -> bool:
        """Append token if it is not a debounced duplicate."""
        if not token:
            return False

        if token == self._last_token:
            self._repeat_count += 1
            if self._repeat_count < self.debounce_window:
                return False
            self._repeat_count = 0
            return False

        self._last_token = token
        self._repeat_count = 0
        self.tokens.append(token)
        return True

    def clear(self) -> None:
        """Clear all sentence state."""
        self.tokens.clear()
        self._last_token = None
        self._repeat_count = 0

    def text(self) -> str:
        """Return sentence text as a space-joined string."""
        return " ".join(self.tokens)
