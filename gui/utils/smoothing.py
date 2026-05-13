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
    """Maintain a rolling window and return weighted-vote predictions."""

    window_size: int
    # Relative vote weight for REST tokens; lower values reduce REST influence.
    rest_weight: float = 0.75
    _window: Deque[tuple[str, float, bool]] = field(init=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=max(1, int(self.window_size)))

    def update(self, token: str, confidence: float = 1.0, is_rest: bool = False) -> str:
        """Push a token and return confidence+recency weighted vote."""
        clipped_conf = max(0.0, min(1.0, float(confidence)))
        self._window.append((token, clipped_conf, bool(is_rest)))
        voted = self._weighted_vote()
        return voted if voted is not None else token

    def _weighted_vote(self) -> str | None:
        if not self._window:
            return None

        weighted_scores: dict[str, float] = {}
        for idx, (token, confidence, is_rest) in enumerate(self._window):
            recency_weight = float(idx + 1)
            token_weight = recency_weight * confidence
            if is_rest:
                token_weight *= self.rest_weight
            weighted_scores[token] = weighted_scores.get(token, 0.0) + token_weight

        return max(weighted_scores.items(), key=lambda item: item[1])[0]

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
