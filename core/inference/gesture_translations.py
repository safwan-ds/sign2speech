import logging
import os
from dataclasses import dataclass

from config.config import BASE_DIR

logger = logging.getLogger(__name__)

GESTURES_PATH = os.path.join(BASE_DIR, "config", "gestures.txt")


@dataclass(slots=True)
class GestureTransitionStateMachine:
    """Require a verified REST transition before accepting a new active gesture."""

    rest_token: str = "REST"
    min_rest_frames: int = 1
    active_token: str | None = None
    rest_frames: int = 0

    def observe(self, token: str, *, valid: bool = True) -> tuple[str, bool, bool]:
        """Return (resolved_token, accepted_transition, is_rest)."""
        normalized = token.strip().upper()
        if not normalized:
            return self.active_token or "", False, False

        is_rest = normalized == self.rest_token
        if is_rest:
            if valid:
                self.rest_frames += 1
                if self.rest_frames >= max(1, self.min_rest_frames):
                    self.active_token = None
                    return self.rest_token, True, True
            return self.active_token or self.rest_token, False, True

        if not valid:
            self.rest_frames = 0
            return self.active_token or normalized, False, False

        if self.active_token is None:
            self.active_token = normalized
            self.rest_frames = 0
            return normalized, True, False

        if normalized == self.active_token:
            self.rest_frames = 0
            return normalized, False, False

        # A direct active->active switch is held until REST verifies a break.
        self.rest_frames = 0
        return self.active_token, False, False


def load_gesture_translations(path: str = GESTURES_PATH):
    translations: dict[str, str] = {}
    if not os.path.exists(path):
        logger.warning(f"Warning: Gestures file not found at {path}")
        return translations

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "-" in line:
                original, translated = [part.strip() for part in line.split("-", 1)]
                if not original:
                    continue
                translations[original.lower()] = translated

    return translations


def translate_gesture(
    gesture: str,
    translations: dict[str, str],
    target_language: str = "tr",
) -> str:
    normalized = gesture.strip()
    if not normalized:
        return ""

    if target_language.lower() == "en":
        return normalized.replace("_", " ")

    translated = translations.get(normalized.lower(), "").strip()
    if translated:
        return translated
    return normalized.replace("_", " ")


def translate_gestures(
    gestures: list[str],
    translations: dict[str, str],
    target_language: str = "tr",
) -> list[str]:
    return [
        translate_gesture(gesture, translations, target_language=target_language)
        for gesture in gestures
    ]
