import logging
import os

from config.config import BASE_DIR

logger = logging.getLogger(__name__)

GESTURES_PATH = os.path.join(BASE_DIR, "config", "gestures.txt")


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
