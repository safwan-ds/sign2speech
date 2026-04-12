import os
import logging
from config import BASE_DIR

logger = logging.getLogger(__name__)

GESTURES_PATH = os.path.join(BASE_DIR, "config", "gestures.txt")


def load_gesture_translations(path: str = GESTURES_PATH):
    translations: dict[str, str | None] = {}
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
                if translated:
                    translations[original.lower()] = translated
                else:
                    translations[original.lower()] = None
            else:
                translations[line.lower()] = None

    return translations


def translate_gesture(gesture: str, translations: dict[str, str | None]) -> str:
    translated = translations.get(gesture.lower())
    if translated:
        return translated
    return gesture.replace("_", " ")


def translate_gestures(
    gestures: list[str], translations: dict[str, str | None]
) -> list[str]:
    return [translate_gesture(gesture, translations) for gesture in gestures]
