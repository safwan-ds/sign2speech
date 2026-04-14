"""Localized UI strings for the Sign2Speech dashboard."""

from __future__ import annotations

import json
from pathlib import Path


def _load_language_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in loaded.items()}


def _build_localization() -> dict[str, dict[str, str]]:
    base = Path(__file__).resolve().parent / "locales"
    return {
        "tr": _load_language_file(base / "tr.json"),
        "en": _load_language_file(base / "en.json"),
    }


LOCALIZATION: dict[str, dict[str, str]] = _build_localization()
