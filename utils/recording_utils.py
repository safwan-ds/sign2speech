"""Shared helpers for gesture recording scripts."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

from config.config import RAW_DATA_DIR

SENSOR_COLUMNS = [
    "flex0",
    "flex1",
    "flex2",
    "flex3",
    "flex4",
    "accelX",
    "accelY",
    "accelZ",
    "gyroX",
    "gyroY",
    "gyroZ",
]

CSV_COLUMNS = ["t_ms", *SENSOR_COLUMNS]


def sanitize_gesture_label(label: str) -> str:
    """Normalize user-provided gesture label for folder/file usage."""
    normalized = label.strip().replace(" ", "_")
    normalized = re.sub(r"[^A-Za-z0-9_\-]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized or "gesture"


def gesture_output_dir(gesture_label: str, base_dir: str = RAW_DATA_DIR) -> Path:
    """Return gesture output directory under data/raw."""
    return Path(base_dir) / gesture_label


def build_recording_file_path(gesture_label: str, base_dir: str = RAW_DATA_DIR) -> Path:
    """Create a timestamped file path for a new recording."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        gesture_output_dir(gesture_label, base_dir) / f"{gesture_label}_{timestamp}.csv"
    )


def build_recording_metadata_path(recording_path: str | Path) -> Path:
    """Create a sidecar metadata path for a recording file."""
    return Path(recording_path).with_suffix(".meta.json")


def save_rows_to_csv(file_path: str | Path, rows: list[dict[str, float | int]]) -> Path:
    """Save recording rows using a consistent schema."""
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def save_recording_metadata(file_path: str | Path, metadata: dict[str, object]) -> Path:
    """Save recording metadata as a sidecar JSON file."""
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return target


def load_gesture_names(project_root: str | Path) -> list[str]:
    """Load gesture names from config/gestures.json (preferred) or gestures.txt.

    The JSON format is an array of objects: [{"name": "hello", "translation": "merhaba"}, ...]
    For backwards compatibility a plain text file where each line is
    "name - translation" will also be read.
    """
    root = Path(project_root)
    json_file = root / "config" / "gestures.json"
    txt_file = root / "config" / "gestures.txt"

    names: list[str] = []
    if json_file.exists():
        try:
            with json_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            # Support either dict mapping or list of objects
            if isinstance(data, dict):
                for name in data.keys():
                    if name:
                        names.append(str(name))
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and entry.get("name"):
                        names.append(str(entry.get("name")))
        except Exception:
            # Fall back to txt parser on error
            pass

    if not names and txt_file.exists():
        with txt_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                name = line.split(" - ", 1)[0].strip()
                if name:
                    names.append(name)

    return names


def load_gestures(project_root: str | Path) -> list[dict[str, str]]:
    """Load gestures with optional translations.

    Returns a list of dicts [{"name": ..., "translation": ...}, ...].
    """
    root = Path(project_root)
    json_file = root / "config" / "gestures.json"
    txt_file = root / "config" / "gestures.txt"

    entries: list[dict[str, str]] = []
    if json_file.exists():
        try:
            with json_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                for k, v in data.items():
                    entries.append({"name": str(k), "translation": str(v)})
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("name")
                        trans = item.get("translation", "")
                        if name:
                            entries.append(
                                {"name": str(name), "translation": str(trans)}
                            )
        except Exception:
            # ignore and try txt fallback
            entries = []

    if not entries and txt_file.exists():
        with txt_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "-" in line:
                    original, translated = [part.strip() for part in line.split("-", 1)]
                else:
                    original, translated = line, ""
                if original:
                    entries.append({"name": original, "translation": translated})

    return entries


def save_gestures(project_root: str | Path, entries: list[dict[str, str]]) -> Path:
    """Save the gestures list to config/gestures.json.

    Entries should be a list of dicts with keys 'name' and optional 'translation'.
    """
    root = Path(project_root)
    json_file = root / "config" / "gestures.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    # Normalize entries to list of objects
    to_write: list[dict[str, str]] = []
    for e in entries:
        name = str(e.get("name", "")).strip()
        translation = str(e.get("translation", "")).strip()
        if not name:
            continue
        to_write.append({"name": name, "translation": translation})

    with json_file.open("w", encoding="utf-8") as handle:
        json.dump(to_write, handle, ensure_ascii=False, indent=2)
    return json_file


def count_csv_samples(gesture_label: str, base_dir: str = RAW_DATA_DIR) -> int:
    """Count saved samples for one gesture folder."""
    target_dir = gesture_output_dir(gesture_label, base_dir)
    if not target_dir.exists():
        return 0
    return len([name for name in os.listdir(target_dir) if name.endswith(".csv")])
