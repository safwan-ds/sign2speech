"""Shared helpers for gesture recording scripts."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

from config import LOGS_DIR

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


def gesture_output_dir(gesture_label: str, base_dir: str = LOGS_DIR) -> Path:
    """Return gesture output directory under data/raw."""
    return Path(base_dir) / gesture_label


def build_recording_file_path(gesture_label: str, base_dir: str = LOGS_DIR) -> Path:
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
    """Load gesture names from config/gestures.txt, preserving file order."""
    root = Path(project_root)
    gestures_file = root / "config" / "gestures.txt"
    if not gestures_file.exists():
        return []

    names: list[str] = []
    with gestures_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            name = line.split(" - ", 1)[0].strip()
            if name:
                names.append(name)
    return names


def count_csv_samples(gesture_label: str, base_dir: str = LOGS_DIR) -> int:
    """Count saved samples for one gesture folder."""
    target_dir = gesture_output_dir(gesture_label, base_dir)
    if not target_dir.exists():
        return 0
    return len([name for name in os.listdir(target_dir) if name.endswith(".csv")])
