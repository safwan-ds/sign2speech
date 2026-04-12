"""Export helpers for sentence/session data."""

from __future__ import annotations

from pathlib import Path

from gui.utils.formatting import now_stamp


def export_sentence(sentence: str, export_dir: Path) -> Path:
    """Write sentence text to a timestamped file and return its path."""
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / f"sentence_{now_stamp()}.txt"
    target.write_text(sentence.strip() + "\n", encoding="utf-8")
    return target
