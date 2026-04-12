"""Formatting helpers for GUI display and export."""

from __future__ import annotations

from datetime import datetime


def now_hms() -> str:
    """Return the current local time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")


def now_stamp() -> str:
    """Return a timestamp suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def percent(value: float | None) -> str:
    """Format confidence float into percentage text."""
    if value is None:
        return "0.0%"
    return f"{value * 100:.1f}%"
