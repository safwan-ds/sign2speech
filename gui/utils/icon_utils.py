"""Helpers for resolving and applying application icons."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget

_ICON_CANDIDATES = (
    Path("assets") / "app.ico",
    Path("assets") / "app.png",
    Path("assets") / "icon.ico",
    Path("assets") / "icon.png",
)


def resolve_app_icon_path(project_root: str | Path) -> Path | None:
    """Return the first existing icon path under the project root, if any."""
    root = Path(project_root)
    for relative in _ICON_CANDIDATES:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def apply_app_icon(target: QWidget, project_root: str | Path) -> None:
    """Apply the app icon to a Qt widget/window when an icon file exists."""
    icon_path = resolve_app_icon_path(project_root)
    if icon_path is None:
        return
    target.setWindowIcon(QIcon(str(icon_path)))
