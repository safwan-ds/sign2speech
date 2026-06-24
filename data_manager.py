"""Dataset manager GUI launcher module."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project package importable before any gui imports.
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication([])

try:
    from gui.ui.data_manager_window import run_data_manager
except ImportError:
    from gui.ui.data_manager_window import run_data_manager

from gui.utils.icon_utils import resolve_app_icon_path


def main() -> None:
    """Start the dataset and model manager GUI."""
    project_root = Path(__file__).resolve().parent
    icon_path = resolve_app_icon_path(project_root)
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))  # type: ignore
    run_data_manager(project_root=project_root)

    if QApplication.instance() is app:
        sys.exit(app.exec())  # type: ignore


if __name__ == "__main__":
    main()
