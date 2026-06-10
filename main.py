"""GUI launcher module."""

from __future__ import annotations

import sys
from pathlib import Path

# Must run before any project imports so utils/core/etc. are always resolvable,
# regardless of whether this file is invoked as a script or as part of the package.
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

app = QApplication.instance()
if app is None:
    app = QApplication([])

from gui.ui.app_window import run_dashboard
from gui.utils.icon_utils import resolve_app_icon_path


def main() -> None:
    """Start the desktop dashboard."""
    print("Initializing Sign2Speech GUI...", flush=True)
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}", flush=True)
    icon_path = resolve_app_icon_path(project_root)
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))  # type: ignore
    print("Starting dashboard...", flush=True)
    run_dashboard(project_root=project_root)

    print("GUI started successfully.", flush=True)
    sys.exit(app.exec())  # type: ignore


if __name__ == "__main__":
    main()
