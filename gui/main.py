"""GUI launcher module."""

from __future__ import annotations

from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication([])

try:
    from .ui.app_window import run_dashboard
except ImportError:
    # Support running this file directly: python gui/main.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from gui.ui.app_window import run_dashboard


def main() -> None:
    """Start the desktop dashboard."""
    print("Initializing Sign Language Glove GUI...", flush=True)
    project_root = Path(__file__).resolve().parents[1]
    print(f"Project root: {project_root}", flush=True)
    print("Starting dashboard...", flush=True)
    run_dashboard(project_root=project_root)

    print("GUI started successfully.", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
