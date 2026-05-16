"""Dataset manager GUI launcher module."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication([])

try:
    from gui.ui.data_manager_window import run_data_manager
except ImportError:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from gui.ui.data_manager_window import run_data_manager


def main() -> None:
    """Start the dataset and model manager GUI."""
    print("Initializing dataset manager GUI...", flush=True)
    project_root = Path(__file__).resolve().parent
    run_data_manager(project_root=project_root)

    if QApplication.instance() is app:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
