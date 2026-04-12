"""GUI launcher module."""

from __future__ import annotations

from pathlib import Path
import sys

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
    project_root = Path(__file__).resolve().parents[1]
    run_dashboard(project_root=project_root)


if __name__ == "__main__":
    main()
