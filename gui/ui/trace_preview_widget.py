"""Trace preview widgets with QtGraphs-first rendering and matplotlib fallback.

The main factory widget TracePreviewWidget picks the best available backend
at runtime. Backend implementations live in sibling modules:

    _trace_qtgraphs.py   — _QtGraphsTracePreview (primary)
    _trace_matplotlib.py — _MatplotlibTracePreview (fallback)
"""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.ui.theme_manager import get_plot_palette

logger = logging.getLogger(__name__)

FLEX_AXES = ["flex0", "flex1", "flex2", "flex3", "flex4"]
ACCEL_AXES = ["accelX", "accelY", "accelZ"]
GYRO_AXES = ["gyroX", "gyroY", "gyroZ"]

DEFAULT_PLOT_PALETTE: dict[str, str] = get_plot_palette("dark")

_QTGRAPHS_IMPORT_ERROR: Exception | None = None
try:
    from PySide6.QtGraphs import QGraphsTheme, QLineSeries, QValueAxis
    from PySide6.QtQuickWidgets import QQuickWidget

    QTGRAPHS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    QTGRAPHS_AVAILABLE = False
    _QTGRAPHS_IMPORT_ERROR = exc

_MATPLOTLIB_IMPORT_ERROR: Exception | None = None
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    MATPLOTLIB_AVAILABLE = False
    _MATPLOTLIB_IMPORT_ERROR = exc

# Backend imports — after constants so sibling modules can import from us
# without triggering circular-import failures.
from gui.ui._trace_qtgraphs import _QtGraphsTracePreview  # noqa: E402
from gui.ui._trace_matplotlib import _MatplotlibTracePreview  # noqa: E402


class TracePreviewWidget(QWidget):
    """3-panel sensor trace viewer used by Data Manager preview panes."""

    def __init__(
        self, parent: QWidget | None = None, minimum_height: int = 300
    ) -> None:
        super().__init__(parent)
        self._plot_palette = dict(DEFAULT_PLOT_PALETTE)
        self._backend_name = "none"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        backend_widget: QWidget
        if QTGRAPHS_AVAILABLE:
            try:
                backend_widget = _QtGraphsTracePreview(
                    parent=self,
                    plot_palette=self._plot_palette,
                    minimum_height=minimum_height,
                )
                self._backend_name = "qtgraphs"
            except Exception as exc:
                logger.warning(
                    "QtGraphs trace preview unavailable, falling back to matplotlib: %s",
                    exc,
                )
                backend_widget = self._build_matplotlib_backend(minimum_height)
        else:
            backend_widget = self._build_matplotlib_backend(minimum_height)

        self._backend = backend_widget
        layout.addWidget(self._backend)

    @property
    def backend_name(self) -> str:
        """Return selected rendering backend name."""
        return self._backend_name

    def _build_matplotlib_backend(self, minimum_height: int) -> QWidget:
        if MATPLOTLIB_AVAILABLE:
            self._backend_name = "matplotlib"
            return _MatplotlibTracePreview(
                parent=self,
                plot_palette=self._plot_palette,
                minimum_height=minimum_height,
            )

        self._backend_name = "none"
        message = QLabel(
            "No plotting backend available. Install PySide6 QtGraphs or matplotlib."
        )
        message.setWordWrap(True)
        logger.error(
            "No trace preview backend available (qtgraphs=%s, matplotlib=%s)",
            _QTGRAPHS_IMPORT_ERROR,
            _MATPLOTLIB_IMPORT_ERROR,
        )
        return message

    def set_plot_palette(self, plot_palette: dict[str, str]) -> None:
        """Apply a color palette to the active backend."""
        self._plot_palette = dict(DEFAULT_PLOT_PALETTE)
        self._plot_palette.update(plot_palette)

        if hasattr(self._backend, "set_plot_palette"):
            self._backend.set_plot_palette(self._plot_palette)  # type: ignore[attr-defined]

    def plot_rows(self, rows: list[dict[str, float | int]]) -> None:
        """Render a trace preview from a list of row dictionaries."""
        if not rows:
            self.clear_plot()
            return
        frame = pd.DataFrame(rows)
        self.plot_dataframe(frame)

    def plot_dataframe(self, frame: pd.DataFrame) -> None:
        """Render trace preview from a dataframe."""
        if hasattr(self._backend, "plot_dataframe"):
            self._backend.plot_dataframe(frame)  # type: ignore[attr-defined]

    def clear_plot(self) -> None:
        """Clear all plotted traces."""
        if hasattr(self._backend, "clear_plot"):
            self._backend.clear_plot()  # type: ignore[attr-defined]
