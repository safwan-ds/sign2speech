"""Trace preview widgets with QtGraphs-first rendering and matplotlib fallback."""

from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Sequence

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QUrl
from PySide6.QtGui import QColor
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
        """Render trace preview from a list of row dictionaries."""
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


class _QtGraphsTracePreview(QWidget):
    """QtGraphs implementation backed by three GraphsView panels."""

    def __init__(
        self,
        parent: QWidget | None,
        plot_palette: dict[str, str],
        minimum_height: int,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        panel_height = max(120, minimum_height // 3)
        self._flex_panel = _QtGraphsLinePanel(
            title="Flex Sensors",
            series_keys=FLEX_AXES,
            min_height=panel_height,
        )
        self._accel_panel = _QtGraphsLinePanel(
            title="Accelerometer",
            series_keys=ACCEL_AXES,
            min_height=panel_height,
        )
        self._gyro_panel = _QtGraphsLinePanel(
            title="Gyroscope",
            series_keys=GYRO_AXES,
            min_height=panel_height,
        )

        layout.addWidget(self._flex_panel, stretch=1)
        layout.addWidget(self._accel_panel, stretch=1)
        layout.addWidget(self._gyro_panel, stretch=1)

        self.set_plot_palette(plot_palette)

    def set_plot_palette(self, plot_palette: dict[str, str]) -> None:
        for panel in (self._flex_panel, self._accel_panel, self._gyro_panel):
            panel.set_plot_palette(plot_palette)

    def plot_dataframe(self, frame: pd.DataFrame) -> None:
        self._flex_panel.plot_dataframe(frame)
        self._accel_panel.plot_dataframe(frame)
        self._gyro_panel.plot_dataframe(frame)

    def clear_plot(self) -> None:
        self._flex_panel.clear_plot()
        self._accel_panel.clear_plot()
        self._gyro_panel.clear_plot()


class _QtGraphsLinePanel(QWidget):
    """Single QtGraphs line panel rendered via QQuickWidget."""

    def __init__(
        self,
        title: str,
        series_keys: Sequence[str],
        min_height: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._series_keys = list(series_keys)
        self._plot_palette = dict(DEFAULT_PLOT_PALETTE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setObjectName("tracePanelTitle")
        layout.addWidget(self._title_label)

        self._quick_widget = QQuickWidget(self)
        self._quick_widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self._quick_widget.setMinimumHeight(min_height)
        layout.addWidget(self._quick_widget, stretch=1)

        self._load_graph()

    def _load_graph(self) -> None:
        series_defs = ",\n".join(
            f'        LineSeries {{ objectName: "series{idx}"; width: 1.6 }}'
            for idx in range(len(self._series_keys))
        )
        qml = f"""import QtQuick 2.15
import QtGraphs

GraphsView {{
    objectName: "graphRoot"
    antialiasing: true
    clipPlotArea: true
    shadowVisible: false
    marginTop: 6
    marginBottom: 12
    marginLeft: 40
    marginRight: 10
    axisX: ValueAxis {{
        objectName: "axisX"
        min: 0
        max: 1
    }}
    axisY: ValueAxis {{
        objectName: "axisY"
        min: 0
        max: 1
    }}
    seriesList: [
{series_defs}
    ]
}}
"""
        qml_url = QUrl("data:text/plain," + urllib.parse.quote(qml))
        self._quick_widget.setSource(qml_url)

        errors = self._quick_widget.errors()
        if errors:
            joined = "; ".join(error.toString() for error in errors)
            raise RuntimeError(f"QtGraphs QML load failed: {joined}")

        root = self._quick_widget.rootObject()
        if root is None:
            raise RuntimeError("QtGraphs root object is unavailable")

        self._root = root
        self._theme = root.property("theme")
        self._axis_x = root.findChild(QValueAxis, "axisX")
        self._axis_y = root.findChild(QValueAxis, "axisY")

        if self._axis_x is None or self._axis_y is None:
            raise RuntimeError("QtGraphs axis objects are unavailable")

        self._series_map: dict[str, QLineSeries] = {}
        for idx, key in enumerate(self._series_keys):
            series = root.findChild(QLineSeries, f"series{idx}")
            if series is None:
                raise RuntimeError(f"QtGraphs line series missing: series{idx}")
            self._series_map[key] = series

    def set_plot_palette(self, plot_palette: dict[str, str]) -> None:
        self._plot_palette = dict(DEFAULT_PLOT_PALETTE)
        self._plot_palette.update(plot_palette)

        grid_color = QColor(self._plot_palette["grid"])
        grid_color.setAlpha(70)

        self._quick_widget.setClearColor(QColor(self._plot_palette["figure_bg"]))
        self._quick_widget.setStyleSheet(
            f"border: 1px solid {self._plot_palette['spine']}; border-radius: 6px;"
        )
        self._title_label.setStyleSheet(
            f"color: {self._plot_palette['text']}; font-weight: 600;"
        )

        if isinstance(self._theme, QGraphsTheme):
            self._theme.setBackgroundVisible(True)
            self._theme.setBackgroundColor(QColor(self._plot_palette["figure_bg"]))
            self._theme.setPlotAreaBackgroundVisible(True)
            self._theme.setPlotAreaBackgroundColor(
                QColor(self._plot_palette["axes_bg"])
            )
            # Keep traces clear by disabling the heavy default grid overlay.
            self._theme.setGridVisible(False)
            self._theme.setLabelTextColor(QColor(self._plot_palette["text"]))
            self._theme.setLabelBackgroundVisible(False)
            self._theme.setLabelBorderVisible(False)

        self._axis_x.setColor(QColor(self._plot_palette["spine"]))
        self._axis_x.setSubColor(grid_color)
        self._axis_x.setTitleColor(QColor(self._plot_palette["text"]))
        self._axis_x.setLabelsVisible(True)
        self._axis_x.setSubTickCount(0)

        self._axis_y.setColor(QColor(self._plot_palette["spine"]))
        self._axis_y.setSubColor(grid_color)
        self._axis_y.setTitleColor(QColor(self._plot_palette["text"]))
        self._axis_y.setLabelsVisible(True)
        self._axis_y.setSubTickCount(0)

        line_colors = [
            self._plot_palette["line_a"],
            self._plot_palette["line_b"],
            self._plot_palette["line_c"],
            self._plot_palette["line_d"],
            self._plot_palette["line_e"],
        ]
        for idx, key in enumerate(self._series_keys):
            series = self._series_map[key]
            series.setColor(QColor(line_colors[idx % len(line_colors)]))
            series.setWidth(1.6)

    def plot_dataframe(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            self.clear_plot()
            return

        self._axis_x.setLabelFormat("%.0f")
        self._axis_y.setLabelFormat("%.2f")

        y_min = np.inf
        y_max = -np.inf
        x_max = 1

        for key in self._series_keys:
            series = self._series_map[key]
            series.clear()

            if key not in frame.columns:
                continue

            values = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
            if values.size == 0:
                continue

            valid_indices = np.flatnonzero(np.isfinite(values))
            if valid_indices.size == 0:
                continue

            x_max = max(x_max, int(values.size - 1))
            y_min = min(y_min, float(np.nanmin(values)))
            y_max = max(y_max, float(np.nanmax(values)))

            for idx in valid_indices.tolist():
                series.append(QPointF(float(idx), float(values[idx])))

        self._axis_x.setRange(0.0, float(x_max))
        self._axis_x.setTickInterval(max(1.0, float(x_max) / 4.0))

        if not np.isfinite(y_min) or not np.isfinite(y_max):
            self._axis_y.setRange(0.0, 1.0)
            self._axis_y.setTickInterval(0.5)
            return

        if abs(y_max - y_min) < 1e-9:
            pad = 1.0 if y_max == 0 else abs(y_max) * 0.05
        else:
            pad = (y_max - y_min) * 0.08

        lower = float(y_min - pad)
        upper = float(y_max + pad)

        self._axis_y.setRange(lower, upper)
        self._axis_y.setTickInterval(max((upper - lower) / 4.0, 1e-3))

        if max(abs(lower), abs(upper)) >= 1000:
            self._axis_y.setLabelFormat("%.0f")

    def clear_plot(self) -> None:
        for series in self._series_map.values():
            series.clear()
        self._axis_x.setRange(0.0, 1.0)
        self._axis_x.setTickInterval(0.5)
        self._axis_y.setRange(0.0, 1.0)
        self._axis_y.setTickInterval(0.5)
        self._axis_y.setLabelFormat("%.2f")


class _MatplotlibTracePreview(QWidget):
    """Matplotlib fallback backend used when QtGraphs cannot be initialized."""

    def __init__(
        self,
        parent: QWidget | None,
        plot_palette: dict[str, str],
        minimum_height: int,
    ) -> None:
        super().__init__(parent)
        self._plot_palette = dict(DEFAULT_PLOT_PALETTE)
        self._plot_palette.update(plot_palette)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._figure = Figure(figsize=(9.4, 4.6), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(minimum_height)
        layout.addWidget(self._canvas)

        self.set_plot_palette(self._plot_palette)
        self.clear_plot()

    def set_plot_palette(self, plot_palette: dict[str, str]) -> None:
        self._plot_palette = dict(DEFAULT_PLOT_PALETTE)
        self._plot_palette.update(plot_palette)
        self._canvas.setStyleSheet(
            f"border: 1px solid {self._plot_palette['grid']}; border-radius: 6px;"
        )
        self._figure.patch.set_facecolor(self._plot_palette["figure_bg"])
        self._canvas.draw_idle()

    def plot_dataframe(self, frame: pd.DataFrame) -> None:
        self._figure.clear()
        self._figure.patch.set_facecolor(self._plot_palette["figure_bg"])

        ax1 = self._figure.add_subplot(311)
        ax2 = self._figure.add_subplot(312)
        ax3 = self._figure.add_subplot(313)

        line_colors = [
            self._plot_palette["line_a"],
            self._plot_palette["line_b"],
            self._plot_palette["line_c"],
            self._plot_palette["line_d"],
            self._plot_palette["line_e"],
        ]

        for axis in (ax1, ax2, ax3):
            axis.set_facecolor(self._plot_palette["axes_bg"])
            axis.tick_params(colors=self._plot_palette["text"])
            axis.grid(alpha=0.12, linewidth=0.6, color=self._plot_palette["grid"])
            for spine in axis.spines.values():
                spine.set_color(self._plot_palette["spine"])

        for idx, col in enumerate(FLEX_AXES):
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                    dtype=float
                )
                ax1.plot(
                    values, linewidth=1.4, color=line_colors[idx % len(line_colors)]
                )
        ax1.set_title("Flex Sensors", color=self._plot_palette["text"])

        for idx, col in enumerate(ACCEL_AXES):
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                    dtype=float
                )
                ax2.plot(
                    values, linewidth=1.4, color=line_colors[idx % len(line_colors)]
                )
        ax2.set_title("Accelerometer", color=self._plot_palette["text"])

        for idx, col in enumerate(GYRO_AXES):
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                    dtype=float
                )
                ax3.plot(
                    values, linewidth=1.4, color=line_colors[idx % len(line_colors)]
                )
        ax3.set_title("Gyroscope", color=self._plot_palette["text"])

        self._figure.tight_layout()
        self._canvas.draw_idle()

    def clear_plot(self) -> None:
        self.plot_dataframe(pd.DataFrame())
