"""QtGraphs-based trace preview backend.

Provides _QtGraphsTracePreview (3-panel QWidget) and _QtGraphsLinePanel
(single panel rendered via QQuickWidget with QLineSeries).
"""

from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Sequence

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, Qt, QUrl
from PySide6.QtGui import QColor
from PySide6.QtGraphs import QGraphsTheme, QLineSeries, QValueAxis
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.ui.trace_preview_widget import (
    ACCEL_AXES,
    DEFAULT_PLOT_PALETTE,
    FLEX_AXES,
    GYRO_AXES,
)

logger = logging.getLogger(__name__)


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

        # Legend label showing series color swatches and names
        self._legend_label = QLabel()
        self._legend_label.setObjectName("tracePanelLegend")
        self._legend_label.setWordWrap(True)
        self._legend_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._legend_label)

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

        # Build a robust rich-text legend using colored bullets and series names.
        legend_parts: list[str] = []
        for idx, key in enumerate(self._series_keys):
            color = line_colors[idx % len(line_colors)]
            legend_parts.append(
                f'<span style="color:{color}; font-weight:700;">\u25cf</span> {key}'
            )
        self._legend_label.setText("   ".join(legend_parts))
        self._legend_label.setStyleSheet(f"color: {self._plot_palette['text']};")

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

            if key not in frame.columns:
                series.clear()
                continue

            values = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
            if values.size == 0:
                series.clear()
                continue

            valid_indices = np.flatnonzero(np.isfinite(values))
            if valid_indices.size == 0:
                series.clear()
                continue

            x_max = max(x_max, int(values.size - 1))
            y_min = min(y_min, float(np.nanmin(values)))
            y_max = max(y_max, float(np.nanmax(values)))

            x_values = valid_indices.astype(float, copy=False)
            y_values = values[valid_indices].astype(float, copy=False)
            self._replace_series_points(series, x_values, y_values)

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

    @staticmethod
    def _replace_series_points(
        series: QLineSeries,
        x_values: np.ndarray,
        y_values: np.ndarray,
    ) -> None:
        replace_np = getattr(series, "replaceNp", None)
        if callable(replace_np):
            try:
                replace_np(x_values, y_values)
                return
            except (RuntimeError, TypeError, ValueError):
                pass

        points = [
            QPointF(float(x_value), float(y_value))
            for x_value, y_value in zip(x_values, y_values, strict=True)
        ]
        replace = getattr(series, "replace", None)
        if callable(replace):
            try:
                replace(points)
                return
            except (RuntimeError, TypeError, ValueError):
                pass

        series.clear()
        for point in points:
            series.append(point)

    def clear_plot(self) -> None:
        for series in self._series_map.values():
            series.clear()
        self._axis_x.setRange(0.0, 1.0)
        self._axis_x.setTickInterval(0.5)
        self._axis_y.setRange(0.0, 1.0)
        self._axis_y.setTickInterval(0.5)
        self._axis_y.setLabelFormat("%.2f")
