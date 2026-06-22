"""Matplotlib-based trace preview backend (fallback).

Provides _MatplotlibTracePreview — used when QtGraphs cannot be initialized.
"""

from __future__ import annotations

import logging

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from gui.ui.trace_preview_widget import (
    ACCEL_AXES,
    DEFAULT_PLOT_PALETTE,
    FLEX_AXES,
    GYRO_AXES,
)

logger = logging.getLogger(__name__)


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
                    values,
                    linewidth=1.4,
                    color=line_colors[idx % len(line_colors)],
                    label=col,
                )
        ax1.set_title("Flex Sensors", color=self._plot_palette["text"])
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            leg = ax1.legend(loc="upper right", fontsize=8)
            leg.get_frame().set_facecolor(self._plot_palette["axes_bg"])
            leg.get_frame().set_edgecolor(self._plot_palette["spine"])
            for text in leg.get_texts():
                text.set_color(self._plot_palette["text"])

        for idx, col in enumerate(ACCEL_AXES):
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                    dtype=float
                )
                ax2.plot(
                    values,
                    linewidth=1.4,
                    color=line_colors[idx % len(line_colors)],
                    label=col,
                )
        ax2.set_title("Accelerometer", color=self._plot_palette["text"])
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            leg = ax2.legend(loc="upper right", fontsize=8)
            leg.get_frame().set_facecolor(self._plot_palette["axes_bg"])
            leg.get_frame().set_edgecolor(self._plot_palette["spine"])
            for text in leg.get_texts():
                text.set_color(self._plot_palette["text"])

        for idx, col in enumerate(GYRO_AXES):
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                    dtype=float
                )
                ax3.plot(
                    values,
                    linewidth=1.4,
                    color=line_colors[idx % len(line_colors)],
                    label=col,
                )
        ax3.set_title("Gyroscope", color=self._plot_palette["text"])
        handles, labels = ax3.get_legend_handles_labels()
        if handles:
            leg = ax3.legend(loc="upper right", fontsize=8)
            leg.get_frame().set_facecolor(self._plot_palette["axes_bg"])
            leg.get_frame().set_edgecolor(self._plot_palette["spine"])
            for text in leg.get_texts():
                text.set_color(self._plot_palette["text"])

        self._figure.tight_layout()
        self._canvas.draw_idle()

    def clear_plot(self) -> None:
        self.plot_dataframe(pd.DataFrame())
