"""
QtGraphs plotting backend for model evaluation.

Provides QtGraphs-based rendering for ROC curves and per-class metrics,
used as an alternative to the matplotlib backend when PySide6 is available.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path

import numpy as np

from config.architecture import architecture

logger = logging.getLogger(__name__)

EVALUATION_PLOT_BACKEND = (
    os.getenv("EVALUATION_PLOT_BACKEND", "matplotlib").strip().lower()
)

_QTGRAPHS_IMPORT_ERROR: Exception | None = None
try:
    from PySide6.QtCore import QMarginsF, QPointF, QSizeF, QUrl
    from PySide6.QtGui import (
        QColor,
        QImage,
        QPainter,
        QPageLayout,
        QPageSize,
        QPdfWriter,
    )
    from PySide6.QtGraphs import (
        QBarCategoryAxis,
        QBarSeries,
        QBarSet,
        QGraphsTheme,
        QLineSeries,
        QValueAxis,
    )
    from PySide6.QtQuickWidgets import QQuickWidget
    from PySide6.QtWidgets import QApplication

    QTGRAPHS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    QTGRAPHS_AVAILABLE = False
    _QTGRAPHS_IMPORT_ERROR = exc


def _qtgraphs_backend_enabled() -> bool:
    """Check whether the QtGraphs backend is active and usable."""
    if EVALUATION_PLOT_BACKEND in {"qtgraphs", "qtgraphs-first"}:
        return QTGRAPHS_AVAILABLE and QApplication.instance() is not None
    return False


def _ensure_qapplication() -> QApplication:
    """Return the active QApplication or raise if none exists."""
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QtGraphs export requires an active QApplication instance")
    return app


def _render_qtgraphs_widget_to_file(
    widget: QQuickWidget,
    save_path: str,
    dpi: int = architecture.evaluation.evaluation_dpi,
) -> None:
    """Render a QQuickWidget to a file (PDF or image)."""
    app = _ensure_qapplication()
    widget.show()
    app.processEvents()
    image = widget.grabFramebuffer()
    widget.hide()

    if image.isNull():
        raise RuntimeError("QtGraphs returned an empty framebuffer")

    output_path = Path(save_path)
    if output_path.suffix.lower() == ".pdf":
        writer = QPdfWriter(str(output_path))
        writer.setResolution(dpi)

        width_inches = max(1.0, image.width() / float(dpi))
        height_inches = max(1.0, image.height() / float(dpi))
        writer.setPageSize(
            QPageSize(QSizeF(width_inches, height_inches), QPageSize.Unit.Inch)
        )
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)

        painter = QPainter(writer)
        painter.drawImage(0, 0, image)
        painter.end()
    else:
        image.save(str(output_path))

    widget.deleteLater()


def _create_qtgraphs_line_widget(
    series_count: int, width: int, height: int
) -> QQuickWidget:
    """Create a QQuickWidget with a GraphsView containing line series."""
    widget = QQuickWidget()
    widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    widget.resize(width, height)

    series_defs = ",\n".join(
        f'        LineSeries {{ objectName: "series{idx}"; width: 2.0 }}'
        for idx in range(series_count)
    )
    qml = f"""import QtQuick 2.15
import QtGraphs

GraphsView {{
    objectName: "graphRoot"
    antialiasing: true
    clipPlotArea: true
    shadowVisible: false
    marginTop: 12
    marginBottom: 28
    marginLeft: 42
    marginRight: 16
    axisX: ValueAxis {{ objectName: "axisX"; min: 0; max: 1 }}
    axisY: ValueAxis {{ objectName: "axisY"; min: 0; max: 1 }}
    seriesList: [
{series_defs}
    ]
}}
"""

    qml_url = QUrl("data:text/plain," + urllib.parse.quote(qml))
    widget.setSource(qml_url)

    errors = widget.errors()
    if errors:
        joined = "; ".join(error.toString() for error in errors)
        raise RuntimeError(f"QtGraphs QML load failed: {joined}")

    return widget


def _create_qtgraphs_bar_widget(width: int, height: int) -> QQuickWidget:
    """Create a QQuickWidget with a GraphsView containing a bar series."""
    widget = QQuickWidget()
    widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    widget.resize(width, height)

    qml = """import QtQuick 2.15
import QtGraphs

GraphsView {
    objectName: "graphRoot"
    antialiasing: true
    clipPlotArea: true
    shadowVisible: false
    marginTop: 12
    marginBottom: 40
    marginLeft: 46
    marginRight: 18
    axisX: BarCategoryAxis { objectName: "axisX" }
    axisY: ValueAxis { objectName: "axisY"; min: 0; max: 1.1 }
    seriesList: [ BarSeries { objectName: "bars" } ]
}
"""

    qml_url = QUrl("data:text/plain," + urllib.parse.quote(qml))
    widget.setSource(qml_url)

    errors = widget.errors()
    if errors:
        joined = "; ".join(error.toString() for error in errors)
        raise RuntimeError(f"QtGraphs QML load failed: {joined}")

    return widget


def _apply_qtgraphs_theme(root) -> None:
    """Apply a white, clean theme to a QtGraphs GraphsView."""
    theme = root.property("theme")
    if isinstance(theme, QGraphsTheme):
        theme.setBackgroundVisible(True)
        theme.setBackgroundColor(QColor("#ffffff"))
        theme.setPlotAreaBackgroundVisible(True)
        theme.setPlotAreaBackgroundColor(QColor("#ffffff"))
        theme.setGridVisible(True)
        theme.setLabelTextColor(QColor("#111827"))
        theme.setLabelBackgroundVisible(False)
        theme.setLabelBorderVisible(False)


def _plot_roc_curves_qtgraphs(
    fpr: dict[int, np.ndarray],
    tpr: dict[int, np.ndarray],
    class_names: list[str],
    roc_auc: dict[int, float],
    save_path: str,
) -> None:
    """Plot multi-class ROC curves using the QtGraphs backend."""
    color_cycle = [
        "#2563eb",
        "#d97706",
        "#059669",
        "#dc2626",
        "#7c3aed",
        "#0891b2",
        "#b45309",
    ]

    widget = _create_qtgraphs_line_widget(
        series_count=len(class_names) + 1,
        width=1200,
        height=900,
    )
    root = widget.rootObject()
    if root is None:
        raise RuntimeError("QtGraphs ROC root object unavailable")

    _apply_qtgraphs_theme(root)

    axis_x = root.findChild(QValueAxis, "axisX")
    axis_y = root.findChild(QValueAxis, "axisY")
    if axis_x is None or axis_y is None:
        raise RuntimeError("QtGraphs ROC axis objects unavailable")

    axis_x.setRange(0.0, 1.0)
    axis_x.setTickInterval(0.2)
    axis_x.setLabelFormat("%.1f")
    axis_y.setRange(0.0, 1.05)
    axis_y.setTickInterval(0.2)
    axis_y.setLabelFormat("%.1f")

    for idx, class_name in enumerate(class_names):
        series = root.findChild(QLineSeries, f"series{idx}")
        if series is None:
            raise RuntimeError(f"QtGraphs ROC series missing: series{idx}")
        series.setName(f"{class_name} (AUC={roc_auc[idx]:.3f})")
        series.setColor(QColor(color_cycle[idx % len(color_cycle)]))

        for x_val, y_val in zip(fpr[idx].tolist(), tpr[idx].tolist()):
            series.append(QPointF(float(x_val), float(y_val)))

    random_series = root.findChild(QLineSeries, f"series{len(class_names)}")
    if random_series is None:
        raise RuntimeError("QtGraphs ROC random baseline series unavailable")
    random_series.setName("Random Classifier")
    random_series.setColor(QColor("#6b7280"))
    random_series.append(QPointF(0.0, 0.0))
    random_series.append(QPointF(1.0, 1.0))

    _render_qtgraphs_widget_to_file(widget, save_path)


def _plot_per_class_metrics_qtgraphs(
    class_names: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    save_path: str,
) -> None:
    """Plot per-class precision, recall, and F1-score using the QtGraphs backend."""
    widget = _create_qtgraphs_bar_widget(width=1400, height=840)
    root = widget.rootObject()
    if root is None:
        raise RuntimeError("QtGraphs metrics root object unavailable")

    _apply_qtgraphs_theme(root)

    axis_x = root.findChild(QBarCategoryAxis, "axisX")
    axis_y = root.findChild(QValueAxis, "axisY")
    bars = root.findChild(QBarSeries, "bars")

    if axis_x is None or axis_y is None or bars is None:
        raise RuntimeError("QtGraphs metrics objects unavailable")

    axis_x.setCategories([str(name) for name in class_names])
    axis_y.setRange(0.0, 1.1)
    axis_y.setTickInterval(0.2)
    axis_y.setLabelFormat("%.1f")

    precision_set = QBarSet("Precision")
    precision_set.append([float(v) for v in precision.tolist()])
    precision_set.setColor(QColor("#38bdf8"))

    recall_set = QBarSet("Recall")
    recall_set.append([float(v) for v in recall.tolist()])
    recall_set.setColor(QColor("#4ade80"))

    f1_set = QBarSet("F1-Score")
    f1_set.append([float(v) for v in f1.tolist()])
    f1_set.setColor(QColor("#f87171"))

    bars.append(precision_set)
    bars.append(recall_set)
    bars.append(f1_set)

    _render_qtgraphs_widget_to_file(widget, save_path)
